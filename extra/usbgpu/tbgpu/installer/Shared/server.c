#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <errno.h>
#include <dispatch/dispatch.h>
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#include <IOKit/IOMessage.h>
#include <mach/mach.h>

// Protocol

enum {
  CMD_PROBE = 0,          // probe devices, returns count
  CMD_MAP_BAR = 1,        // map PCI BAR, returns size
  CMD_MAP_SYSMEM_FD = 2,  // alloc DMA memory, returns fd via SCM_RIGHTS
  CMD_CFG_READ = 3,       // read PCI config space
  CMD_CFG_WRITE = 4,      // write PCI config space
  CMD_RESET = 5,          // reset device
  CMD_MMIO_READ = 6,      // bulk read from BAR
  CMD_MMIO_WRITE = 7,     // bulk write to BAR
  CMD_MAP_SYSMEM = 8,     // map system memory
  CMD_SYSMEM_READ = 9,    // bulk read from system memory
  CMD_SYSMEM_WRITE = 10,  // bulk write to system memory
  CMD_RESIZE_BAR = 11,    // resize bar (noop)
  CMD_HANDSHAKE = 15, CMD_LEASE_ACQUIRE = 16, CMD_LEASE_RELEASE = 17, CMD_KEEPALIVE_STATUS = 18,
  RESP_OK = 0, RESP_ERR = 1, RESP_UNSUPPORTED_VERSION = 2, RESP_UNSUPPORTED_CAPABILITY = 3,
  RESP_MALFORMED_REQUEST = 4, RESP_INVALID_STATE = 5, RESP_BUSY = 6, RESP_INTERNAL_ERROR = 7,
  RESP_PROVIDER_UNAVAILABLE = 8, RESP_DEVICE_LOST = 9, RESP_NATIVE_ERROR = 10,
};

typedef struct { uint8_t cmd; uint32_t dev_id, bar; uint64_t arg0, arg1, arg2; } request_t;
typedef struct { uint8_t status; uint64_t resp0, resp1; } response_t;

// Constants and state

#define BULK_BUF_SIZE (64 << 20)
#define MAX_BARS 6
#define MAX_SYSMEM 128

typedef struct {
  int fd, handshaken, cleaned;
  uint64_t lease;
  io_connect_t conn;
  uint8_t *bulk;
  struct { mach_vm_address_t addr; mach_vm_size_t size; } bars[MAX_BARS];
  struct { mach_vm_address_t addr; mach_vm_size_t size; int shm_fd; char shm_name[64]; } sysmem[MAX_SYSMEM];
  int sysmem_count;
} client_session_t;
static volatile int g_client_active;
static volatile int g_provider_terminated;
static IONotificationPortRef g_notification_port;
static io_object_t g_service_notification;

// Utilities

static uint32_t get_le32(const uint8_t *p) { return (uint32_t)p[0] | (uint32_t)p[1]<<8 | (uint32_t)p[2]<<16 | (uint32_t)p[3]<<24; }
static uint16_t get_le16(const uint8_t *p) { return (uint16_t)p[0] | (uint16_t)p[1]<<8; }
static uint64_t get_le64(const uint8_t *p) { uint64_t v=0; for (unsigned i=0;i<8;i++) v |= (uint64_t)p[i] << (8*i); return v; }
static void put_le16(uint8_t *p, uint16_t v) { p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); }
static void put_le32(uint8_t *p, uint32_t v) { for (unsigned i=0;i<4;i++) p[i]=(uint8_t)(v>>(8*i)); }
static void put_le64(uint8_t *p, uint64_t v) { for (unsigned i=0;i<8;i++) p[i]=(uint8_t)(v>>(8*i)); }
static int recvall(int fd, void *buf, size_t len) { for (size_t off=0; off<len;) { ssize_t r=recv(fd,(uint8_t*)buf+off,len-off,0); if (r<0 && errno==EINTR) continue; if (r<=0) return -1; off+=(size_t)r; } return 0; }
static int sendall(int fd, const void *buf, size_t len) { for (size_t off=0; off<len;) { ssize_t n=send(fd,(const uint8_t*)buf+off,len-off,MSG_NOSIGNAL); if(n<0&&errno==EINTR)continue; if(n<=0)return -1; off+=(size_t)n; } return 0; }
static int read_request(int fd, request_t *req) { uint8_t b[33]; if (recvall(fd,b,sizeof(b))) return -1; req->cmd=b[0]; req->dev_id=get_le32(b+1); req->bar=get_le32(b+5); req->arg0=get_le64(b+9); req->arg1=get_le64(b+17); req->arg2=get_le64(b+25); return 0; }

static int send_response(int fd, response_t *resp, int send_fd) {
  uint8_t wire[17] = {resp->status};
  put_le64(wire + 1, resp->resp0);
  put_le64(wire + 9, resp->resp1);
  char cmsgbuf[CMSG_SPACE(sizeof(int))];
  struct iovec iov = {wire, sizeof(wire)};
  struct msghdr msg = {.msg_iov = &iov, .msg_iovlen = 1};

  if (send_fd >= 0) {
    msg.msg_control = cmsgbuf;
    msg.msg_controllen = sizeof(cmsgbuf);
    struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
    *cmsg = (struct cmsghdr){.cmsg_level = SOL_SOCKET, .cmsg_type = SCM_RIGHTS, .cmsg_len = CMSG_LEN(sizeof(int))};
    memcpy(CMSG_DATA(cmsg), &send_fd, sizeof(int));
  }
  for (;;) {
    ssize_t n = sendmsg(fd, &msg, MSG_NOSIGNAL);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) return -1;
    // SCM_RIGHTS is attached only to this first byte range. Continue a short
    // header write with plain send so the descriptor is never duplicated.
    return (size_t)n == sizeof(wire) ? 0 : sendall(fd, wire + n, sizeof(wire) - (size_t)n);
  }
}

static int send_typed_error(int fd, uint8_t status, const char *code, const char *message) {
  char payload[1025];
  int n = snprintf(payload,sizeof(payload),"{\"schema\":\"tinygpu.error.v1\",\"code\":\"%s\",\"message\":\"%s\"}",code,message);
  if(n<=0 || n>1024) return -1;
  response_t resp = {.status=status,.resp0=(uint64_t)n};
  return send_response(fd,&resp,-1) || sendall(fd,payload,(size_t)n);
}

static int send_kern_error(int fd, kern_return_t kr, const char *operation) {
  uint8_t status = RESP_NATIVE_ERROR;
  const char *code = "native_error";
  if (kr == kIOReturnNotAttached || kr == kIOReturnNotReady || kr == kIOReturnNotOpen || kr == kIOReturnNotFound) {
    status = RESP_PROVIDER_UNAVAILABLE; code = "provider_unavailable";
  } else if (kr == kIOReturnNoDevice || kr == kIOReturnOffline) {
    status = RESP_DEVICE_LOST; code = "device_lost";
  } else if (kr == kIOReturnBadArgument) {
    status = RESP_MALFORMED_REQUEST; code = "malformed_request";
  } else if (kr == kIOReturnBusy || kr == kIOReturnExclusiveAccess) {
    status = RESP_BUSY; code = "busy";
  } else if (kr == kIOReturnNotPermitted) {
    status = RESP_INVALID_STATE; code = "invalid_state";
  }
  char message[160];
  snprintf(message, sizeof(message), "%s failed (kern_return=0x%08x)", operation, (unsigned int)kr);
  return send_typed_error(fd, status, code, message);
}

static int request_valid(const request_t *r) {
  if(r->cmd==CMD_HANDSHAKE) return r->dev_id==0 && r->bar==0 && r->arg1<=UINT16_MAX;
  if(r->cmd==CMD_LEASE_ACQUIRE || r->cmd==CMD_KEEPALIVE_STATUS || r->cmd==CMD_RESET)
    return r->dev_id==0 && r->bar==0 && r->arg0==0 && r->arg1==0 && r->arg2==0;
  if(r->cmd==CMD_LEASE_RELEASE) return r->dev_id==0 && r->bar==0 && r->arg0!=0 && r->arg1==0 && r->arg2==0;
  if(r->cmd>CMD_RESIZE_BAR) return r->cmd >= 12 && r->cmd <= 19;
  if(r->dev_id!=0) return 0;
  if((r->cmd==CMD_CFG_READ || r->cmd==CMD_CFG_WRITE) &&
     !((r->arg1==1 || r->arg1==2 || r->arg1==4) && r->arg0%r->arg1==0 && r->arg0+r->arg1<=4096)) return 0;
  if(r->cmd==CMD_CFG_READ && (r->bar || r->arg2)) return 0;
  if(r->cmd==CMD_CFG_WRITE && (r->bar || (r->arg1<4 && r->arg2 >= (UINT64_C(1) << (r->arg1*8))))) return 0;
  if((r->cmd==CMD_MMIO_READ || r->cmd==CMD_MMIO_WRITE) &&
     (r->bar>=MAX_BARS || r->arg1>BULK_BUF_SIZE || r->arg0>UINT64_MAX-r->arg1 || r->arg2!=0)) return 0;
  if(r->cmd==CMD_MAP_BAR && (r->bar>=MAX_BARS || r->arg0 || r->arg1 || r->arg2)) return 0;
  if(r->cmd==CMD_MAP_SYSMEM_FD && (r->bar || r->arg0==0 || r->arg0>INT64_MAX-0xfff || r->arg1>1 || r->arg2)) return 0;
  if(r->cmd==CMD_RESIZE_BAR && (r->bar>=MAX_BARS || r->arg0 || r->arg1 || r->arg2)) return 0;
  return 1;
}

// Driver interface

static void on_disconnect(void *refcon, io_service_t svc, uint32_t msg, void *arg) {
  (void)refcon; (void)svc; (void)arg;
  if (msg == kIOMessageServiceIsTerminated) g_provider_terminated = 1;
}

static io_connect_t open_tinygpu(kern_return_t *out_error) {
  if (out_error) *out_error = kIOReturnNotFound;
  if (g_provider_terminated && g_service_notification) {
    IOObjectRelease(g_service_notification);
    g_service_notification = IO_OBJECT_NULL;
  }
  io_service_t svc = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceNameMatching("tinygpu"));
  if (!svc) return IO_OBJECT_NULL;

  if (!g_notification_port) {
    g_notification_port = IONotificationPortCreate(kIOMainPortDefault);
    if (!g_notification_port) { IOObjectRelease(svc); if (out_error) *out_error = kIOReturnNoResources; return IO_OBJECT_NULL; }
    IONotificationPortSetDispatchQueue(g_notification_port, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0));
  }
  if (!g_service_notification) {
    kern_return_t notify_error = IOServiceAddInterestNotification(g_notification_port, svc, kIOGeneralInterest,
                                                                   on_disconnect, NULL, &g_service_notification);
    if (notify_error != KERN_SUCCESS) { IOObjectRelease(svc); if (out_error) *out_error = notify_error; return IO_OBJECT_NULL; }
  }

  io_connect_t conn;
  kern_return_t kr = IOServiceOpen(svc, mach_task_self(), 0, &conn);
  IOObjectRelease(svc);
  if (out_error) *out_error = kr;
  if (kr == KERN_SUCCESS) g_provider_terminated = 0;
  return kr == KERN_SUCCESS ? conn : IO_OBJECT_NULL;
}

static kern_return_t dext_rpc(client_session_t *s, uint32_t sel, uint64_t *in, uint32_t in_cnt, uint64_t *out_val) {
  if (g_provider_terminated || s->conn == IO_OBJECT_NULL) return kIOReturnNotAttached;
  uint64_t out[2];
  uint32_t out_cnt = 2;
  kern_return_t kr = IOConnectCallMethod(s->conn, sel, in, in_cnt, NULL, 0, out, &out_cnt, NULL, NULL);
  if (kr != KERN_SUCCESS) return kr;
  if (g_provider_terminated) return kIOReturnNotAttached;
  if (out_val) *out_val = out[0];
  return kIOReturnSuccess;
}

static kern_return_t dext_mmio_read(client_session_t *s, uint32_t bar, uint64_t off, uint32_t width, uint32_t *value) {
  uint64_t in[3] = {bar, off, width}, out = 0;
  kern_return_t err = dext_rpc(s, 8, in, 3, &out);
  if (!err && value) *value = (uint32_t)out;
  return err;
}

static kern_return_t dext_mmio_write(client_session_t *s, uint32_t bar, uint64_t off, uint32_t width, uint32_t value) {
  uint64_t in[4] = {bar, off, width, value};
  return dext_rpc(s, 9, in, 4, NULL);
}

static uint32_t mmio_access_width(uint64_t off, uint64_t remaining) {
  if (!(off & 3) && remaining >= 4) return 4;
  if (!(off & 1) && remaining >= 2) return 2;
  return 1;
}

static kern_return_t mediated_mmio_read(client_session_t *s, uint32_t bar, uint64_t off, uint8_t *out, uint64_t size) {
  for (uint64_t done = 0; done < size;) {
    const uint32_t width = mmio_access_width(off + done, size - done);
    uint32_t value = 0;
    kern_return_t err = dext_mmio_read(s, bar, off + done, width, &value);
    if (err) return err;
    if (width == 4) put_le32(out + done, value);
    else if (width == 2) put_le16(out + done, (uint16_t)value);
    else out[done] = (uint8_t)value;
    done += width;
  }
  return kIOReturnSuccess;
}

static kern_return_t mediated_mmio_write(client_session_t *s, uint32_t bar, uint64_t off, const uint8_t *in, uint64_t size) {
  const int fence_each_store = bar == 0 && size > 4;
  for (uint64_t done = 0; done < size;) {
    const uint32_t width = mmio_access_width(off + done, size - done);
    const uint32_t value = width == 4 ? get_le32(in + done) : width == 2 ? get_le16(in + done) : in[done];
    kern_return_t err = dext_mmio_write(s, bar, off + done, width, value);
    if (err) return err;
    // v3/v4 serialized bulk BAR0 stores by reading the same location back.
    // Preserve that ordering primitive without inventing readbacks for exact
    // single-register writes on BAR5.
    if (fence_each_store) {
      uint32_t ignored = 0;
      if ((err = dext_mmio_read(s, bar, off + done, width, &ignored))) return err;
    }
    done += width;
  }
  return kIOReturnSuccess;
}

static kern_return_t ensure_connection(client_session_t *s) {
  if (s->conn != IO_OBJECT_NULL && !g_provider_terminated) return kIOReturnSuccess;
  if (s->conn != IO_OBJECT_NULL) { IOServiceClose(s->conn); s->conn = IO_OBJECT_NULL; }
  kern_return_t err = kIOReturnNotFound;
  s->conn = open_tinygpu(&err);
  if (s->conn == IO_OBJECT_NULL) return err;
  uint64_t in[3] = {1, 0, 0}, out[2] = {0, 0};
  uint32_t out_count = 2;
  err = IOConnectCallMethod(s->conn, 4, in, 3, NULL, 0, out, &out_count, NULL, NULL);
  if (err != KERN_SUCCESS ||
      out_count != 2 || out[0] != 1 || out[1] != 3) {
    IOServiceClose(s->conn);
    s->conn = IO_OBJECT_NULL;
    return err != KERN_SUCCESS ? err : kIOReturnUnsupported;
  }
  return kIOReturnSuccess;
}

static kern_return_t handshake_rpc(client_session_t *s, char *out, size_t *size) {
  if (!out || !size || !*size) return kIOReturnBadArgument;
  kern_return_t err = ensure_connection(s);
  if (err) return err;
  const char *json = "{\"schema\":\"tinygpu.handshake.v1\",\"protocol_major\":1,\"protocol_minor\":0,\"capabilities\":3,\"server_build_id\":\"tinygrad-arkey-native\"}";
  size_t length = strlen(json);
  if (length >= *size) return kIOReturnNoSpace;
  memcpy(out, json, length + 1);
  *size = length;
  return kIOReturnSuccess;
}

static kern_return_t status_rpc(client_session_t *s, char *out, size_t *size) {
  const size_t capacity = *size;
  if (!capacity || capacity > 4096) return kIOReturnBadArgument;
  memset(out, 0, capacity);
  size_t driver_size = capacity;
  kern_return_t err = ensure_connection(s);
  if (err) return err;
  err = IOConnectCallStructMethod(s->conn, 5, NULL, 0, out, &driver_size);
  if (err != KERN_SUCCESS) return err;
  // DriverKit's descriptor path may report the caller's fixed capacity rather
  // than the JSON length. The provider always writes one NUL-terminated object.
  const size_t actual = strnlen(out, capacity);
  if (!actual || actual == capacity) return kIOReturnBadMedia;
  *size = actual;
  return kIOReturnSuccess;
}

static kern_return_t map_bar(client_session_t *s, uint32_t bar, response_t *resp) {
  if (bar >= MAX_BARS) return kIOReturnBadArgument;
  if (!s->bars[bar].addr) {
    kern_return_t err = IOConnectMapMemory64(s->conn, bar, mach_task_self(), &s->bars[bar].addr, &s->bars[bar].size, kIOMapAnywhere);
    if (err) return err;
  }
  resp->resp0 = s->bars[bar].addr;
  resp->resp1 = s->bars[bar].size;
  return kIOReturnSuccess;
}

static kern_return_t map_sysmem_fd(client_session_t *s, uint64_t size, int contiguous, response_t *resp, int *out_fd) {
  (void)contiguous;
  if (s->sysmem_count >= MAX_SYSMEM) return kIOReturnNoResources;
  int idx = s->sysmem_count;
  int fd = -1;
  void *ptr = MAP_FAILED;
  char shm_name[32];
  kern_return_t error = kIOReturnNoMemory;

  // page-align, min 16KB for IOMemoryDescriptor
  size_t alloc_sz = (size + 0xfff) & ~0xfff;
  if (alloc_sz < 0x4000) alloc_sz = 0x4000;

  snprintf(shm_name, sizeof(shm_name), "/tinygpu_%d_%d", getpid(), idx);
  shm_unlink(shm_name);
  if ((fd = shm_open(shm_name, O_CREAT | O_RDWR, 0600)) < 0) goto fail;
  if (ftruncate(fd, (off_t)alloc_sz) < 0) goto fail;
  if ((ptr = mmap(NULL, alloc_sz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)) == MAP_FAILED) goto fail;
  if (shm_unlink(shm_name) < 0) goto fail;

  // PrepareDMA writes physical addresses to output buffer, copy to shared mem
  uint8_t paddr_buf[8192] = {0};
  size_t out_sz = sizeof(paddr_buf);
  error = IOConnectCallStructMethod(s->conn, 3, ptr, alloc_sz, paddr_buf, &out_sz);
  if (error != KERN_SUCCESS) goto fail;
  memcpy(ptr, paddr_buf, out_sz);

  s->sysmem[idx] = (typeof(s->sysmem[idx])){.addr = (mach_vm_address_t)ptr, .size = alloc_sz, .shm_fd = fd};
  strncpy(s->sysmem[idx].shm_name, shm_name, sizeof(s->sysmem[idx].shm_name));
  s->sysmem_count++;

  *resp = (response_t){.resp0 = alloc_sz, .resp1 = idx};
  *out_fd = fd;
  return kIOReturnSuccess;

fail:
  if (ptr != MAP_FAILED) munmap(ptr, alloc_sz);
  if (fd >= 0) close(fd);
  shm_unlink(shm_name);
  return error;
}

static int validate_bar(client_session_t *s, uint8_t bar, uint64_t off, uint64_t sz) {
  return (bar < MAX_BARS && s->bars[bar].addr && off + sz <= s->bars[bar].size && sz <= BULK_BUF_SIZE) ? 0 : -1;
}

static void cleanup_workload(client_session_t *s) {
  for (int i = 0; i < MAX_BARS; i++)
    if (s->bars[i].addr) {
      if (!g_provider_terminated && s->conn != IO_OBJECT_NULL)
        (void)IOConnectUnmapMemory64(s->conn, i, mach_task_self(), s->bars[i].addr);
      s->bars[i].addr = 0;
      s->bars[i].size = 0;
    }

  for (int i = 0; i < s->sysmem_count; i++) {
    munmap((void*)s->sysmem[i].addr, s->sysmem[i].size);
    close(s->sysmem[i].shm_fd);
    shm_unlink(s->sysmem[i].shm_name);
  }

  s->sysmem_count = 0;
}

static void cleanup(client_session_t *s) {
  if (s->cleaned) return;
  s->cleaned = 1;
  cleanup_workload(s);
  if (s->lease && s->conn != IO_OBJECT_NULL && !g_provider_terminated) {
    (void)dext_rpc(s, 7, &s->lease, 1, NULL);
  }
  s->lease = 0;
  if (s->conn != IO_OBJECT_NULL) { IOServiceClose(s->conn); s->conn = IO_OBJECT_NULL; }
  free(s->bulk);
  s->bulk = NULL;
}

int tinygpu_keepalive_status(char *out, unsigned long out_cap, unsigned long *out_len) {
  if (!out || !out_len || out_cap == 0 || out_cap > 4096) return -1;
  client_session_t session = {.fd=-1, .handshaken=1, .conn=IO_OBJECT_NULL};
  size_t size = (size_t)out_cap;
  int result = status_rpc(&session, out, &size);
  cleanup(&session);
  if (result || size > out_cap) return -1;
  *out_len = (unsigned long)size;
  return 0;
}

int tinygpu_keepalive_handshake(char *out, unsigned long out_cap, unsigned long *out_len) {
  if (!out || !out_len || out_cap == 0 || out_cap > 4096) return -1;
  client_session_t session = {.fd=-1, .handshaken=1, .conn=IO_OBJECT_NULL};
  size_t size = (size_t)out_cap;
  int result = handshake_rpc(&session, out, &size);
  cleanup(&session);
  if (result || size > out_cap) return -1;
  *out_len = (unsigned long)size;
  return 0;
}

static void handle_client(int fd) {
  client_session_t session = {.fd=fd, .conn=IO_OBJECT_NULL};
  session.bulk = malloc(BULK_BUF_SIZE);
  if (!session.bulk) {
    send_typed_error(fd, RESP_INTERNAL_ERROR, "internal_error", "transfer buffer allocation failed");
    return;
  }
  int bufsize = BULK_BUF_SIZE;
  setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &bufsize, sizeof(bufsize));
  setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(bufsize));
  printf("client connected\n");

  request_t req;
  response_t resp;
  while (read_request(fd, &req) == 0) {
    resp = (response_t){0};
    if(!request_valid(&req)) {
      (void)send_typed_error(fd,RESP_MALFORMED_REQUEST,"malformed_request","request fields are invalid");
      break;
    }

    if (req.cmd == CMD_HANDSHAKE) {
      if(session.handshaken) { if (send_typed_error(fd,RESP_INVALID_STATE,"invalid_state","handshake already completed")) break; continue; }
      if(req.arg0!=1) { if (send_typed_error(fd,RESP_UNSUPPORTED_VERSION,"unsupported_version","protocol major is unsupported")) break; continue; }
      if(req.arg1>0) { if (send_typed_error(fd,RESP_UNSUPPORTED_VERSION,"unsupported_version","protocol minor is unsupported")) break; continue; }
      if(req.arg2 & ~UINT64_C(3)) { if (send_typed_error(fd,RESP_UNSUPPORTED_CAPABILITY,"unsupported_capability","required capability is unavailable")) break; continue; }
      session.handshaken = 1;
      const char *hello = "{\"schema\":\"tinygpu.handshake.v1\",\"protocol_major\":1,\"protocol_minor\":0,\"capabilities\":3,\"server_build_id\":\"tinygrad-arkey\"}";
      resp.resp0 = strlen(hello); resp.resp1 = 1;
      if (send_response(fd, &resp, -1) || sendall(fd, hello, resp.resp0)) break;
      continue;
    }
    if(!session.handshaken) { if (send_typed_error(fd,RESP_INVALID_STATE,"invalid_state","handshake required")) break; continue; }
    if (req.cmd >= 12 && req.cmd <= 14) {
      if (send_typed_error(fd,RESP_UNSUPPORTED_CAPABILITY,"unsupported_capability","reserved command is unsupported")) break;
      continue;
    }
    if (req.cmd == 19) {
      if (send_typed_error(fd,RESP_UNSUPPORTED_CAPABILITY,"unsupported_capability","policy override is unsupported")) break;
      continue;
    }
    if (req.cmd == CMD_KEEPALIVE_STATUS) {
      char status[4096]; size_t size = sizeof(status);
      kern_return_t status_error = status_rpc(&session,status,&size);
      if (status_error) {
        (void)send_kern_error(fd,status_error,"keepalive status");
        break;
      } else {
        resp.resp0 = size;
        if (send_response(fd,&resp,-1) || sendall(fd,status,size)) break;
      }
      continue;
    }
    if (req.cmd == CMD_LEASE_ACQUIRE) {
      if(session.lease) {
        if (send_typed_error(fd,RESP_BUSY,"busy","a workload lease is active")) break;
      } else {
        kern_return_t lease_error = ensure_connection(&session);
        if (!lease_error) lease_error = dext_rpc(&session,6,NULL,0,&session.lease);
        if (lease_error || !session.lease) {
          session.lease = 0;
          if (lease_error) (void)send_kern_error(fd,lease_error,"workload lease acquire");
          else (void)send_typed_error(fd,RESP_INTERNAL_ERROR,"internal_error","provider returned an empty workload lease");
          break;
        }
        resp.resp1=session.lease;
        if (send_response(fd,&resp,-1)) break;
      }
      continue;
    }
    if (req.cmd == CMD_LEASE_RELEASE) {
      if(!session.lease||req.arg0!=session.lease) {
        if (send_typed_error(fd,RESP_INVALID_STATE,"invalid_state","workload lease is not active")) break;
      } else {
        cleanup_workload(&session);
        kern_return_t release_error = dext_rpc(&session,7,&session.lease,1,NULL);
        if(release_error) {
          (void)send_kern_error(fd,release_error,"workload lease release");
          break;
        } else {
          session.lease=0;
          if (send_response(fd,&resp,-1)) break;
        }
      }
      continue;
    }
    if (req.cmd <= CMD_RESIZE_BAR && req.cmd != CMD_RESET && !session.lease) {
      if (send_typed_error(fd,RESP_INVALID_STATE,"invalid_state","workload lease required")) break;
      continue;
    }

    switch (req.cmd) {
    case CMD_MAP_BAR: {
      kern_return_t err = map_bar(&session, req.bar, &resp);
      if (err) { (void)send_kern_error(fd,err,"BAR map"); goto disconnected; }
      break;
    }

    case CMD_MAP_SYSMEM_FD: {
      int shm_fd = -1;
      kern_return_t err = map_sysmem_fd(&session, req.arg0, (int)req.arg1, &resp, &shm_fd);
      if (err) { (void)send_kern_error(fd,err,"DMA map"); goto disconnected; }
      if (send_response(fd, &resp, shm_fd)) goto disconnected;
      continue;
    }

    case CMD_CFG_READ: {
      uint64_t in[2] = {req.arg0, req.arg1};
      kern_return_t err = dext_rpc(&session,0,in,2,&resp.resp0);
      if (err) { (void)send_kern_error(fd,err,"configuration read"); goto disconnected; }
      break;
    }

    case CMD_CFG_WRITE: {
      uint64_t in[3] = {req.arg0, req.arg1, req.arg2};
      kern_return_t err = dext_rpc(&session,1,in,3,NULL);
      if (err) { (void)send_kern_error(fd,err,"configuration write"); goto disconnected; }
      break;
    }

    case CMD_RESIZE_BAR:
      break;

    case CMD_RESET: {
      kern_return_t err = ensure_connection(&session);
      if (!err) err = dext_rpc(&session,2,NULL,0,NULL);
      if (err) { (void)send_kern_error(fd,err,"device reset"); goto disconnected; }
      break;
    }

    case CMD_MMIO_READ: {
      if (validate_bar(&session,req.bar,req.arg0,req.arg1)) {
        (void)send_typed_error(fd,RESP_MALFORMED_REQUEST,"malformed_request","MMIO range is outside the mapped BAR");
        goto disconnected;
      }
      kern_return_t err = mediated_mmio_read(&session,req.bar,req.arg0,session.bulk,req.arg1);
      if (err) { (void)send_kern_error(fd,err,"MMIO read"); goto disconnected; }
      resp.resp0 = req.arg1;
      if (send_response(fd, &resp, -1) || sendall(fd,session.bulk,req.arg1)) goto disconnected;
      continue;
    }

    case CMD_MMIO_WRITE: {
      if(req.arg1>BULK_BUF_SIZE||recvall(fd,session.bulk,req.arg1)) goto disconnected;
      if (validate_bar(&session,req.bar,req.arg0,req.arg1)) {
        (void)send_typed_error(fd,RESP_MALFORMED_REQUEST,"malformed_request","MMIO range is outside the mapped BAR");
        goto disconnected;
      }
      kern_return_t err = mediated_mmio_write(&session,req.bar,req.arg0,session.bulk,req.arg1);
      if (err) { (void)send_kern_error(fd,err,"MMIO write"); goto disconnected; }
      break;
    }

    default:
      resp.status = 1;
    }
    if (send_response(fd, &resp, -1)) break;
  }

disconnected:
  printf("client disconnected\n");
  cleanup(&session);
}

int run_server(const char *sock_path) {
  if (!sock_path || !*sock_path || strlen(sock_path) >= sizeof(((struct sockaddr_un *)0)->sun_path)) return 1;
  int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (server_fd < 0) { perror("socket"); return 1; }

  struct sockaddr_un addr = {.sun_family = AF_UNIX};
  strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);
  unlink(sock_path);

  if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) { perror("bind"); close(server_fd); return 1; }
  if (listen(server_fd, 1) < 0) { perror("listen"); close(server_fd); return 1; }
  printf("listening on %s\n", sock_path);

  while (1) {
    int client_fd = accept(server_fd, NULL, NULL);
    if (client_fd < 0) { if (errno == EINTR) continue; perror("accept"); break; }
    if (!__sync_bool_compare_and_swap(&g_client_active, 0, 1)) {
      send_typed_error(client_fd,RESP_BUSY,"busy","a workload client is active");
      close(client_fd);
      continue;
    }
    dispatch_async(dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0), ^{
      handle_client(client_fd);
      close(client_fd);
      __sync_lock_release(&g_client_active);
    });
  }

  close(server_fd);
  unlink(sock_path);
  return 0;
}
