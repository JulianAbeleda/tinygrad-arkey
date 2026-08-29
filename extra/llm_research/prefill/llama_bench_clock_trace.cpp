// LD_PRELOAD marker shim for llama-bench's two high_resolution_clock calls.
// It does not modify the pinned llama binary.  Every system_clock::now call is
// bracketed by NVTX marks carrying the exact returned epoch nanoseconds.
#include <chrono>
#include <cstdio>
#include <ctime>
#include <sys/syscall.h>
#include <unistd.h>
#include <nvtx3/nvToolsExt.h>

extern "C" std::chrono::system_clock::time_point traced_system_clock_now() noexcept
  asm("_ZNSt6chrono3_V212system_clock3nowEv");

extern "C" std::chrono::system_clock::time_point traced_system_clock_now() noexcept {
  static unsigned long long sequence = 0;
  const unsigned long long seq = __atomic_fetch_add(&sequence, 1ULL, __ATOMIC_RELAXED);
  char before[96];
  std::snprintf(before, sizeof(before), "LLAMA_CLOCK_BEFORE:%llu", seq);
  nvtxMarkA(before);
  struct timespec ts {};
  syscall(SYS_clock_gettime, CLOCK_REALTIME, &ts);
  const long long ns = static_cast<long long>(ts.tv_sec)*1000000000LL + ts.tv_nsec;
  char after[128];
  std::snprintf(after, sizeof(after), "LLAMA_CLOCK_AFTER:%llu:%lld", seq, ns);
  nvtxMarkA(after);
  return std::chrono::system_clock::time_point(std::chrono::nanoseconds(ns));
}
