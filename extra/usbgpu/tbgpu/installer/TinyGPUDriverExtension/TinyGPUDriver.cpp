#include "TinyGPUDriver.h"
#include "TinyGPUDriverUserClient.h"

#include <DriverKit/IOLib.h>
#include <DriverKit/OSSharedPtr.h>
#include <PCIDriverKit/PCIDriverKit.h>
#include <stdio.h>
#include <time.h>

struct TinyGPUPolicy {
	const char* id;
	uint32_t identity;
	uint32_t intervalMS;
	uint32_t leewayMS;
};

static constexpr TinyGPUPolicy kUSB4AMD744C = {"usb4_amd_744c_v1", 0x744c1002, 1000, 100};
static uint64_t gProviderGeneration = 0;

enum ProviderState : uint32_t {
	kDetached,
	kStarting,
	kActiveHealthy,
	kActiveDegraded,
	kQuiescing,
	kStopped,
};

struct TinyGPUDriver_IVars {
	IOPCIDevice* pci = nullptr;
	IODispatchQueue* gate = nullptr;
	IOTimerDispatchSource* timer = nullptr;
	OSAction* timerAction = nullptr;
	ProviderState state = kDetached;
	bool timerEnabled = false;
	bool counterSaturated = false;
	uint64_t generation = 0;
	uint64_t attempts = 0;
	uint64_t successes = 0;
	uint64_t failures = 0;
	uint64_t consecutiveFailures = 0;
	uint64_t lastAttempt = 0;
	uint64_t lastSuccess = 0;
	uint64_t overLeeway = 0;
	uint64_t maxGapMS = 0;
	uint64_t leaseID = 0;
	uint32_t lastIdentity = 0;
	uint32_t activeLeases = 0;
	uint32_t activeBars = 0;
	uint32_t activeDMA = 0;
	int32_t timerError = 0;
};

static uint64_t UptimeNS() {
	return clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
}

static uint64_t NextGeneration() {
	for (;;) {
		const uint64_t current = gProviderGeneration;
		if (current == UINT64_MAX) return UINT64_MAX;
		if (__sync_bool_compare_and_swap(&gProviderGeneration, current, current + 1)) return current + 1;
	}
}

static void SaturatingIncrement(uint64_t* value, bool* saturated) {
	if (*value == UINT64_MAX) *saturated = true;
	else ++*value;
}

static bool ValidConfig(uint32_t offset, uint32_t width) {
	return (width == 1 || width == 2 || width == 4) && offset < 4096 &&
		(offset % width) == 0 && offset <= 4096 - width;
}

static const char* StateName(ProviderState state) {
	switch (state) {
	case kActiveHealthy: return "active_healthy";
	case kActiveDegraded: return "active_degraded";
	case kQuiescing: return "quiescing";
	case kStopped: return "stopped";
	default: return "inactive";
	}
}

static kern_return_t DrainTimer(IODispatchQueue* gate, IOTimerDispatchSource* timer, bool cancel) {
	if (!gate || !timer) return kIOReturnSuccess;
	uint8_t disableToken = 0, cancelToken = 0;
	void* disableEvent = &disableToken;
	void* cancelEvent = &cancelToken;
	return gate->RunAction(^{
		kern_return_t err = timer->SetEnableWithCompletion(false, ^{ (void)gate->Wakeup(disableEvent); });
		if (err) return err;
		err = gate->SleepWithDeadline(disableEvent, 0, 0);
		if (err || !cancel) return err;
		err = timer->Cancel(^{ (void)gate->Wakeup(cancelEvent); });
		if (err) return err;
		return gate->SleepWithDeadline(cancelEvent, 0, 0);
	});
}

bool TinyGPUDriver::init() {
	if (!super::init()) return false;
	ivars = new TinyGPUDriver_IVars();
	return ivars != nullptr;
}

void TinyGPUDriver::free() {
	if (ivars) {
		if (ivars->timer) ivars->timer->release();
		if (ivars->timerAction) ivars->timerAction->release();
		if (ivars->gate) ivars->gate->release();
	}
	IOSafeDeleteNULL(ivars, TinyGPUDriver_IVars, 1);
	super::free();
}

kern_return_t TinyGPUDriver::Start_Impl(IOService* provider) {
	kern_return_t err = Start(provider, SUPERDISPATCH);
	if (err) return err;
	ivars->pci = OSDynamicCast(IOPCIDevice, provider);
	if (!ivars->pci) return kIOReturnNoDevice;
	err = IODispatchQueue::Create("tinygpu.provider.gate", 0, 0, &ivars->gate);
	if (err) return err;
	err = ivars->gate->RunAction(^{ return ivars->pci->Open(this, 0); });
	if (err) return err;

	ivars->state = kStarting;
	ivars->generation = NextGeneration();
	if (ivars->generation == UINT64_MAX) ivars->counterSaturated = true;
	__block uint32_t identity = 0;
	ivars->gate->DispatchSync(^{ ivars->pci->ConfigurationRead32(kIOPCIConfigurationOffsetVendorID, &identity); });
	ivars->lastIdentity = identity;
	if (identity != kUSB4AMD744C.identity) { err = kIOReturnUnsupported; goto fail; }

	err = IOTimerDispatchSource::Create(ivars->gate, &ivars->timer);
	if (err) goto fail;
	err = CreateActionKeepaliveTimer(0, &ivars->timerAction);
	if (err) goto fail;
	err = ivars->timer->SetHandler(ivars->timerAction);
	if (err) goto fail;
	err = ivars->timer->SetEnable(true);
	if (err) goto fail;
	ivars->timerEnabled = true;
	ivars->gate->DispatchSync(^{ KeepaliveTimer_Impl(nullptr, UptimeNS()); });
	if (ivars->state != kActiveHealthy) { err = kIOReturnNoDevice; goto fail; }

	IOServiceName name;
	memcpy(name, "tinygpu", 8);
	SetName(name);
	RegisterService();
	return kIOReturnSuccess;

fail:
	if (ivars->gate) ivars->gate->DispatchSync(^{ ivars->state = kStopped; ivars->timerEnabled = false; });
	if (ivars->timer) {
		(void)DrainTimer(ivars->gate, ivars->timer, true);
		ivars->timer->release();
		ivars->timer = nullptr;
	}
	if (ivars->timerAction) {
		ivars->timerAction->Cancel(nullptr);
		ivars->timerAction->release();
		ivars->timerAction = nullptr;
	}
	if (ivars->gate) ivars->gate->DispatchSync(^{
		if (ivars->pci) { ivars->pci->Close(this, 0); ivars->pci = nullptr; }
	});
	return err;
}

kern_return_t TinyGPUDriver::Stop_Impl(IOService* provider) {
	if (!ivars) return kIOReturnSuccess;
	if (ivars->gate) ivars->gate->DispatchSync(^{
		ivars->state = kQuiescing;
		ivars->timerEnabled = false;
	});
	const kern_return_t drainError = DrainTimer(ivars->gate, ivars->timer, true);
	if (drainError) {
		if (ivars->gate) ivars->gate->DispatchSync(^{ ivars->timerError = (int32_t)drainError; });
		return drainError;
	}

	if (ivars->timer) { ivars->timer->release(); ivars->timer = nullptr; }
	if (ivars->timerAction) {
		ivars->timerAction->Cancel(nullptr);
		ivars->timerAction->release();
		ivars->timerAction = nullptr;
	}
	__block kern_return_t closeError = kIOReturnSuccess;
	if (ivars->gate) ivars->gate->DispatchSync(^{
		if (ivars->activeLeases || ivars->activeBars || ivars->activeDMA) { closeError = kIOReturnBusy; return; }
		if (ivars->pci) { ivars->pci->Close(this, 0); ivars->pci = nullptr; }
		ivars->state = kStopped;
	});
	if (closeError) return closeError;
	return Stop(provider, SUPERDISPATCH);
}

void IMPL(TinyGPUDriver, KeepaliveTimer) {
	if (!ivars->timerEnabled || !ivars->pci || ivars->state == kQuiescing || ivars->state == kStopped) return;
	const uint64_t now = UptimeNS();
	const bool canCount = ivars->attempts != UINT64_MAX;
	if (canCount) ++ivars->attempts;
	else ivars->counterSaturated = true;
	ivars->lastAttempt = now;

	uint32_t identity = 0;
	ivars->pci->ConfigurationRead32(kIOPCIConfigurationOffsetVendorID, &identity);
	ivars->lastIdentity = identity;
	if (identity == kUSB4AMD744C.identity) {
		if (ivars->lastSuccess) {
			const uint64_t gap = now - ivars->lastSuccess;
			const uint64_t gapMS = (gap + 999999ull) / 1000000ull;
			if (gapMS > ivars->maxGapMS) ivars->maxGapMS = gapMS;
			if (gapMS > kUSB4AMD744C.intervalMS + kUSB4AMD744C.leewayMS)
				SaturatingIncrement(&ivars->overLeeway, &ivars->counterSaturated);
		}
		if (canCount) ++ivars->successes;
		ivars->lastSuccess = now;
		ivars->consecutiveFailures = 0;
		ivars->state = ivars->counterSaturated ? kActiveDegraded : kActiveHealthy;
	} else {
		if (canCount) ++ivars->failures;
		SaturatingIncrement(&ivars->consecutiveFailures, &ivars->counterSaturated);
		ivars->state = kActiveDegraded;
	}
	if (ivars->successes != ivars->attempts - ivars->failures) {
		ivars->counterSaturated = true;
		ivars->state = kActiveDegraded;
	}

	if (!ivars->timerEnabled || !ivars->timer) return;
	ivars->timerError = (int32_t)ivars->timer->WakeAtTime(
		kIOTimerClockUptimeRaw, now + uint64_t(kUSB4AMD744C.intervalMS) * 1000000ull,
		uint64_t(kUSB4AMD744C.leewayMS) * 1000000ull);
	if (ivars->timerError) ivars->state = kActiveDegraded;
}

kern_return_t TinyGPUDriver::NewUserClient_Impl(uint32_t type, IOUserClient** out) {
	if (!out || !ivars->gate) return kIOReturnBadArgument;
	__block bool admitted = false;
	ivars->gate->DispatchSync(^{ admitted = ivars->state == kActiveHealthy || ivars->state == kActiveDegraded; });
	if (!admitted) return kIOReturnNotReady;
	OSSharedPtr<IOService> service;
	kern_return_t err = Create(this, "TinyGPUDriverUserClientProperties", service.attach());
	if (err) return err;
	IOUserClient* client = OSDynamicCast(IOUserClient, service.get());
	if (!client) {
		IO_FOR_ANALYZER(service.get()->release());
		return kIOReturnError;
	}
	*out = client;
	(void)service.detach();
	// NewUserClient transfers this +1 reference to DriverKit's output contract.
	IO_FOR_ANALYZER(client->release());
	return kIOReturnSuccess;
}

kern_return_t TinyGPUDriver::CfgRead(uint32_t offset, uint32_t width, uint32_t* out) {
	if (!out || !ValidConfig(offset, width) || !ivars->gate) return kIOReturnBadArgument;
	__block kern_return_t err = kIOReturnSuccess;
	ivars->gate->DispatchSync(^{
		if (!ivars->pci || !ivars->activeLeases || ivars->state != kActiveHealthy) { err = kIOReturnNotReady; return; }
		if (width == 1) { uint8_t value = 0; ivars->pci->ConfigurationRead8(offset, &value); *out = value; }
		else if (width == 2) { uint16_t value = 0; ivars->pci->ConfigurationRead16(offset, &value); *out = value; }
		else ivars->pci->ConfigurationRead32(offset, out);
	});
	return err;
}

kern_return_t TinyGPUDriver::CfgWrite(uint32_t offset, uint32_t width, uint32_t value) {
	if (!ValidConfig(offset, width) || !ivars->gate) return kIOReturnBadArgument;
	return ivars->gate->RunAction(^{
		if (!ivars->pci || !ivars->activeLeases || ivars->state != kActiveHealthy) return kIOReturnNotReady;
		if (width == 1) ivars->pci->ConfigurationWrite8(offset, (uint8_t)value);
		else if (width == 2) ivars->pci->ConfigurationWrite16(offset, (uint16_t)value);
		else ivars->pci->ConfigurationWrite32(offset, value);
		return kIOReturnSuccess;
	});
}

kern_return_t TinyGPUDriver::ResetDevice() {
	if (!ivars->gate || !ivars->timer) return kIOReturnNotReady;
	__block kern_return_t err = kIOReturnSuccess;
	ivars->gate->DispatchSync(^{
		if (!ivars->pci) err = kIOReturnNotReady;
		else if (ivars->state == kQuiescing || ivars->state == kStopped ||
			ivars->activeLeases || ivars->activeBars || ivars->activeDMA) err = kIOReturnBusy;
		else { ivars->state = kQuiescing; ivars->timerEnabled = false; }
	});
	if (err) return err;

	err = DrainTimer(ivars->gate, ivars->timer, false);
	if (err) {
		ivars->gate->DispatchSync(^{ ivars->timerError = (int32_t)err; ivars->state = kActiveDegraded; });
		return err;
	}
	ivars->gate->DispatchSync(^{
		err = ivars->pci->Reset(kIOPCIDeviceResetTypeFunctionReset);
		if (!err) {
			uint32_t identity = 0;
			ivars->pci->ConfigurationRead32(kIOPCIConfigurationOffsetVendorID, &identity);
			ivars->lastIdentity = identity;
			if (identity != kUSB4AMD744C.identity) err = kIOReturnNoDevice;
		}
	});
	if (err) { ivars->gate->DispatchSync(^{ ivars->state = kActiveDegraded; }); return err; }

	err = ivars->timer->SetEnable(true);
	if (err) {
		ivars->gate->DispatchSync(^{ ivars->timerError = (int32_t)err; ivars->state = kActiveDegraded; });
		return err;
	}
	ivars->gate->DispatchSync(^{
		ivars->state = kStarting;
		ivars->timerEnabled = true;
		KeepaliveTimer_Impl(nullptr, UptimeNS());
		if (ivars->state != kActiveHealthy) err = kIOReturnNoDevice;
	});
	return err;
}

kern_return_t TinyGPUDriver::MapBar(uint64_t leaseID, uint32_t bar, IOMemoryDescriptor** memory) {
	if (!leaseID || !memory || bar >= 6 || !ivars->gate) return kIOReturnBadArgument;
	__block kern_return_t err = kIOReturnSuccess;
	ivars->gate->DispatchSync(^{
		uint8_t index = 0, type = 0;
		uint64_t bytes = 0;
		if (!ivars->pci || ivars->activeLeases != 1 || leaseID != ivars->leaseID || ivars->state != kActiveHealthy)
			err = kIOReturnNotReady;
		else if (ivars->activeBars == UINT32_MAX) err = kIOReturnNoResources;
		else if (!(err = ivars->pci->GetBARInfo(bar, &index, &bytes, &type)) &&
		         !(err = ivars->pci->_CopyDeviceMemoryWithIndex(index, memory, this)))
			++ivars->activeBars;
	});
	return err;
}

static kern_return_t WriteDMASegments(IOMemoryDescriptor* memory, IOAddressSegment* segments,
		uint32_t count, uint64_t offset = 0, uint64_t length = 0) {
	IOMemoryMap* map = nullptr;
	kern_return_t err = memory->CreateMapping(0, 0, 0, offset, length, &map);
	if (err || !map) return err ?: kIOReturnError;
	uint64_t* out = (uint64_t*)map->GetAddress();
	for (uint32_t i = 0; i < count; i++) { out[i * 2] = segments[i].address; out[i * 2 + 1] = segments[i].length; }
	out[count * 2] = 0;
	out[count * 2 + 1] = 0;
	map->release();
	return kIOReturnSuccess;
}

kern_return_t TinyGPUDriver::SetupDMA(uint64_t leaseID, IOMemoryDescriptor* memory, uint64_t size, IODMACommand** out,
		IOAddressSegment* segments, uint32_t* segmentCount) {
	if (!leaseID || !memory || !size || !out || !segments || !segmentCount || !ivars->gate) return kIOReturnBadArgument;
	__block kern_return_t err = kIOReturnSuccess;
	__block IODMACommand* command = nullptr;
	ivars->gate->DispatchSync(^{
		if (!ivars->pci || ivars->activeLeases != 1 || leaseID != ivars->leaseID || ivars->state != kActiveHealthy) {
			err = kIOReturnNotReady; return;
		}
		if (ivars->activeDMA == UINT32_MAX) { err = kIOReturnNoResources; return; }
		IODMACommandSpecification specification = {.options = 0, .maxAddressBits = 40};
		err = IODMACommand::Create(ivars->pci, kIODMACommandCreateNoOptions, &specification, &command);
		if (err) return;
		uint64_t flags = kIOMemoryDirectionInOut;
		err = command->PrepareForDMA(kIODMACommandPrepareForDMANoOptions, memory, 0, size, &flags, segmentCount, segments);
		if (err) { command->release(); command = nullptr; }
		else ++ivars->activeDMA;
	});
	if (!err) *out = command;
	return err;
}

kern_return_t TinyGPUDriver::CompleteDMA(IODMACommand* command) {
	if (!command || !ivars->gate) return kIOReturnBadArgument;
	return ivars->gate->RunAction(^{ return command->CompleteDMA(kIODMACommandCompleteDMANoOptions); });
}

kern_return_t TinyGPUDriver::CreateDMA(uint64_t leaseID, size_t size, TinyGPUCreateDMAResp* descriptor) {
	if (!leaseID || !size || !descriptor) return kIOReturnBadArgument;
	IOBufferMemoryDescriptor* buffer = nullptr;
	kern_return_t err = IOBufferMemoryDescriptor::Create(kIOMemoryDirectionInOut, size, IOVMPageSize, &buffer);
	if (err) return err;
	IODMACommand* command = nullptr;
	IOAddressSegment segments[32];
	uint32_t count = 32;
	err = SetupDMA(leaseID, buffer, size, &command, segments, &count);
	if (!err) err = WriteDMASegments(buffer, segments, count, IOVMPageSize, IOVMPageSize);
	if (err) {
		if (command) { (void)CompleteDMA(command); command->release(); ReleaseDMAAllocation(); }
		buffer->release();
		return err;
	}
	descriptor->sharedBuf = buffer;
	descriptor->dmaCmd = command;
	return kIOReturnSuccess;
}

kern_return_t TinyGPUDriver::AcquireWorkloadLease(uint64_t* out) {
	if (!out || !ivars->gate) return kIOReturnBadArgument;
	return ivars->gate->RunAction(^{
		if (ivars->activeLeases) return kIOReturnBusy;
		if (ivars->state != kActiveHealthy || ivars->counterSaturated) return kIOReturnNotReady;
		if (ivars->leaseID == UINT64_MAX) { ivars->counterSaturated = true; ivars->state = kActiveDegraded; return kIOReturnNoResources; }
		++ivars->leaseID;
		++ivars->activeLeases;
		*out = ivars->leaseID;
		return kIOReturnSuccess;
	});
}

kern_return_t TinyGPUDriver::ReleaseWorkloadLease(uint64_t id) {
	if (!id || !ivars->gate) return kIOReturnBadArgument;
	return ivars->gate->RunAction(^{
		if (!ivars->activeLeases || id != ivars->leaseID) return kIOReturnNotFound;
		if (ivars->activeBars || ivars->activeDMA) return kIOReturnBusy;
		--ivars->activeLeases;
		return kIOReturnSuccess;
	});
}

void TinyGPUDriver::ReleaseBarMapping() {
	if (ivars->gate) ivars->gate->DispatchSync(^{ if (ivars->activeBars) --ivars->activeBars; });
}

void TinyGPUDriver::ReleaseDMAAllocation() {
	if (ivars->gate) ivars->gate->DispatchSync(^{ if (ivars->activeDMA) --ivars->activeDMA; });
}

kern_return_t TinyGPUDriver::GetKeepaliveStatus(char* out, size_t* length) {
	if (!out || !length || !*length) return kIOReturnBadArgument;
	struct Snapshot {
		ProviderState state;
		bool enabled, saturated;
		uint64_t generation, attempts, successes, failures, consecutiveFailures;
		uint64_t lastAttempt, lastSuccess, overLeeway, maxGapMS;
		uint32_t lastIdentity, leases, bars, dma;
		int32_t timerError;
	};
	__block Snapshot snapshot{};
	auto capture = ^{
		snapshot.state = ivars->state;
		snapshot.enabled = ivars->timerEnabled;
		snapshot.saturated = ivars->counterSaturated;
		snapshot.generation = ivars->generation;
		snapshot.attempts = ivars->attempts;
		snapshot.successes = ivars->successes;
		snapshot.failures = ivars->failures;
		snapshot.consecutiveFailures = ivars->consecutiveFailures;
		snapshot.lastAttempt = ivars->lastAttempt;
		snapshot.lastSuccess = ivars->lastSuccess;
		snapshot.overLeeway = ivars->overLeeway;
		snapshot.maxGapMS = ivars->maxGapMS;
		snapshot.lastIdentity = ivars->lastIdentity;
		snapshot.leases = ivars->activeLeases;
		snapshot.bars = ivars->activeBars;
		snapshot.dma = ivars->activeDMA;
		snapshot.timerError = ivars->timerError;
	};
	if (ivars->gate && !ivars->gate->OnQueue()) ivars->gate->DispatchSync(capture);
	else capture();

	int written = snprintf(out, *length,
		"{\"schema\":\"tinygpu.keepalive.v1\",\"provider_generation\":%llu,\"state\":\"%s\",\"enabled\":%s,"
		"\"policy_id\":\"usb4_amd_744c_v1\",\"interval_ms\":1000,\"maximum_timer_leeway_ms\":100,"
		"\"expected_identity\":\"1002:744c\",\"last_identity_dword\":\"0x%08x\",\"attempts\":%llu,"
		"\"successes\":%llu,\"failures\":%llu,\"consecutive_failures\":%llu,\"last_attempt_monotonic_ns\":%llu,"
		"\"last_success_monotonic_ns\":%llu,\"success_gap_over_leeway_count\":%llu,\"max_success_gap_ms\":%llu,"
		"\"timer_error\":%d,\"counter_saturated\":%s,\"active_workload_leases\":%u,"
		"\"active_bar_mappings\":%u,\"active_dma_allocations\":%u}",
		(unsigned long long)snapshot.generation, StateName(snapshot.state), snapshot.enabled ? "true" : "false",
		snapshot.lastIdentity, (unsigned long long)snapshot.attempts, (unsigned long long)snapshot.successes,
		(unsigned long long)snapshot.failures, (unsigned long long)snapshot.consecutiveFailures,
		(unsigned long long)snapshot.lastAttempt, (unsigned long long)snapshot.lastSuccess,
		(unsigned long long)snapshot.overLeeway, (unsigned long long)snapshot.maxGapMS, snapshot.timerError,
		snapshot.saturated ? "true" : "false", snapshot.leases, snapshot.bars, snapshot.dma);
	if (written < 0 || (size_t)written >= *length) return kIOReturnNoSpace;
	*length = (size_t)written;
	return kIOReturnSuccess;
}
