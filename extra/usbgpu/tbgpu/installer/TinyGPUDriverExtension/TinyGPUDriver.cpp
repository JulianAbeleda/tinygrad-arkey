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
static constexpr const char* kPowerResidencyPolicy = "driverkit_full_power_v1";
static constexpr const char* kBARResidencyPolicy = "driverkit_bar5_mapping_v1";
static constexpr const char* kPCICommandPolicy = "pci_command_enable_v1";
static constexpr uint32_t kKeeperBAR = 5;
static constexpr uint32_t kFullPowerFlags = kIOServicePowerCapabilityOn;
static constexpr uint32_t kReleasedPowerFlags = kIOServicePowerCapabilityOff;
static constexpr uint16_t kRequiredPCICommand =
	kIOPCICommandIOSpace | kIOPCICommandMemorySpace | kIOPCICommandBusMaster;
static constexpr uint64_t kMaxPowerRequestAttempts = 3;
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
	bool canaryHealthy = false;
	bool counterSaturated = false;
	bool fullPowerRequested = false;
	bool powerRequestAccepted = false;
	bool powerRequestConfirmed = false;
	bool powerReleaseAttempted = false;
	bool barResidencyRequested = false;
	bool barResidencyActive = false;
	bool pciCommandRequested = false;
	bool pciCommandConfirmed = false;
	uint64_t generation = 0;
	uint64_t attempts = 0;
	uint64_t successes = 0;
	uint64_t failures = 0;
	uint64_t consecutiveFailures = 0;
	uint64_t lastAttempt = 0;
	uint64_t lastSuccess = 0;
	uint64_t overLeeway = 0;
	uint64_t maxGapMS = 0;
	uint64_t powerTransitions = 0;
	uint64_t unexpectedPowerDowngrades = 0;
	uint64_t lastPowerTransition = 0;
	uint64_t powerRequestAttempts = 0;
	uint64_t lastPowerRequestTick = 0;
	uint64_t lastPCICommandTick = 0;
	uint64_t leaseID = 0;
	uint32_t lastIdentity = 0;
	uint32_t desiredPowerFlags = kReleasedPowerFlags;
	uint32_t lastObservedPowerFlags = kIOServicePowerCapabilityOff;
	uint32_t activeLeases = 0;
	uint32_t activeBars = 0;
	uint32_t activeDMA = 0;
	uint32_t stopBusyLeases = 0;
	uint32_t stopBusyBars = 0;
	uint32_t stopBusyDMA = 0;
	uint32_t barResidencyType = 0;
	uint16_t pciCommandBefore = 0;
	uint16_t pciCommandAfter = 0;
	uint64_t barResidencyBytes = 0;
	IOMemoryDescriptor* barResidencyMemory = nullptr;
	IOMemoryMap* barResidencyMap = nullptr;
	IOMemoryDescriptor* mmioMemory[6] = {};
	IOMemoryMap* mmioMap[6] = {};
	uint64_t mmioSize[6] = {};
	int32_t timerError = 0;
	int32_t overrideProbePreJoinError = 0;
	int32_t overrideProbePostJoinError = 0;
	int32_t powerRequestError = 0;
	int32_t powerReleaseError = 0;
	int32_t barResidencyError = 0;
	int32_t pciCommandError = 0;
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

static bool ValidMMIOWidth(uint64_t offset, uint32_t width) {
	return (width == 1 || width == 2 || width == 4) && (offset % width) == 0;
}

// The provider used by every confirmed pre-prune TinyGPU run enabled PCI I/O,
// memory decoding, and bus mastering before publishing the service. Keep the
// operation on the provider gate and require readback instead of assuming the
// config write took effect across the USB4 tunnel.
static kern_return_t EnablePCICommand(TinyGPUDriver_IVars* state) {
	state->pciCommandRequested = true;
	state->pciCommandConfirmed = false;
	state->pciCommandError = 0;
	state->lastPCICommandTick = UptimeNS();
	state->pci->ConfigurationRead16(kIOPCIConfigurationOffsetCommand, &state->pciCommandBefore);
	const uint16_t enabled = state->pciCommandBefore | kRequiredPCICommand;
	if (enabled != state->pciCommandBefore)
		state->pci->ConfigurationWrite16(kIOPCIConfigurationOffsetCommand, enabled);
	state->pci->ConfigurationRead16(kIOPCIConfigurationOffsetCommand, &state->pciCommandAfter);
	state->pciCommandConfirmed = (state->pciCommandAfter & kRequiredPCICommand) == kRequiredPCICommand;
	if (!state->pciCommandConfirmed) state->pciCommandError = (int32_t)kIOReturnNotReady;
	return (kern_return_t)state->pciCommandError;
}

static void ObservePCICommand(TinyGPUDriver_IVars* state) {
	state->pci->ConfigurationRead16(kIOPCIConfigurationOffsetCommand, &state->pciCommandAfter);
	state->pciCommandConfirmed = state->pciCommandRequested &&
		(state->pciCommandAfter & kRequiredPCICommand) == kRequiredPCICommand;
	state->pciCommandError = state->pciCommandConfirmed ? 0 : (int32_t)kIOReturnNotReady;
}

static bool PCICommandReady(const TinyGPUDriver_IVars* state) {
	return state->pciCommandRequested && state->pciCommandConfirmed && !state->pciCommandError &&
		(state->pciCommandAfter & kRequiredPCICommand) == kRequiredPCICommand && state->lastPCICommandTick;
}

static void ReleaseMMIOMappings(TinyGPUDriver_IVars* state) {
	if (!state) return;
	for (uint32_t bar = 0; bar < 6; ++bar) {
		if (state->mmioMap[bar]) { state->mmioMap[bar]->release(); state->mmioMap[bar] = nullptr; }
		if (state->mmioMemory[bar]) { state->mmioMemory[bar]->release(); state->mmioMemory[bar] = nullptr; }
		state->mmioSize[bar] = 0;
	}
}

static void ReleaseBARResidency(TinyGPUDriver_IVars* state) {
	if (!state) return;
	if (state->barResidencyMap) { state->barResidencyMap->release(); state->barResidencyMap = nullptr; }
	if (state->barResidencyMemory) { state->barResidencyMemory->release(); state->barResidencyMemory = nullptr; }
	state->barResidencyActive = false;
	state->barResidencyBytes = 0;
	state->barResidencyType = 0;
}

// The historical Python bridge retained its BAR mappings for the lifetime of
// the opened device. Reproduce only the first and smallest part of that state:
// AMD initialization maps BAR5 first, and Apple IOPCIFamily's device-memory
// path establishes the separate tunneled-PCIe L1 veto before returning the
// descriptor. This provider-owned mapping is not a workload BAR resource.
static kern_return_t AcquireBARResidency(TinyGPUDriver* owner, TinyGPUDriver_IVars* state) {
	if (state->barResidencyActive) return kIOReturnSuccess;
	state->barResidencyRequested = true;
	state->barResidencyError = 0;
	uint8_t index = 0, type = 0;
	uint64_t bytes = 0;
	kern_return_t err = state->pci->GetBARInfo(kKeeperBAR, &index, &bytes, &type);
	if (!err && !bytes) err = kIOReturnBadMedia;
	IOMemoryDescriptor* memory = nullptr;
	IOMemoryMap* map = nullptr;
	if (!err) err = state->pci->_CopyDeviceMemoryWithIndex(index, &memory, owner);
	if (!err && !memory) err = kIOReturnError;
	if (!err) err = memory->CreateMapping(0, 0, 0, 0, 0, &map);
	if (!err && !map) err = kIOReturnError;
	if (err) {
		if (map) map->release();
		if (memory) memory->release();
		state->barResidencyError = (int32_t)err;
		return err;
	}
	state->barResidencyMemory = memory;
	state->barResidencyMap = map;
	state->barResidencyBytes = bytes;
	state->barResidencyType = type;
	state->barResidencyActive = true;
	return kIOReturnSuccess;
}

// Called only from the provider gate. Keeping this mapping provider-owned is
// what makes every BAR access participate in the same ordering domain as the
// keepalive timer and provider lifecycle.
static kern_return_t EnsureMMIOMapping(TinyGPUDriver* owner, TinyGPUDriver_IVars* state, uint32_t bar) {
	if (state->mmioMap[bar]) return kIOReturnSuccess;
	uint8_t index = 0, type = 0;
	uint64_t bytes = 0;
	kern_return_t err = state->pci->GetBARInfo(bar, &index, &bytes, &type);
	if (err) return err;
	IOMemoryDescriptor* memory = nullptr;
	err = state->pci->_CopyDeviceMemoryWithIndex(index, &memory, owner);
	if (err || !memory) return err ?: kIOReturnError;
	IOMemoryMap* map = nullptr;
	err = memory->CreateMapping(0, 0, 0, 0, 0, &map);
	if (err || !map) { memory->release(); return err ?: kIOReturnError; }
	state->mmioMemory[bar] = memory;
	state->mmioMap[bar] = map;
	state->mmioSize[bar] = bytes;
	return kIOReturnSuccess;
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

static bool FullPowerStateConfirmsRequest(const TinyGPUDriver_IVars* state) {
	return state->fullPowerRequested && state->powerRequestAccepted && !state->powerRequestError &&
		!state->powerReleaseAttempted && state->powerTransitions &&
		state->lastObservedPowerFlags == kFullPowerFlags && state->lastPowerRequestTick;
}

static bool PowerResidencyReady(const TinyGPUDriver_IVars* state) {
	return state->powerRequestConfirmed && FullPowerStateConfirmsRequest(state) && PCICommandReady(state) &&
		state->lastSuccess > state->lastPowerRequestTick && state->lastSuccess > state->lastPCICommandTick;
}

static bool ProviderResidencyReady(const TinyGPUDriver_IVars* state) {
	return PowerResidencyReady(state) && state->barResidencyRequested && state->barResidencyActive &&
		!state->barResidencyError && state->barResidencyBytes;
}

static void RefreshProviderHealth(TinyGPUDriver_IVars* state) {
	if (state->state == kQuiescing || state->state == kStopped) return;
	state->state = state->canaryHealthy && !state->counterSaturated && !state->timerError && ProviderResidencyReady(state)
		? kActiveHealthy : kActiveDegraded;
}

// These are post-Start IOService power-management requests on the TinyGPU
// service. API acceptance alone is not evidence that the request took effect;
// readiness also requires an observed On callback and a later PCI canary.
static kern_return_t RequestPowerResidency(TinyGPUDriver* owner, TinyGPUDriver_IVars* state) {
	SaturatingIncrement(&state->powerRequestAttempts, &state->counterSaturated);
	state->lastPowerRequestTick = UptimeNS();
	state->fullPowerRequested = true;
	state->powerRequestAccepted = false;
	state->powerRequestConfirmed = false;
	state->desiredPowerFlags = kFullPowerFlags;
	kern_return_t requestError = owner->ChangePowerState(kFullPowerFlags);
	state->powerRequestError = (int32_t)requestError;
	state->powerRequestAccepted = requestError == kIOReturnSuccess;
	// ChangePowerState may synchronously deliver SetPowerState before the return
	// value can be recorded. Reconcile that callback after recording acceptance;
	// asynchronous callbacks use the same state predicate below.
	state->powerRequestConfirmed = FullPowerStateConfirmsRequest(state);
	if (state->powerRequestAttempts == 1)
		state->overrideProbePostJoinError = (int32_t)owner->SetPowerOverride(false);
	return requestError;
}

static kern_return_t ReleasePowerResidency(TinyGPUDriver* owner, TinyGPUDriver_IVars* state) {
	if (state->powerReleaseAttempted) return kIOReturnSuccess;
	state->powerReleaseAttempted = true;
	state->powerRequestConfirmed = false;
	kern_return_t releaseError = kIOReturnSuccess;
	if (state->powerRequestAccepted) {
		releaseError = owner->ChangePowerState(kReleasedPowerFlags);
		state->powerReleaseError = (int32_t)releaseError;
	}
	state->desiredPowerFlags = kReleasedPowerFlags;
	return releaseError;
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
		ReleaseMMIOMappings(ivars);
		ReleaseBARResidency(ivars);
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
	// The expected pre-join error makes the DriverKit lifecycle boundary
	// observable without enabling an override or changing power state.
	ivars->overrideProbePreJoinError = (int32_t)SetPowerOverride(false);
	__block uint32_t identity = 0;
	ivars->gate->DispatchSync(^{ ivars->pci->ConfigurationRead32(kIOPCIConfigurationOffsetVendorID, &identity); });
	ivars->lastIdentity = identity;
	if (identity != kUSB4AMD744C.identity) { err = kIOReturnUnsupported; goto fail; }
	err = ivars->gate->RunAction(^{ return EnablePCICommand(ivars); });
	if (err) goto fail;
	err = ivars->gate->RunAction(^{ return AcquireBARResidency(this, ivars); });
	if (err) goto fail;

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
	if (!ivars->canaryHealthy) { err = kIOReturnNoDevice; goto fail; }

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
		(void)ReleasePowerResidency(this, ivars);
		ReleaseBARResidency(ivars);
		if (ivars->pci) { ivars->pci->Close(this, 0); ivars->pci = nullptr; }
	});
	return err;
}

kern_return_t TinyGPUDriver::Stop_Impl(IOService* provider) {
	__block kern_return_t releaseError = kIOReturnSuccess;
	kern_return_t drainError = kIOReturnSuccess;
	if (ivars) {
		if (ivars->gate) ivars->gate->DispatchSync(^{
			ivars->stopBusyLeases = ivars->activeLeases;
			ivars->stopBusyBars = ivars->activeBars;
			ivars->stopBusyDMA = ivars->activeDMA;
			ivars->state = kQuiescing;
			ivars->timerEnabled = false;
		});
		drainError = DrainTimer(ivars->gate, ivars->timer, true);
		if (drainError && ivars->gate)
			ivars->gate->DispatchSync(^{ ivars->timerError = (int32_t)drainError; });

		if (ivars->timer) { ivars->timer->release(); ivars->timer = nullptr; }
		if (ivars->timerAction) {
			ivars->timerAction->Cancel(nullptr);
			ivars->timerAction->release();
			ivars->timerAction = nullptr;
		}
		if (ivars->gate) ivars->gate->DispatchSync(^{
			ReleaseMMIOMappings(ivars);
			ReleaseBARResidency(ivars);
			releaseError = ReleasePowerResidency(this, ivars);
			if (ivars->pci) { ivars->pci->Close(this, 0); ivars->pci = nullptr; }
			ivars->state = kStopped;
		});
		else {
			ReleaseMMIOMappings(ivars);
			ReleaseBARResidency(ivars);
			releaseError = ReleasePowerResidency(this, ivars);
			if (ivars->pci) { ivars->pci->Close(this, 0); ivars->pci = nullptr; }
			ivars->state = kStopped;
		}
		if (ivars->stopBusyLeases || ivars->stopBusyBars || ivars->stopBusyDMA || drainError)
			IOLog("tinygpu: forced Stop cleanup leases=%u bars=%u dma=%u drain=%d\n",
				ivars->stopBusyLeases, ivars->stopBusyBars, ivars->stopBusyDMA, (int32_t)drainError);
	}
	kern_return_t stopError = Stop(provider, SUPERDISPATCH);
	return stopError ? stopError : (releaseError ? releaseError : drainError);
}

kern_return_t TinyGPUDriver::SetPowerState_Impl(uint32_t powerFlags) {
	if (ivars) {
		auto observe = ^{
			const bool unexpected = ivars->fullPowerRequested && ivars->powerRequestAccepted &&
				!ivars->powerReleaseAttempted && powerFlags != kFullPowerFlags;
			SaturatingIncrement(&ivars->powerTransitions, &ivars->counterSaturated);
			if (unexpected) SaturatingIncrement(&ivars->unexpectedPowerDowngrades, &ivars->counterSaturated);
			ivars->lastObservedPowerFlags = powerFlags;
			ivars->lastPowerTransition = UptimeNS();
			ivars->powerRequestConfirmed = FullPowerStateConfirmsRequest(ivars);
			RefreshProviderHealth(ivars);
		};
		if (ivars->gate && !ivars->gate->OnQueue()) ivars->gate->DispatchSync(observe);
		else observe();
	}
	return SetPowerState(powerFlags, SUPERDISPATCH);
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
	ObservePCICommand(ivars);
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
		ivars->canaryHealthy = !ivars->counterSaturated;
	} else {
		if (canCount) ++ivars->failures;
		SaturatingIncrement(&ivars->consecutiveFailures, &ivars->counterSaturated);
		ivars->canaryHealthy = false;
	}
	if (ivars->successes != ivars->attempts - ivars->failures) {
		ivars->counterSaturated = true;
		ivars->canaryHealthy = false;
	}
	if (action && (!ivars->fullPowerRequested ||
		(ivars->powerRequestAccepted && !ivars->powerRequestConfirmed &&
		 ivars->powerRequestAttempts < kMaxPowerRequestAttempts)))
		(void)RequestPowerResidency(this, ivars);

	if (!ivars->timerEnabled || !ivars->timer) { RefreshProviderHealth(ivars); return; }
	ivars->timerError = (int32_t)ivars->timer->WakeAtTime(
		kIOTimerClockUptimeRaw, now + uint64_t(kUSB4AMD744C.intervalMS) * 1000000ull,
		uint64_t(kUSB4AMD744C.leewayMS) * 1000000ull);
	RefreshProviderHealth(ivars);
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
		IOService* invalid = service.detach();
		if (invalid) invalid->release();
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

kern_return_t TinyGPUDriver::MMIORead(uint64_t leaseID, uint32_t bar, uint64_t offset, uint32_t width, uint32_t* out) {
	if (!leaseID || !out || bar >= 6 || !ValidMMIOWidth(offset, width) || !ivars->gate) return kIOReturnBadArgument;
	return ivars->gate->RunAction(^{
		if (!ivars->pci || ivars->activeLeases != 1 || leaseID != ivars->leaseID || ivars->state != kActiveHealthy)
			return kIOReturnNotReady;
		kern_return_t err = EnsureMMIOMapping(this, ivars, bar);
		if (err) return err;
		if (offset > ivars->mmioSize[bar] || width > ivars->mmioSize[bar] - offset) return kIOReturnBadArgument;
		volatile uint8_t* address = (volatile uint8_t*)(uintptr_t)ivars->mmioMap[bar]->GetAddress() + offset;
		if (width == 1) *out = *(volatile uint8_t*)address;
		else if (width == 2) *out = *(volatile uint16_t*)address;
		else *out = *(volatile uint32_t*)address;
		return kIOReturnSuccess;
	});
}

kern_return_t TinyGPUDriver::MMIOWrite(uint64_t leaseID, uint32_t bar, uint64_t offset, uint32_t width, uint32_t value) {
	if (!leaseID || bar >= 6 || !ValidMMIOWidth(offset, width) || !ivars->gate) return kIOReturnBadArgument;
	return ivars->gate->RunAction(^{
		if (!ivars->pci || ivars->activeLeases != 1 || leaseID != ivars->leaseID || ivars->state != kActiveHealthy)
			return kIOReturnNotReady;
		kern_return_t err = EnsureMMIOMapping(this, ivars, bar);
		if (err) return err;
		if (offset > ivars->mmioSize[bar] || width > ivars->mmioSize[bar] - offset) return kIOReturnBadArgument;
		volatile uint8_t* address = (volatile uint8_t*)(uintptr_t)ivars->mmioMap[bar]->GetAddress() + offset;
		if (width == 1) *(volatile uint8_t*)address = (uint8_t)value;
		else if (width == 2) *(volatile uint16_t*)address = (uint16_t)value;
		else *(volatile uint32_t*)address = value;
		return kIOReturnSuccess;
	});
}

kern_return_t TinyGPUDriver::ResetDevice() {
	if (!ivars->gate || !ivars->timer) return kIOReturnNotReady;
	__block kern_return_t err = kIOReturnSuccess;
	ivars->gate->DispatchSync(^{
		if (!ivars->pci || !ProviderResidencyReady(ivars)) err = kIOReturnNotReady;
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
		ReleaseMMIOMappings(ivars);
		ReleaseBARResidency(ivars);
		err = ivars->pci->Reset(kIOPCIDeviceResetTypeFunctionReset);
		if (!err) {
			uint32_t identity = 0;
			ivars->pci->ConfigurationRead32(kIOPCIConfigurationOffsetVendorID, &identity);
			ivars->lastIdentity = identity;
			if (identity != kUSB4AMD744C.identity) err = kIOReturnNoDevice;
			else if (!(err = EnablePCICommand(ivars))) err = AcquireBARResidency(this, ivars);
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
		if (ivars->state != kActiveHealthy || ivars->counterSaturated || !ProviderResidencyReady(ivars)) return kIOReturnNotReady;
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
		ReleaseMMIOMappings(ivars);
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

kern_return_t TinyGPUDriver::GetPowerResidencyStatus(char* out, size_t* length) {
	if (!out || !length || !*length) return kIOReturnBadArgument;
	struct Snapshot {
		bool fullPowerRequested, requestAccepted, requestConfirmed, releaseAttempted;
		bool barRequested, barActive, pciCommandRequested, pciCommandConfirmed, publishable;
		uint64_t generation, requestAttempts, lastRequest, transitions, unexpectedDowngrades, lastTransition, lastCanarySuccess;
		uint64_t barBytes, lastPCICommand;
		uint32_t desiredFlags, observedFlags, lastIdentity, stopBusyLeases, stopBusyBars, stopBusyDMA, barType;
		uint32_t pciCommandRequired, pciCommandBefore, pciCommandAfter;
		int32_t overrideProbePreJoinError, overrideProbePostJoinError, requestError, releaseError, barError, pciCommandError;
	};
	__block Snapshot snapshot{};
	auto capture = ^{
		snapshot.fullPowerRequested = ivars->fullPowerRequested;
		snapshot.requestAccepted = ivars->powerRequestAccepted;
		snapshot.requestConfirmed = ivars->powerRequestConfirmed;
		snapshot.releaseAttempted = ivars->powerReleaseAttempted;
		snapshot.barRequested = ivars->barResidencyRequested;
		snapshot.barActive = ivars->barResidencyActive;
		snapshot.pciCommandRequested = ivars->pciCommandRequested;
		snapshot.pciCommandConfirmed = ivars->pciCommandConfirmed;
		snapshot.publishable = ivars->state == kActiveHealthy && ProviderResidencyReady(ivars);
		snapshot.generation = ivars->generation;
		snapshot.requestAttempts = ivars->powerRequestAttempts;
		snapshot.lastRequest = ivars->lastPowerRequestTick;
		snapshot.transitions = ivars->powerTransitions;
		snapshot.unexpectedDowngrades = ivars->unexpectedPowerDowngrades;
		snapshot.lastTransition = ivars->lastPowerTransition;
		snapshot.lastCanarySuccess = ivars->lastSuccess;
		snapshot.desiredFlags = ivars->desiredPowerFlags;
		snapshot.observedFlags = ivars->lastObservedPowerFlags;
		snapshot.lastIdentity = ivars->lastIdentity;
		snapshot.stopBusyLeases = ivars->stopBusyLeases;
		snapshot.stopBusyBars = ivars->stopBusyBars;
		snapshot.stopBusyDMA = ivars->stopBusyDMA;
		snapshot.barBytes = ivars->barResidencyBytes;
		snapshot.barType = ivars->barResidencyType;
		snapshot.lastPCICommand = ivars->lastPCICommandTick;
		snapshot.pciCommandRequired = kRequiredPCICommand;
		snapshot.pciCommandBefore = ivars->pciCommandBefore;
		snapshot.pciCommandAfter = ivars->pciCommandAfter;
		snapshot.overrideProbePreJoinError = ivars->overrideProbePreJoinError;
		snapshot.overrideProbePostJoinError = ivars->overrideProbePostJoinError;
		snapshot.requestError = ivars->powerRequestError;
		snapshot.releaseError = ivars->powerReleaseError;
		snapshot.barError = ivars->barResidencyError;
		snapshot.pciCommandError = ivars->pciCommandError;
	};
	if (ivars->gate && !ivars->gate->OnQueue()) ivars->gate->DispatchSync(capture);
	else capture();

	int written = snprintf(out, *length,
		"{\"schema\":\"tinygpu.power-residency.v4\",\"provider_generation\":%llu,"
		"\"policy_id\":\"%s\",\"full_power_requested\":%s,\"power_request_accepted\":%s,"
		"\"power_request_confirmed\":%s,\"power_request_attempts\":%llu,"
		"\"last_power_request_monotonic_ns\":%llu,"
		"\"power_release_attempted\":%s,\"desired_power_flags\":%u,\"last_observed_power_flags\":%u,"
		"\"override_probe_prejoin_error\":%d,\"override_probe_postjoin_error\":%d,"
		"\"power_request_error\":%d,\"power_release_error\":%d,\"transition_count\":%llu,"
		"\"unexpected_downgrade_count\":%llu,"
		"\"last_transition_monotonic_ns\":%llu,\"last_canary_identity_dword\":\"0x%08x\","
		"\"last_canary_success_monotonic_ns\":%llu,\"stop_busy_leases\":%u,\"stop_busy_bars\":%u,"
		"\"stop_busy_dma\":%u,\"bar_residency_policy_id\":\"%s\",\"bar_residency_requested\":%s,"
		"\"bar_residency_active\":%s,\"bar_residency_bar\":%u,\"bar_residency_type\":%u,"
		"\"bar_residency_bytes\":%llu,\"bar_residency_error\":%d,"
		"\"pci_command_policy_id\":\"%s\",\"pci_command_requested\":%s,\"pci_command_confirmed\":%s,"
		"\"pci_command_required_mask\":%u,\"pci_command_before\":%u,\"pci_command_after\":%u,"
		"\"last_pci_command_monotonic_ns\":%llu,\"pci_command_error\":%d,\"publishable\":%s}",
		(unsigned long long)snapshot.generation, kPowerResidencyPolicy,
		snapshot.fullPowerRequested ? "true" : "false", snapshot.requestAccepted ? "true" : "false",
		snapshot.requestConfirmed ? "true" : "false", (unsigned long long)snapshot.requestAttempts,
		(unsigned long long)snapshot.lastRequest, snapshot.releaseAttempted ? "true" : "false",
		snapshot.desiredFlags, snapshot.observedFlags, snapshot.overrideProbePreJoinError,
		snapshot.overrideProbePostJoinError, snapshot.requestError, snapshot.releaseError,
		(unsigned long long)snapshot.transitions,
		(unsigned long long)snapshot.unexpectedDowngrades, (unsigned long long)snapshot.lastTransition,
		snapshot.lastIdentity, (unsigned long long)snapshot.lastCanarySuccess, snapshot.stopBusyLeases,
		snapshot.stopBusyBars, snapshot.stopBusyDMA, kBARResidencyPolicy,
		snapshot.barRequested ? "true" : "false", snapshot.barActive ? "true" : "false", kKeeperBAR,
		snapshot.barType, (unsigned long long)snapshot.barBytes, snapshot.barError, kPCICommandPolicy,
		snapshot.pciCommandRequested ? "true" : "false", snapshot.pciCommandConfirmed ? "true" : "false",
		snapshot.pciCommandRequired, snapshot.pciCommandBefore, snapshot.pciCommandAfter,
		(unsigned long long)snapshot.lastPCICommand, snapshot.pciCommandError,
		snapshot.publishable ? "true" : "false");
	if (written < 0 || (size_t)written >= *length) return kIOReturnNoSpace;
	*length = (size_t)written;
	return kIOReturnSuccess;
}
