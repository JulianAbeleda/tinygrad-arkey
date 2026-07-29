#include "TinyGPUDriverUserClient.h"
#include "TinyGPUDriver.h"
#include <DriverKit/DriverKit.h>
#include <PCIDriverKit/PCIDriverKit.h>

struct TinyGPUDriverUserClient_IVars
{
	TinyGPUDriver* provider = nullptr;
	IODispatchQueue* gate = nullptr;

	TinyGPUCreateDMAResp *dmas = nullptr;
	size_t dmaCount = 0;
	size_t dmaCap = 0;
	uint64_t leaseID = 0;
	uint32_t barMappings = 0;
	bool handshaken = false;
	bool stopping = false;

	int ensureDMACap(size_t need)
	{
		// not thread-safe
		if (need <= dmaCap) return 0;

		size_t newCap = dmaCap ? dmaCap * 2 : 16;
		while (newCap < need) newCap *= 2;

		auto *newArr = IONewZero(TinyGPUCreateDMAResp, newCap);
		if (!newArr) return -kIOReturnNoMemory;

		if (dmas && dmaCount) {
			memcpy(newArr, dmas, dmaCount * sizeof(TinyGPUCreateDMAResp));
		}

		IOSafeDeleteNULL(dmas, TinyGPUCreateDMAResp, dmaCap);
		dmas = newArr;
		dmaCap = newCap;
		return 0;
	}
};

static void ReleaseClientResources(TinyGPUDriverUserClient_IVars* state) {
	if (!state) return;
	for (size_t i = 0; i < state->dmaCount; i++) {
		auto& dma = state->dmas[i];
		if (dma.dmaCmd) {
			if (state->provider) (void)state->provider->CompleteDMA(dma.dmaCmd);
			dma.dmaCmd->release();
			dma.dmaCmd = nullptr;
		}
		if (dma.sharedBuf) { dma.sharedBuf->release(); dma.sharedBuf = nullptr; }
		if (state->provider) state->provider->ReleaseDMAAllocation();
	}
	state->dmaCount = 0;
	while (state->barMappings && state->provider) {
		state->provider->ReleaseBarMapping();
		--state->barMappings;
	}
}

static void StopClientState(TinyGPUDriverUserClient_IVars* state) {
	if (!state || state->stopping) return;
	state->stopping = true;
	ReleaseClientResources(state);
	if (state->leaseID && state->provider) (void)state->provider->ReleaseWorkloadLease(state->leaseID);
	state->leaseID = 0;
	IOSafeDeleteNULL(state->dmas, TinyGPUCreateDMAResp, state->dmaCap);
	state->dmas = nullptr;
}

bool TinyGPUDriverUserClient::init()
{
	auto ok = super::init();
	if (!ok) return false;

	ivars = IONewZero(TinyGPUDriverUserClient_IVars, 1);
	if (!ivars) return false;
	if (IODispatchQueue::Create("tinygpu.userclient.gate", 0, 0, &ivars->gate)) {
		IOSafeDeleteNULL(ivars, TinyGPUDriverUserClient_IVars, 1);
		return false;
	}
	return true;
}

void TinyGPUDriverUserClient::free()
{
	if (ivars) {
		if (ivars->provider) {
			auto cleanup = ^{ StopClientState(ivars); };
			if (ivars->gate && !ivars->gate->OnQueue()) ivars->gate->DispatchSync(cleanup); else cleanup();
			ivars->provider->release();
			ivars->provider = nullptr;
		}
		if (ivars->gate) { ivars->gate->release(); ivars->gate = nullptr; }
		IOSafeDeleteNULL(ivars, TinyGPUDriverUserClient_IVars, 1);
	}
	super::free();
}

kern_return_t TinyGPUDriverUserClient::Start_Impl(IOService* in_provider)
{
	kern_return_t err = kIOReturnSuccess;
	if (!in_provider) {
		os_log(OS_LOG_DEFAULT, "tinygpu: provider is null");
		err = kIOReturnBadArgument;
		goto error;
	}

	err = Start(in_provider, SUPERDISPATCH);
	if (err) {
		os_log(OS_LOG_DEFAULT, "tinygpu: failed to start super (%d)", err);
		goto error;
	}

	ivars->provider = OSDynamicCast(TinyGPUDriver, in_provider);
	if (!ivars->provider) { err = kIOReturnNotAttached; goto error; }
	ivars->provider->retain();
	return 0;

error:
	if (ivars->provider) { ivars->provider->release(); ivars->provider = nullptr; }
	return err;
}

kern_return_t TinyGPUDriverUserClient::Stop_Impl(IOService* in_provider)
{
	if (ivars) {
		auto cleanup = ^{ StopClientState(ivars); };
		if (ivars->gate && !ivars->gate->OnQueue()) ivars->gate->DispatchSync(cleanup); else cleanup();
		if (ivars->provider) { ivars->provider->release(); ivars->provider = nullptr; }
	}

	return Stop(in_provider, SUPERDISPATCH);
}

kern_return_t TinyGPUDriverUserClient::ExternalMethod(uint64_t selector, IOUserClientMethodArguments* args, const IOUserClientMethodDispatch* in_dispatch, OSObject* in_target, void* in_reference)
{
	if (!args || !ivars) return kIOReturnNotAttached;
	if (ivars->gate && !ivars->gate->OnQueue()) {
		__block kern_return_t result = kIOReturnError;
		ivars->gate->DispatchSync(^{ result = ExternalMethod(selector, args, in_dispatch, in_target, in_reference); });
		return result;
	}
	if (ivars->stopping || !ivars->provider) return kIOReturnNotAttached;
	kern_return_t err = 0;

	os_log(OS_LOG_DEFAULT, "tinygpu: rpc (%llu) in:%d, out:%d", selector, args->scalarInputCount, args->scalarOutputCount);

	if (selector != TinyGPURPC::Handshake && !ivars->handshaken) return kIOReturnNotPermitted;
	if (selector == TinyGPURPC::ReadCfg) {
		if (!ivars->leaseID) return kIOReturnNotPermitted;
		if (args->scalarInputCount != 2 or args->scalarOutputCount < 1) return kIOReturnBadArgument;

		uint32_t off = uint32_t(args->scalarInput[0]);
		uint32_t size = uint32_t(args->scalarInput[1]);

		uint32_t val = 0;
		err = ivars->provider->CfgRead(off, size, &val);
		os_log(OS_LOG_DEFAULT, "tinygpu: read cfg off:%x sz:%d, val:%x", off, size, val);

		if (!err) {
			args->scalarOutput[0] = val;
			args->scalarOutputCount = 1;
		}
		return err;
	} else if (selector == TinyGPURPC::WriteCfg) {
		if (!ivars->leaseID) return kIOReturnNotPermitted;
		if (args->scalarInputCount != 3) return kIOReturnBadArgument;

		uint32_t off = uint32_t(args->scalarInput[0]);
		uint32_t size = uint32_t(args->scalarInput[1]);
		uint32_t val = uint32_t(args->scalarInput[2]);

		os_log(OS_LOG_DEFAULT, "tinygpu: wr cfg off:%x sz:%d, val:%x", off, size, val);
		return ivars->provider->CfgWrite(off, size, val);
	} else if (selector == TinyGPURPC::MMIORead) {
		if (!ivars->leaseID) return kIOReturnNotPermitted;
		if (args->scalarInputCount != 3 || args->scalarOutputCount < 1) return kIOReturnBadArgument;
		const uint32_t bar = uint32_t(args->scalarInput[0]);
		const uint64_t off = args->scalarInput[1];
		const uint32_t size = uint32_t(args->scalarInput[2]);
		uint32_t value = 0;
		err = ivars->provider->MMIORead(ivars->leaseID, bar, off, size, &value);
		if (!err) { args->scalarOutput[0] = value; args->scalarOutputCount = 1; }
		return err;
	} else if (selector == TinyGPURPC::MMIOWrite) {
		if (!ivars->leaseID) return kIOReturnNotPermitted;
		if (args->scalarInputCount != 4) return kIOReturnBadArgument;
		return ivars->provider->MMIOWrite(ivars->leaseID, uint32_t(args->scalarInput[0]), args->scalarInput[1],
		                                  uint32_t(args->scalarInput[2]), uint32_t(args->scalarInput[3]));
	} else if (selector == TinyGPURPC::Reset) {
		if (ivars->leaseID) return kIOReturnNotPermitted;
		if (args->scalarInputCount || args->scalarOutputCount) return kIOReturnBadArgument;
		os_log(OS_LOG_DEFAULT, "tinygpu: reset");
		return ivars->provider->ResetDevice();
	} else if (selector == TinyGPURPC::PrepareDMA) {
		if (!ivars->leaseID) return kIOReturnNotPermitted;
		// both input and output buffers must be >= 4097 bytes for IOMemoryDescriptor
		if (!args->structureInputDescriptor || !args->structureOutputDescriptor) {
			os_log(OS_LOG_DEFAULT, "tinygpu: PrepareDMA requires buffers >= 4097 bytes");
			return kIOReturnBadArgument;
		}
		if (ivars->ensureDMACap(ivars->dmaCount + 1)) return kIOReturnNoMemory;

		uint64_t size = 0, outputSize = 0;
		args->structureInputDescriptor->GetLength(&size);
		args->structureOutputDescriptor->GetLength(&outputSize);
		if (!size || outputSize < sizeof(uint64_t) * 2 * 33) return kIOReturnNoSpace;

		IODMACommand* dmaCmd = nullptr;
		IOAddressSegment segments[32];
		uint32_t segCount = 32;
		err = ivars->provider->SetupDMA(ivars->leaseID, args->structureInputDescriptor, size, &dmaCmd, segments, &segCount);
		if (err) return err;
		if (segCount > 32 || outputSize < sizeof(uint64_t) * 2 * (segCount + 1)) {
			(void)ivars->provider->CompleteDMA(dmaCmd); dmaCmd->release(); ivars->provider->ReleaseDMAAllocation();
			return kIOReturnNoSpace;
		}

		// write physical addresses to output: [addr0, len0, addr1, len1, ..., 0, 0]
		IOMemoryMap* outMap = nullptr;
		err = args->structureOutputDescriptor->CreateMapping(0, 0, 0, 0, 0, &outMap);
		if (err || !outMap) {
			os_log(OS_LOG_DEFAULT, "tinygpu: output map failed err=%d", err);
			(void)ivars->provider->CompleteDMA(dmaCmd);
			dmaCmd->release();
			ivars->provider->ReleaseDMAAllocation();
			return err ?: kIOReturnError;
		}

		uint64_t* out = (uint64_t*)outMap->GetAddress();
		for (uint32_t i = 0; i < segCount; i++) { out[i * 2] = segments[i].address; out[i * 2 + 1] = segments[i].length; }
		out[segCount * 2] = 0; out[segCount * 2 + 1] = 0;
		outMap->release();

		os_log(OS_LOG_DEFAULT, "tinygpu: PrepareDMA size=%llu segs=%u", size, segCount);
		ivars->dmas[ivars->dmaCount++] = {nullptr, dmaCmd};
		return kIOReturnSuccess;
	} else if (selector == TinyGPURPC::Handshake) {
		if (ivars->handshaken) return kIOReturnExclusiveAccess;
		if (args->scalarInputCount != 3 || args->scalarOutputCount < 2) return kIOReturnBadArgument;
		if (args->scalarInput[0] != 1 || args->scalarInput[1] != 0) return kIOReturnUnsupported;
		const uint64_t required = args->scalarInput[2];
		const uint64_t caps = 11; // keepalive status + single workload lease + power-residency status
		if (required & ~caps) return kIOReturnUnsupported;
		args->scalarOutput[0] = 1; args->scalarOutput[1] = caps; args->scalarOutputCount = 2;
		ivars->handshaken = true;
		return kIOReturnSuccess;
	} else if (selector == TinyGPURPC::KeepaliveStatus) {
		if (args->scalarInputCount || args->scalarOutputCount) return kIOReturnBadArgument;
		if (args->structureOutputDescriptor) {
			uint64_t available = 0; args->structureOutputDescriptor->GetLength(&available);
			if (!available || available > 4096) return kIOReturnNoSpace;
			IOMemoryMap* map = nullptr; err = args->structureOutputDescriptor->CreateMapping(0, 0, 0, 0, 0, &map);
			if (err || !map) return err ?: kIOReturnError;
			size_t length = (size_t)available;
			err = ivars->provider->GetKeepaliveStatus((char*)map->GetAddress(), &length);
			map->release();
			return err;
		}

		// IOConnectCallStructMethod carries outputs up to 4096 bytes inline as
		// OSData. Larger outputs arrive as descriptors, but the status contract
		// deliberately caps its JSON at 4096 bytes.
		const uint64_t available = args->structureOutputMaximumSize;
		if (!available || available > 4096) return kIOReturnNoSpace;
		char output[4096] = {};
		size_t length = (size_t)available;
		err = ivars->provider->GetKeepaliveStatus(output, &length);
		if (err) return err;
		args->structureOutput = OSData::withBytes(output, length);
		return args->structureOutput ? kIOReturnSuccess : kIOReturnNoMemory;
	} else if (selector == TinyGPURPC::PowerResidencyStatus) {
		if (args->scalarInputCount || args->scalarOutputCount) return kIOReturnBadArgument;
		if (args->structureOutputDescriptor) {
			uint64_t available = 0; args->structureOutputDescriptor->GetLength(&available);
			if (!available || available > 4096) return kIOReturnNoSpace;
			IOMemoryMap* map = nullptr; err = args->structureOutputDescriptor->CreateMapping(0, 0, 0, 0, 0, &map);
			if (err || !map) return err ?: kIOReturnError;
			size_t length = (size_t)available;
			err = ivars->provider->GetPowerResidencyStatus((char*)map->GetAddress(), &length);
			map->release();
			return err;
		}
		const uint64_t available = args->structureOutputMaximumSize;
		if (!available || available > 4096) return kIOReturnNoSpace;
		char output[4096] = {};
		size_t length = (size_t)available;
		err = ivars->provider->GetPowerResidencyStatus(output, &length);
		if (err) return err;
		args->structureOutput = OSData::withBytes(output, length);
		return args->structureOutput ? kIOReturnSuccess : kIOReturnNoMemory;
	} else if (selector == TinyGPURPC::LeaseAcquire) {
		if (args->scalarInputCount || args->scalarOutputCount < 1 || ivars->leaseID) return kIOReturnBadArgument;
		err = ivars->provider->AcquireWorkloadLease(&ivars->leaseID);
		if (!err) { args->scalarOutput[0] = ivars->leaseID; args->scalarOutputCount = 1; }
		return err;
	} else if (selector == TinyGPURPC::LeaseRelease) {
		if (args->scalarInputCount != 1 || !ivars->leaseID || args->scalarInput[0] != ivars->leaseID) return kIOReturnBadArgument;
		ReleaseClientResources(ivars);
		err = ivars->provider->ReleaseWorkloadLease(ivars->leaseID);
		if (!err) ivars->leaseID = 0;
		return err;
	}

	return kIOReturnUnsupported;
}

kern_return_t IMPL(TinyGPUDriverUserClient, CopyClientMemoryForType)
{
	if (!memory || !ivars) return kIOReturnBadArgument;
	if (ivars->gate && !ivars->gate->OnQueue()) {
		__block kern_return_t result = kIOReturnError;
		ivars->gate->DispatchSync(^{ result = CopyClientMemoryForType(type, options, memory); });
		return result;
	}
	if (ivars->stopping || !ivars->provider) return kIOReturnNotAttached;
	if (!ivars->leaseID) return kIOReturnNotPermitted;

	// bar handling, type is bar num
	if (type < 6) {
		uint32_t bar = (uint32_t)type;
		kern_return_t err = ivars->provider->MapBar(ivars->leaseID, bar, memory);
		if (!err) ++ivars->barMappings;
		return err;
	}

	// dma handling, type is size
	if (ivars->ensureDMACap(ivars->dmaCount + 1)) {
		os_log(OS_LOG_DEFAULT, "tinygpu: cannot grow dma array");
		return kIOReturnNoMemory;
	}

	TinyGPUCreateDMAResp buf{};
	kern_return_t err = ivars->provider->CreateDMA(ivars->leaseID, type, &buf);
	if (err) return err;

	ivars->dmas[ivars->dmaCount++] = buf;
	*memory = buf.sharedBuf;
	return 0;
}
