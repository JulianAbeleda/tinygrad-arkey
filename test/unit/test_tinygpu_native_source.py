from pathlib import Path

ROOT = Path(__file__).parents[2]
NATIVE = ROOT / "extra/usbgpu/tbgpu/installer/TinyGPUDriverExtension"
DRIVER = (NATIVE / "TinyGPUDriver.cpp").read_text()
CLIENT = (NATIVE / "TinyGPUDriverUserClient.cpp").read_text()
IIG = (NATIVE / "TinyGPUDriver.iig").read_text()
CLIENT_IIG = (NATIVE / "TinyGPUDriverUserClient.iig").read_text()


def body(source:str, start:str, end:str) -> str:
  return source[source.index(start):source.index(end, source.index(start))]


def test_keeper_is_one_shot_uptime_raw_and_never_resets():
  tick = body(DRIVER, "void IMPL(TinyGPUDriver, KeepaliveTimer)", "kern_return_t TinyGPUDriver::NewUserClient_Impl")
  assert "CLOCK_UPTIME_RAW" in DRIVER and "kIOTimerClockUptimeRaw" in tick
  assert "intervalMS) * 1000000ull" in tick and "leewayMS) * 1000000ull" in tick
  assert "Reset(" not in tick and DRIVER.count("->Reset(") == 1
  assert "ConfigurationRead32(kIOPCIConfigurationOffsetVendorID" in tick


def test_timer_disable_and_cancel_are_completion_drained_on_provider_gate():
  drain = body(DRIVER, "static kern_return_t DrainTimer", "bool TinyGPUDriver::init")
  assert "gate->RunAction" in drain
  assert drain.index("SetEnableWithCompletion(false") < drain.index("SleepWithDeadline(disableEvent")
  assert drain.index("timer->Cancel") < drain.index("SleepWithDeadline(cancelEvent")
  stop = body(DRIVER, "kern_return_t TinyGPUDriver::Stop_Impl", "void IMPL(TinyGPUDriver, KeepaliveTimer)")
  assert stop.index("DrainTimer") < stop.index("ReleasePowerResidency") < stop.index("ivars->pci->Close")
  assert "stopBusyLeases = ivars->activeLeases" in stop
  assert "stopBusyBars = ivars->activeBars" in stop and "stopBusyDMA = ivars->activeDMA" in stop


def test_every_pci_operation_lives_in_provider_and_uses_the_serial_gate():
  assert "->pci" not in CLIENT and "IOPCIDevice" not in CLIENT
  for name in ("CfgRead", "CfgWrite", "ResetDevice", "MapBar", "SetupDMA", "CompleteDMA"):
    section = body(DRIVER, f"TinyGPUDriver::{name}", "\n}")
    assert "ivars->gate" in section
  assert "IODispatchQueue::Create(\"tinygpu.provider.gate\"" in DRIVER


def test_resource_creation_and_accounting_share_the_lease_checked_gate():
  map_bar = body(DRIVER, "TinyGPUDriver::MapBar", "static kern_return_t WriteDMASegments")
  setup_dma = body(DRIVER, "TinyGPUDriver::SetupDMA", "TinyGPUDriver::CompleteDMA")
  for section, counter in ((map_bar, "activeBars"), (setup_dma, "activeDMA")):
    assert "leaseID != ivars->leaseID" in section
    assert f"++ivars->{counter}" in section
  assert "RetainBarMapping" not in DRIVER + CLIENT + IIG
  assert "RetainDMAAllocation" not in DRIVER + CLIENT + IIG


def test_user_client_serializes_handshake_lease_dma_and_teardown():
  assert "IODispatchQueue::Create(\"tinygpu.userclient.gate\"" in CLIENT
  assert CLIENT.count("!ivars->gate->OnQueue()") >= 3
  assert "StopClientState" in CLIENT and "state->stopping = true" in CLIENT
  assert "outputSize < sizeof(uint64_t) * 2 * 33" in CLIENT
  assert "segCount > 32" in CLIENT
  release = body(CLIENT, "static void ReleaseClientResources", "static void StopClientState")
  assert release.index("CompleteDMA") < release.index("ReleaseDMAAllocation")


def test_keepalive_status_supports_inline_and_descriptor_outputs():
  status = body(CLIENT, "selector == TinyGPURPC::KeepaliveStatus", "selector == TinyGPURPC::LeaseAcquire")
  assert "structureOutputDescriptor" in status and "CreateMapping" in status
  assert "structureOutputMaximumSize" in status
  assert "OSData::withBytes(output, length)" in status
  assert "available > 4096" in status


def test_power_residency_request_notification_and_release_are_distinct():
  confirmation = body(DRIVER, "static bool FullPowerTransitionConfirmsRequest", "static bool PowerResidencyReady")
  request = body(DRIVER, "static kern_return_t RequestPowerResidency", "static kern_return_t ReleasePowerResidency")
  release = body(DRIVER, "static kern_return_t ReleasePowerResidency", "static kern_return_t DrainTimer")
  start = body(DRIVER, "kern_return_t TinyGPUDriver::Start_Impl", "kern_return_t TinyGPUDriver::Stop_Impl")
  stop = body(DRIVER, "kern_return_t TinyGPUDriver::Stop_Impl", "kern_return_t TinyGPUDriver::SetPowerState_Impl")
  notification = body(DRIVER, "kern_return_t TinyGPUDriver::SetPowerState_Impl", "void IMPL(TinyGPUDriver, KeepaliveTimer)")
  keepalive = body(DRIVER, "void IMPL(TinyGPUDriver, KeepaliveTimer)", "kern_return_t TinyGPUDriver::NewUserClient_Impl")
  assert "SetPowerOverride(true)" not in DRIVER
  assert "ChangePowerState(kFullPowerFlags)" in request and "SetPowerOverride(false)" in request
  assert request.index("powerRequestAccepted = false") < request.index("ChangePowerState(kFullPowerFlags)")
  assert request.index("powerRequestConfirmed = false") < request.index("ChangePowerState(kFullPowerFlags)")
  assert request.index("powerRequestAccepted = requestError == kIOReturnSuccess") < \
    request.index("powerRequestConfirmed = FullPowerTransitionConfirmsRequest(state)")
  for token in ("powerRequestAccepted", "!state->powerRequestError", "powerTransitions",
                "lastObservedPowerFlags == kFullPowerFlags", "lastPowerTransition > state->lastPowerRequestTick"):
    assert token in confirmation
  assert "ChangePowerState(kReleasedPowerFlags)" in release and "SetPowerOverride" not in release
  assert "kReleasedPowerFlags = kIOServicePowerCapabilityOff" in DRIVER and "powerReleaseAttempted" in release
  assert "SetPowerOverride(false)" in start and "RequestPowerResidency" not in start and "ChangePowerState" not in start
  assert "action &&" in keepalive and "RequestPowerResidency(this, ivars)" in keepalive
  assert "lastSuccess > state->lastPowerRequestTick" in DRIVER
  assert "lastObservedPowerFlags = powerFlags" in notification
  assert "powerRequestConfirmed = FullPowerTransitionConfirmsRequest(ivars)" in notification
  assert "SetPowerState(powerFlags, SUPERDISPATCH)" in notification
  assert stop.count("\n\treturn ") == 1 and "Stop(provider, SUPERDISPATCH)" in stop
  assert "stopBusyLeases" in stop and "DrainTimer" in stop and "ReleaseMMIOMappings" in stop
  assert "PowerResidencyReady(ivars)" in DRIVER


def test_power_residency_status_is_separate_from_frozen_keepalive_status():
  keepalive = body(DRIVER, "TinyGPUDriver::GetKeepaliveStatus", "TinyGPUDriver::GetPowerResidencyStatus")
  power = DRIVER[DRIVER.index("TinyGPUDriver::GetPowerResidencyStatus"):]
  assert "tinygpu.keepalive.v1" in keepalive and "tinygpu.power-residency.v2" not in keepalive
  assert "tinygpu.power-residency.v2" in power and "override_probe_prejoin_error" in power
  assert "power_request_attempts" in power and "last_power_request_monotonic_ns" in power
  assert "unexpected_downgrade_count" in power and "publishable" in power
  assert "PowerResidencyStatus = 10" in CLIENT_IIG


def test_new_user_client_transfer_is_explicit_and_state_admission_is_gated():
  create = body(DRIVER, "TinyGPUDriver::NewUserClient_Impl", "TinyGPUDriver::CfgRead")
  assert "ivars->gate->DispatchSync" in create and "service.attach()" in create
  assert "IO_FOR_ANALYZER(service.get()->release())" not in create
  assert create.index("IOService* invalid = service.detach()") < create.index("invalid->release()")
  assert "service.detach()" in create and "IO_FOR_ANALYZER(client->release())" in create
