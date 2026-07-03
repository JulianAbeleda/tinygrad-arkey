from tinygrad.runtime.support.am.ip.common import *

class AM_SOC(AM_IP):
  def init_sw(self):
    self.module = import_soc(self.adev.ip_ver[am.GC_HWIP])
    self.ih_clients = am.enum_soc21_ih_clientid if (ih_soc21:=self.adev.ip_ver[am.GC_HWIP][0] >= 11) else am.enum_soc15_ih_clientid

    self.gfx_ih_clients = [am.SOC21_IH_CLIENTID_GRBM_CP, am.SOC21_IH_CLIENTID_GFX] \
      if ih_soc21 else [am.SOC15_IH_CLIENTID_GRBM_CP] + [getattr(am, f'SOC15_IH_CLIENTID_SE{i}SH') for i in range(4)]
    self.sdma_ih_clients = [] if ih_soc21 else [getattr(am, f'SOC15_IH_CLIENTID_SDMA{i}') for i in range(8)]

    def _ih_srcs(pref:str, hwip:int) -> dict[int, str]:
      return {getattr(am, k): k[off+9:] for k in dir(am) if k.startswith(f'{pref}_{self.adev.ip_ver[hwip][0]}') and (off:=k.find('__SRCID__')) != -1}

    gfx_srcs, sdma_srcs = _ih_srcs('GFX', am.GC_HWIP), _ih_srcs('SDMA0', am.SDMA0_HWIP)
    self.ih_srcs_names:dict[int, dict[int, str]] = {**{k: gfx_srcs for k in self.gfx_ih_clients}, **{k: sdma_srcs for k in self.sdma_ih_clients}}

  def init_hw(self):
    if getenv("AM_NBIO_REMAP_HDP", 0) and hasattr(self.adev, "regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL"):
      mem_reg = self.adev.regBIF_BX0_REMAP_HDP_MEM_FLUSH_CNTL.addr[0]
      reg_reg = mem_reg + 1
      old_mem, old_reg = self.adev.rreg(mem_reg), self.adev.rreg(reg_reg)
      self.adev.wreg(mem_reg, 0x7f000)
      self.adev.wreg(reg_reg, 0x7f004)
      if getenv("AM_PSP_TRACE", 0):
        print(f"am {self.adev.devfmt}: SOC remap HDP mem {old_mem:#x}->0x7f000 reg {old_reg:#x}->0x7f004", flush=True)
    if getenv("AM_NBIO_CLEAR_STRAP2", 0) and hasattr(self.adev, "regRCC_DEV0_EPF2_STRAP2"):
      old = self.adev.regRCC_DEV0_EPF2_STRAP2.read()
      new = old & ~(1 << 7)
      self.adev.regRCC_DEV0_EPF2_STRAP2.write(new)
      if getenv("AM_PSP_TRACE", 0): print(f"am {self.adev.devfmt}: SOC clear strap2 old={old:#x} new={new:#x}", flush=True)
    if self.adev.ip_ver[am.NBIO_HWIP] in {(7,9,0), (7,9,1)}:
      self.adev.regXCC_DOORBELL_FENCE.write(0x0)
      for aid in range(1, self.adev.gmc.vmhubs):
        self.adev.indirect_wreg_pcie(self.adev.regXCC_DOORBELL_FENCE.addr[0], self.adev.regXCC_DOORBELL_FENCE.encode(shub_slv_mode=1), aid=aid)
      if getenv("AM_NBIO_PCIE_BIFC", 0):
        self.adev.indirect_wreg_pcie(self.adev.regBIFC_GFX_INT_MONITOR_MASK.addr[0], 0x7ff)
        self.adev.indirect_wreg_pcie(self.adev.regBIFC_DOORBELL_ACCESS_EN_PF.addr[0], 0xfffff)
        if getenv("AM_PSP_TRACE", 0):
          gfx = self.adev.indirect_rreg_pcie(self.adev.regBIFC_GFX_INT_MONITOR_MASK.addr[0])
          db = self.adev.indirect_rreg_pcie(self.adev.regBIFC_DOORBELL_ACCESS_EN_PF.addr[0])
          print(f"am {self.adev.devfmt}: SOC pcie BIFC gfx_int_monitor={gfx:#x} doorbell_access={db:#x}", flush=True)
      else:
        self.adev.regBIFC_GFX_INT_MONITOR_MASK.write(0x7ff)
        self.adev.regBIFC_DOORBELL_ACCESS_EN_PF.write(0xfffff)
    else: self.adev.regRCC_DEV0_EPF2_STRAP2.update(strap_no_soft_reset_dev0_f2=0x0)
    self.adev.regRCC_DEV0_EPF0_RCC_DOORBELL_APER_EN.write(0x1)
  def set_clockgating_state(self):
    if self.adev.ip_ver[am.HDP_HWIP] >= (5,2,1): self.adev.regHDP_MEM_POWER_CTRL.update(atomic_mem_power_ctrl_en=1, atomic_mem_power_ds_en=1)

  def doorbell_enable(self, port, awid=0, awaddr_31_28_value=0, offset=0, size=0, aid=0):
    reg = self.adev.reg(f"{'regGDC_S2A0_S2A' if self.adev.ip_ver[am.GC_HWIP] >= (12,0,0) else 'regS2A'}_DOORBELL_ENTRY_{port}_CTRL")
    val = reg.encode(**{f"s2a_doorbell_port{port}_enable":1, f"s2a_doorbell_port{port}_awid":awid,  f"s2a_doorbell_port{port}_range_size":size,
      f"s2a_doorbell_port{port}_awaddr_31_28_value":awaddr_31_28_value, f"s2a_doorbell_port{port}_range_offset":offset})

    if self.adev.ip_ver[am.NBIO_HWIP] in {(7,9,0), (7,9,1)}: self.adev.indirect_wreg_pcie(reg.addr[0], val, aid=aid)
    else: reg.write(val)
