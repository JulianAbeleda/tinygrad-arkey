"""Single source of truth for the AMD PSP C2PMSG mailbox register map (BAR5 dword offsets).

These are PSP (Platform Security Processor) hardware register offsets used by the eGPU repro/gate
(`extra/remote/amd_repro.py`) and the MMHUB/GART snapshot (`linux_mmhub_gart_snapshot.py`). One table
here so the offsets are not maintained in two dirs. Dependency-free (pure dict) -> safe to import
anywhere, including before the tinygrad env-ordering barrier.
"""
from __future__ import annotations

PSP_C2PMSG_REGS = {
  "C2PMSG33_VMBX": 0x16061, "C2PMSG35_BL": 0x16063, "C2PMSG36_ADDR": 0x16064,
  "C2PMSG58_SOS_FW_VERSION": 0x1607a,
  "C2PMSG64_RING": 0x16080, "C2PMSG67_WPTR": 0x16083, "C2PMSG69_RING_LO": 0x16085,
  "C2PMSG70_RING_HI": 0x16086, "C2PMSG71_RING_SIZE": 0x16087, "C2PMSG73_SPI_DOORBELL": 0x16089,
  "C2PMSG81_SOS": 0x16091, "C2PMSG90_SMU": 0x1609a, "C2PMSG92_STATUS": 0x1609c,
  "C2PMSG101_GPCOM_CMD": 0x160a5, "C2PMSG102_GPCOM_LO": 0x160a6, "C2PMSG103_GPCOM_HI": 0x160a7,
  "C2PMSG115_SPI": 0x160b3, "C2PMSG116_SPI_ARG": 0x160b4, "C2PMSG127_RAS_CAP": 0x160bf,
}

PSP_DENSE_C2PMSG_REGS = {
  **{f"MP0_C2PMSG{i:03d}": 0x16040 + i for i in range(128)},
  **{f"MP1_C2PMSG{i:03d}": 0x16240 + i for i in range(128)},
}

# The subset the eGPU clean-gate (amd_repro) inspects, in its historical order.
PSP_GATE_REG_NAMES = ("C2PMSG33_VMBX", "C2PMSG35_BL", "C2PMSG36_ADDR", "C2PMSG64_RING", "C2PMSG67_WPTR",
                      "C2PMSG69_RING_LO", "C2PMSG70_RING_HI", "C2PMSG71_RING_SIZE", "C2PMSG81_SOS",
                      "C2PMSG90_SMU", "C2PMSG92_STATUS", "C2PMSG115_SPI")
PSP_GATE_REGS = {k: PSP_C2PMSG_REGS[k] for k in PSP_GATE_REG_NAMES}
