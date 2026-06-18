from collections import OrderedDict
from typing import Optional
from llm_memory_calculator.genz.unit import Unit
import warnings
from llm_memory_calculator.genz.system import System
import pandas as pd
from llm_memory_calculator.systems.system_configs import system_configs
from llm_memory_calculator.hardware import get_hardware_config

OFFLOAD_BW = 64   # CPU/host offload over PCIe 5.0 x16 (~63 GB/s unidirectional, PCIe spec). The prior
                  # 128 implied NVLink-class C2C, which over-states host-offload bandwidth ~2×. Devices
                  # with a real C2C/NVLink offload path should set `external_mem_bw` on their config.

unit = Unit()

# C2 (decode realism): default memory-bandwidth efficiency (MBU) by memory TECHNOLOGY, applied when a
# device has no measured `inference_calibration` block. Real sustained streaming bandwidth is a fraction
# of the datasheet peak; this fraction is a documented physical property of the memory technology, not a
# per-device tuning constant — so it is grounded AND generalizable (keyed off each config's `memory_type`).
# Anchors: GB10 LPDDR5X MEASURED on this box at 0.73-0.77 (copy) / 0.70 (decode) — see
# audit_microbench_data.md; HBM datacenter parts sustain ~0.80-0.90 in published vLLM/STREAM decode-MBU
# studies (conservative mid-band used). A per-device measured calibration block always OVERRIDES this.
_ETA_MEM_BY_MEMORY_TYPE = {
    'hbm3e': 0.85, 'hbm3': 0.85, 'hbm2e': 0.80, 'hbm2': 0.80, 'hbm': 0.82,
    'lpddr5x': 0.73, 'lpddr5': 0.73, 'lpddr4x': 0.70, 'lpddr4': 0.70,
    'gddr6x': 0.75, 'gddr6': 0.75, 'gddr5x': 0.72, 'gddr5': 0.72,
    'ddr5': 0.65, 'ddr4': 0.62,  # CPU main memory sustains a lower fraction of peak
}
# Conservative default for an unknown/missing memory_type: 0.80 (typical accelerator HBM/GDDR midpoint).
_DEFAULT_ETA_MEM = 0.80

# x86 SERVER CPU sustained-DRAM-streaming efficiency (STREAM eta). The generic 'ddr5' band (0.65)
# under-states what multi-channel server x86 sustains: published STREAM TRIAD on 8-channel DDR5
# Sapphire Rapids and 12-channel DDR5 Genoa sustains 0.80-0.86 of the theoretical peak DRAM BW.
#   - Dell PowerEdge "Workload-Based DDR5 Memory Guidance" (Sapphire Rapids DDR5-4800, 8ch): STREAM
#     TRIAD lands between 80% and 95% of theoretical peak.
#   - AMD EPYC Genoa (DDR5-4800, 12ch, 460.8 GB/s peak): measured TRIAD ~400 GB/s ≈ 0.86 of peak
#     (Fujitsu PRIMERGY / Dell PowerEdge AMD reports). The x86 write-allocate (RFO) ceiling for the
#     naive write stream is 3/4=0.75, but a read-dominated stream (LLM weight/KV streaming during
#     decode) avoids that penalty and sustains higher.
# We take the conservative LOWER edge of the published measured band (0.80) — grounded, generalizable
# across server x86, and a measured calibration block still overrides it. ARM CPUs (Grace/Graviton)
# keep the LPDDR5/DDR band measured for them; they are NOT promoted to this x86 value.
_ETA_MEM_X86_SERVER_CPU = 0.80
_X86_CPU_VENDORS = ('intel', 'amd')


def _default_eta_mem(memory_type) -> float:
    """Documented MBU for a memory technology (see _ETA_MEM_BY_MEMORY_TYPE). Generalizable, not tuned."""
    return _ETA_MEM_BY_MEMORY_TYPE.get(str(memory_type or '').lower(), _DEFAULT_ETA_MEM)


def _default_eta_mem_cpu(config) -> float:
    """Sustained DRAM-streaming efficiency for a CPU device. x86 server parts (Intel/AMD) sustain the
    published STREAM band (~0.80, see _ETA_MEM_X86_SERVER_CPU); ARM CPUs fall back to their memory-tech
    DDR/LPDDR band. ``config`` may be a dict (static path) or a System (object path)."""
    def _get(key):
        if isinstance(config, dict):
            return config.get(key)
        return getattr(config, key, None)
    vendor = str(_get('manufacturer') or _get('vendor') or '').lower()
    if any(v in vendor for v in _X86_CPU_VENDORS):
        return _ETA_MEM_X86_SERVER_CPU
    # ARM / unknown CPU: use the explicit memory_type band, else the conservative DDR5 band.
    mt = _get('memory_type') or 'ddr5'
    return _default_eta_mem(mt)


# C3 (prefill compute realism): default compute efficiency (achievable large-GEMM MFU vs DENSE peak)
# by tensor-core generation, applied when a device has no measured `inference_calibration` block.
# Replaces the removed ungrounded flat-0.85 "tensor core efficiency" (C4). Anchor: published
# cuBLAS/CUTLASS large dense bf16 GEMM sustains ~0.75-0.85 of the dense tensor-core peak; newer gens
# sustain slightly more. This is the achievable GEMM MFU — additional prefill overhead (kernel-launch
# floor, small-op/attention inefficiency) is modeled by the per-device t_launch term (0 until measured,
# honestly unmodeled rather than faked). A measured calibration block always OVERRIDES this default.
_ETA_COMPUTE_BY_TENSOR_CORE_GEN = {
    'gen5': 0.82, 'gen4': 0.80, 'gen3': 0.78, 'gen2': 0.72, 'gen1': 0.72,
    'cdna3': 0.78, 'cdna2': 0.75, 'cdna': 0.75, 'xmx': 0.75,
}
_DEFAULT_ETA_COMPUTE_GPU = 0.75   # unknown GPU tensor-core gen: conservative published mid-band
_DEFAULT_ETA_COMPUTE_CPU = 0.65   # CPU AMX/AVX-512 GEMM sustains a lower fraction of vector peak


def _default_eta_compute(config: dict) -> float:
    """Achievable large-GEMM MFU for a device, keyed off its tensor-core generation / type.
    Generalizable (per-arch), grounded in published GEMM MFU — not a per-device tuned constant."""
    if str(config.get('type', '')).lower() == 'cpu':
        return _DEFAULT_ETA_COMPUTE_CPU
    gen = str(config.get('tensor_cores') or '').lower()
    if gen in _ETA_COMPUTE_BY_TENSOR_CORE_GEN:
        return _ETA_COMPUTE_BY_TENSOR_CORE_GEN[gen]
    return _DEFAULT_ETA_COMPUTE_GPU

class RuntimeBreakdown():
    def __init__(self):
        # Blockwise
        self.Embedding: float = 0
        self.MHA: float = 0
        self.FFN: float = 0
        self.Collective: float = 0
        # Smaller Layers
        self.LA_layers: float = 0
        self.QKVO_layers: float = 0
        self.FFN_layers: float = 0
        # Others
        self.Softmax: float = 0
        self.AR_time: float = 0
        self.A2A_time: float = 0
        self.Send_Recv_time: float = 0
        self.Mamba_time: float = 0

    def __repr__(self):
        variables = vars(self)
        return ', '.join(f'{name}: {value}' for name, value in variables.items())

    def to_dict(self):
        return vars(self)

class ModdelingOutput(dict):
    """Output container for inference modeling results.

    Stores modeling results as both dict items and instance attributes
    for flexible access patterns.
    """

    def __init__(self,
                 Latency: float = 0,
                 Throughput: float = 0,
                 Throughput_tokens_per_sec: float = 0,
                 Runtime_breakdown: Optional[RuntimeBreakdown] = None,
                 is_offload: Optional[bool] = False,
                 model_df: Optional[pd.DataFrame] = None,
                 summary_table: Optional[pd.DataFrame] = None,
                 tokens_generated: Optional[int] = 0,
                 **kwargs):
        # Initialize dict with all values
        super().__init__(
            Latency=Latency,
            Throughput=Throughput,
            Throughput_tokens_per_sec=Throughput_tokens_per_sec,
            Runtime_breakdown=Runtime_breakdown,
            is_offload=is_offload,
            model_df=model_df,
            summary_table=summary_table,
            tokens_generated=tokens_generated,
            **kwargs
        )
        # Also set as instance attributes for attribute access
        self.Latency = Latency
        self.Throughput = Throughput
        self.Throughput_tokens_per_sec = Throughput_tokens_per_sec
        self.Runtime_breakdown = Runtime_breakdown
        self.is_offload = is_offload
        self.model_df = model_df
        self.summary_table = summary_table
        self.tokens_generated = tokens_generated
        # Handle any additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

def get_offload_system(system, total_memory_req, debug):
    """Create a new system with offloaded memory connections

    Args:
        system (System): System Definition without offload
        total_memory_req (float): Memory required for the model weights and activations in MB
        debug (bool): Debug flag for printing

    Returns:
        System: System Definition with offload
    """
    total_device_memory = unit.raw_to_unit(system.off_chip_mem_size, type='M')/1024 ## GB
    total_memory_req = total_memory_req/1024 ## GB
    memory_offloaded = total_memory_req - total_device_memory

    offload_bw = OFFLOAD_BW if system.external_mem_bw == 0 else system.get_external_mem_bw() 
    if debug:
        print(f'Total Memory Req:{total_memory_req}GB, Total device mem:{total_device_memory}GB, Mem Offload:{memory_offloaded}GB @ {offload_bw} GB/s')

    ###############################################################################################
    ### Memory Time = max(min(size_required,size_hbm)/BW_hbm, (size_required-size_hbm)/BW_offload)
    ###############################################################################################

    if memory_offloaded > 0:
        new_offchip_BW = (total_memory_req ) / max(min(total_memory_req,total_device_memory)/unit.raw_to_unit(system.offchip_mem_bw, type='BW'), memory_offloaded/offload_bw) 
        system.set_offchip_mem_bw(new_offchip_BW)
        if debug:
            print(f'New BW:{new_offchip_BW}')
    return system

def get_inference_system(system_name='A100_40GB_GPU', bits='bf16', ceff=1, meff=1,
                        collective_strategy='GenZ', network_config=None,
                        parallelism_heirarchy = "TP{1}_EP{1}_PP{1}",
                        phase=None,
                         **kwargs):
    ##################################################################################################
    ### System Declaration
    ##################################################################################################
    # Check if it's a CPUSystem object
    from llm_memory_calculator.genz.cpu.cpu_system import CPUSystem
    if isinstance(system_name, CPUSystem):
        # Return the CPUSystem itself, not just the base_system
        # This preserves CPU-specific functionality
        return system_name
    
    if isinstance(system_name, str):
        # Try hardware manager with alias support first
        hw_config = get_hardware_config(system_name)
        if hw_config:
            system_name = hw_config
        elif system_name in system_configs:
            # Fallback to direct lookup for backward compatibility
            system_name = system_configs[system_name]
        else:
            raise ValueError(f'System mentioned:{system_name} not present in predefined systems. Please use systems from systems/system_configs')
    # Optional per-hardware inference calibration (default no-op). Sourced from an
    # `inference_calibration: {decode:{...}, prefill:{...}}` block on the hardware config dict; only
    # applied for the matching `phase`. Absent block / phase → kernel_launch=stream=0 and ceff/meff
    # keep their passed values → byte-identical to today for every uncalibrated hardware.
    kernel_launch_latency_ms = 0.0
    per_stream_overhead_ms = 0.0
    if isinstance(system_name, dict):
        # Initialize variables with defaults from the dict
        NUM_FLOPS = system_name.get('Flops', 320)
        OFFCHIP_MEM_BW = system_name.get('Memory_BW', 40)
        per_chip_memory = system_name.get('Memory_size', 2000)
        C2C_BW = system_name.get('ICN', 150)
        C2C_LL = system_name.get('ICN_LL', 1)
        # L1: warn about FP8 ONLY on hardware whose architecture lacks FP8 tensor support (here the
        # device arch is known, unlike System.__init__). FP8 is supported on Hopper/Ada/Blackwell and
        # AMD CDNA3 (MI300); warning on those was noise.
        if bits in ('fp8', 'fp8_e4m3', 'fp8_e5m2', 'mixed_fp8', 'mixed_fp8_bf16'):
            _arch = str(system_name.get('architecture', '')).lower()
            _tc = str(system_name.get('tensor_cores', '')).lower()
            _name = str(system_name.get('name', '')).lower()
            _fp8_capable = (_tc in ('gen4', 'gen5')
                            or any(a in _arch for a in ('hopper', 'ada', 'blackwell', 'cdna3'))
                            or 'mi300' in _name or 'mi325' in _name)
            if not _fp8_capable:
                warnings.warn(
                    f"FP8 selected on {system_name.get('name', 'this device')} (arch={_arch or 'unknown'}): "
                    f"FP8 tensor cores require Hopper/Ada/Blackwell or AMD CDNA3 (MI300+); estimates may "
                    f"not reflect real throughput on this device.", stacklevel=2)

        cal = system_name.get('inference_calibration')
        calibrated_mem = calibrated_compute = False
        if cal and phase and phase in cal:
            pc = cal[phase]
            if 'eta_compute' in pc:
                ceff = pc['eta_compute']
                calibrated_compute = True
            if 'eta_mem' in pc:
                meff = pc['eta_mem']
                calibrated_mem = True
            kernel_launch_latency_ms = pc.get('t_launch_ms', 0.0)
            per_stream_overhead_ms = pc.get('c_stream_ms_per_layer', 0.0)
        # C2/C3: when the caller didn't pass an explicit efficiency (==1) and no measured calibration
        # block set it, fall back to the grounded per-technology bands — memory-tech MBU (decode realism,
        # ~0.8 HBM / ~0.73 LPDDR5X) and tensor-core-gen large-GEMM MFU (prefill realism, ~0.8) — instead
        # of an impossible 100%. A measured calibration block overrides these.
        if not calibrated_mem and meff == 1:
            _mt = system_name.get('memory_type')
            # R2-G2 / x86-STREAM: CPU configs omit memory_type. Server x86 (Intel/AMD) sustains the
            # published STREAM band (~0.80); ARM CPUs use their DDR/LPDDR memory-tech band. Both are
            # still lower than accelerator HBM and below the generic 0.80 default for ARM-DDR.
            if str(system_name.get('type', '')).lower() == 'cpu' and not _mt:
                meff = _default_eta_mem_cpu(system_name)
            else:
                meff = _default_eta_mem(_mt)
        if not calibrated_compute and ceff == 1:
            ceff = _default_eta_compute(system_name)
    elif isinstance(system_name, System):
        system_name.bits = bits
        # R2-G1: a System object built directly (e.g. BudEvolve HardwareExplorer) previously bypassed the
        # C2/C3 efficiency bands and ran at 100% MFU/MBU (~25-28% optimistic vs the dict path). Apply the
        # same bands here when the caller passed the default efficiency (==1): use the System's own
        # memory_type/tensor_cores if it carries them, else the generic GPU bands. A System whose
        # efficiency was already set (≠1) is respected.
        if ceff == 1:
            ceff = _default_eta_compute({
                'type': getattr(system_name, 'type', 'gpu'),
                'tensor_cores': getattr(system_name, 'tensor_cores', None),
            })
        if meff == 1:
            _mt = getattr(system_name, 'memory_type', None)
            if str(getattr(system_name, 'type', '')).lower() == 'cpu' and not _mt:
                meff = _default_eta_mem_cpu(system_name)
            else:
                meff = _default_eta_mem(_mt)
        system_name.compute_efficiency = ceff
        system_name.memory_efficiency = meff
        system_name.collective_strategy = collective_strategy
        system_name.parallelism_heirarchy = parallelism_heirarchy
        system_name.network_config = network_config
        return system_name
    else:
        raise TypeError(f'System should be weight str or dict with Flops,Memory, ICN values: System_name: {system_name}')

    return System(unit,frequency=1000 , flops=NUM_FLOPS, off_chip_mem_size=(per_chip_memory*1024), compute_efficiency=ceff, memory_efficiency=meff,
                    offchip_mem_bw=OFFCHIP_MEM_BW, bits=bits, external_mem_bw=OFFLOAD_BW, interchip_link_bw=C2C_BW, interchip_link_latency=C2C_LL,
                    collective_strategy=collective_strategy, network_config=network_config, parallelism_heirarchy = parallelism_heirarchy,
                    kernel_launch_latency_ms=kernel_launch_latency_ms, per_stream_overhead_ms=per_stream_overhead_ms)