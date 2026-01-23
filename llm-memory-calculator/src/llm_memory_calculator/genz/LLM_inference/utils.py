from collections import OrderedDict
from typing import Optional
from llm_memory_calculator.genz.unit import Unit
import warnings
from llm_memory_calculator.genz.system import System
import pandas as pd
from llm_memory_calculator.systems.system_configs import system_configs
from llm_memory_calculator.hardware import get_hardware_config

OFFLOAD_BW = 128

unit = Unit()

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
    if isinstance(system_name, dict):
        # Initialize variables with defaults from the dict
        NUM_FLOPS = system_name.get('Flops', 320)
        OFFCHIP_MEM_BW = system_name.get('Memory_BW', 40)
        per_chip_memory = system_name.get('Memory_size', 2000)
        C2C_BW = system_name.get('ICN', 150)
        C2C_LL = system_name.get('ICN_LL', 1)
    elif isinstance(system_name, System):
        system_name.bits = bits
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
                    collective_strategy=collective_strategy, network_config=network_config, parallelism_heirarchy = parallelism_heirarchy)