"""
Static hardware configurations for LLM Memory Calculator.

This module contains predefined hardware configurations including GPUs, CPUs, TPUs, and ASICs.
Configurations are merged from multiple sources to provide a comprehensive hardware library.
"""

from typing import Any, Dict, Optional
from llm_memory_calculator.hardware.cpu_specs import CPU_CONFIGS

HARDWARE_CONFIGS: Dict[str, Dict[str, Any]] = {
    # NVIDIA GPUs
    'A100_40GB_GPU': {
        'name': 'A100_40GB_GPU',
        'Flops': 312,
        'Memory_size': 40,
        'Memory_BW': 1600,
        'ICN': 150,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'AMPERE',
        'generation': 'Data Center GPU',
        'compute_capability': '8.0',
        'release_year': 2020,
        'tensor_cores': 'gen3',
        'rt_cores': None,
        'memory_type': 'HBM2e',
        'interconnect': 'nvlink',
        'interconnect_bandwidth_gbps': 600,
        'pci_ids': ['20b0', '20f1'],  # PCIe and SXM4 variants
        'aliases': ['A100', 'TESLA A100', 'A100-SXM4-40GB', 'A100-PCIE-40GB', 'NVIDIA A100 40GB'],
        'cost': {
            'aws_on_demand': 2.21,
            'aws_spot': 0.80,
            'gcp_on_demand': 2.06,
            'gcp_preemptible': 0.72,
            'azure_on_demand': 2.42,
            'azure_spot': 0.73,
            'lambda_labs': 1.10,
            'coreweave': 1.28,
            'runpod': 0.79,
            'vast_ai': 0.80,
            'purchase_price_usd': 10000,
            'tdp_watts': 400,
        },
    },
    'A100_80GB_GPU': {
        'name': 'A100_80GB_GPU',
        'Flops': 312,
        'Memory_size': 80,
        'Memory_BW': 2039,
        'ICN': 150,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'AMPERE',
        'generation': 'Data Center GPU',
        'compute_capability': '8.0',
        'release_year': 2020,
        'tensor_cores': 'gen3',
        'rt_cores': None,
        'memory_type': 'HBM2e',
        'interconnect': 'nvlink',
        'interconnect_bandwidth_gbps': 600,
        'pci_ids': ['20b2', '20b5'],  # SXM4 and PCIe variants
        'aliases': ['A100 80GB', 'TESLA A100 80GB', 'A100-SXM4-80GB', 'A100-PCIE-80GB', 'NVIDIA A100 80GB'],
        'cost': {
            'aws_on_demand': 3.67,
            'aws_spot': 1.40,
            'gcp_on_demand': 3.67,
            'gcp_preemptible': 1.10,
            'azure_on_demand': 3.67,
            'azure_spot': 1.28,
            'lambda_labs': 1.29,
            'coreweave': 1.85,
            'runpod': 1.19,
            'vast_ai': 1.10,
            'purchase_price_usd': 15000,
            'tdp_watts': 400,
        },
    },
    'V100_16GB_GPU': {
        'name': 'V100_16GB_GPU',
        'Flops': 125,  # Tensor Core FP16 TFLOPS (15.7 FP32 TFLOPS)
        'Memory_size': 16,
        'Memory_BW': 900,
        'ICN': 300,  # NVLink 2.0
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'VOLTA',
        'generation': 'Data Center GPU',
        'compute_capability': '7.0',
        'release_year': 2017,
        'tensor_cores': 'gen1',  # First generation tensor cores
        'rt_cores': None,
        'memory_type': 'HBM2',
        'interconnect': 'nvlink',
        'interconnect_bandwidth_gbps': 300,
        'pci_ids': ['1db4', '1db6'],  # PCIe and SXM2 16GB variants
        'aliases': ['V100 16GB', 'TESLA V100 16GB', 'V100-SXM2-16GB', 'V100-PCIE-16GB', 'NVIDIA V100 16GB'],
        'cost': {
            'aws_on_demand': 0.90,
            'aws_spot': 0.27,
            'gcp_on_demand': 0.74,
            'gcp_preemptible': 0.22,
            'lambda_labs': 0.50,
            'runpod': 0.39,
            'vast_ai': 0.25,
            'purchase_price_usd': 8000,  # New ~$8k-$11k, used ~$2k-$4k
            'tdp_watts': 250,
        },
    },
    'V100_32GB_GPU': {
        'name': 'V100_32GB_GPU',
        'Flops': 125,  # Tensor Core FP16 TFLOPS (15.7 FP32 TFLOPS)
        'Memory_size': 32,
        'Memory_BW': 900,
        'ICN': 300,  # NVLink 2.0
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'VOLTA',
        'generation': 'Data Center GPU',
        'compute_capability': '7.0',
        'release_year': 2017,
        'tensor_cores': 'gen1',  # First generation tensor cores
        'rt_cores': None,
        'memory_type': 'HBM2',
        'interconnect': 'nvlink',
        'interconnect_bandwidth_gbps': 300,
        'pci_ids': ['1db5', '1dbe', '1df6'],  # PCIe and SXM2 32GB variants
        'aliases': ['V100 32GB', 'TESLA V100 32GB', 'V100-SXM2-32GB', 'V100-PCIE-32GB', 'NVIDIA V100 32GB', 'V100S'],
        'cost': {
            'aws_on_demand': 1.21,
            'aws_spot': 0.36,
            'gcp_on_demand': 0.99,
            'gcp_preemptible': 0.30,
            'lambda_labs': 0.60,
            'runpod': 0.49,
            'vast_ai': 0.35,
            'purchase_price_usd': 10000,  # New ~$8k-$11k, used ~$3k-$5k
            'tdp_watts': 250,
        },
    },
    'H100_GPU': {
        'name': 'H100_GPU',
        'Flops': 989.5,  # dense bf16 tensor TFLOPS (NVIDIA H100 SXM datasheet). 1979 = 2:4 sparse, not achievable by dense LLM GEMMs.
        'Memory_size': 80,
        'Memory_BW': 3350,
        'ICN': 450,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'HOPPER',
        'generation': 'Data Center GPU',
        'compute_capability': '9.0',
        'release_year': 2022,
        'tensor_cores': 'gen4',
        'rt_cores': None,
        'memory_type': 'HBM3',
        'interconnect': 'nvlink',
        'interconnect_bandwidth_gbps': 900,
        'pci_ids': ['2330', '2331', '2339'],  # SXM5, PCIe, NVL variants
        'aliases': ['H100', 'HOPPER', 'H100-SXM5', 'H100-PCIE', 'H100-NVL', 'NVIDIA H100'],
        'cost': {
            'aws_on_demand': 4.76,
            'aws_spot': 2.00,
            'gcp_on_demand': 4.76,
            'gcp_preemptible': 1.90,
            'azure_on_demand': 5.12,
            'azure_spot': 2.05,
            'lambda_labs': 2.49,
            'coreweave': 2.85,
            'runpod': 2.39,
            'vast_ai': 2.00,
            'purchase_price_usd': 30000,
            'tdp_watts': 700,
        },
    },
    'H100_PCIe_GPU': {
        'name': 'H100_PCIe_GPU',
        'Flops': 756,  # dense bf16 tensor TFLOPS (NVIDIA H100 PCIe datasheet). 1513 = 2:4 sparse.
        'Memory_size': 80,
        'Memory_BW': 2000,
        'ICN': 300,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'HOPPER',
        'generation': 'Data Center GPU',
        'compute_capability': '9.0',
        'release_year': 2022,
        'tensor_cores': 'gen4',
        'rt_cores': None,
        'memory_type': 'HBM3',
        'interconnect': 'pcie5',
        'interconnect_bandwidth_gbps': 128,
        'pci_ids': ['2331'],
        'aliases': ['H100 PCIe', 'H100-PCIE', 'H100-PCIe-80GB', 'NVIDIA H100 PCIe'],
        'cost': {
            'aws_on_demand': 3.50,
            'purchase_price_usd': 25000,
            'tdp_watts': 350,
        },
    },
    'H200_GPU': {
        'name': 'H200_GPU',
        'Flops': 989.5,  # dense bf16 tensor TFLOPS (H200 = H100 compute die, same dense peak). 1979 = 2:4 sparse.
        'Memory_size': 141,
        'Memory_BW': 4800,
        'ICN': 450,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'HOPPER',
        'generation': 'Data Center GPU',
        'compute_capability': '9.0',
        'release_year': 2024,
        'tensor_cores': 'gen4',
        'rt_cores': None,
        'memory_type': 'HBM3e',
        'interconnect': 'nvlink',
        'interconnect_bandwidth_gbps': 900,
        'pci_ids': ['2335'],
        'aliases': ['H200', 'H200-141GB', 'H200-SXM', 'NVIDIA H200'],
        'cost': {
            'aws_on_demand': 5.50,
            'purchase_price_usd': 35000,
            'tdp_watts': 700,
        },
    },
    'GH200_GPU': {
        'name': 'GH200_GPU',
        'Flops': 989.5,  # dense bf16 tensor TFLOPS (GH200 = H100 GPU die). 1979 = 2:4 sparse.
        'Memory_size': 144,
        'Memory_BW': 4900,
        'ICN': 450,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'HOPPER',
        'generation': 'Grace Hopper Superchip',
        'compute_capability': '9.0',
        'release_year': 2023,
        'tensor_cores': 'gen4',
        'rt_cores': None,
        'memory_type': 'HBM3',
        'interconnect': 'nvlink',
        'interconnect_bandwidth_gbps': 900,
        'pci_ids': ['233a'],  # GH200
        'aliases': ['GH200', 'GRACE HOPPER', 'GH200-144GB', 'NVIDIA GH200'],
        'cost': {
            'aws_on_demand': 5.50,
            'aws_spot': 2.50,
            'gcp_on_demand': 5.50,
            'gcp_preemptible': 2.20,
            'lambda_labs': 3.00,
            'coreweave': 3.50,
            'purchase_price_usd': 45000,  # Superchip (Grace CPU + H100 GPU), systems start ~$41.5k
            'tdp_watts': 900,
        },
    },
    'B100': {
        'name': 'B100',
        'Flops': 1750,  # BF16 TFLOPS (3500 FP8). B100 GPU: ~1.75 PFLOPS BF16
        'Memory_size': 192,
        'Memory_BW': 8000,
        'ICN': 900,
        'ICN_LL': 0.25,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'BLACKWELL',
        'generation': 'Data Center GPU',
        'compute_capability': '10.0',
        'release_year': 2024,
        'tensor_cores': 'gen5',
        'rt_cores': None,
        'memory_type': 'HBM3e',
        'cost': {
            'lambda_labs': 5.00,
            'coreweave': 5.50,
            'purchase_price_usd': 32500,  # Estimated $30k-$35k (HSBC estimate)
            'tdp_watts': 700,
        },
    },
    'GB200': {
        'name': 'GB200',
        'Flops': 2250,  # BF16 TFLOPS (4500 FP8). B200 GPU: ~2.25 PFLOPS BF16
        'Memory_size': 192,
        'Memory_BW': 8000,
        'ICN': 900,
        'ICN_LL': 0.25,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'BLACKWELL',
        'generation': 'Grace Blackwell Superchip',
        'compute_capability': '10.0',
        'release_year': 2024,
        'tensor_cores': 'gen5',
        'rt_cores': None,
        'memory_type': 'HBM3e',
        'cost': {
            'lambda_labs': 7.00,
            'coreweave': 7.50,
            'purchase_price_usd': 70000,
            'tdp_watts': 1000,
        },
    },
    'GB10': {
        'name': 'GB10',
        # Blackwell GPU in the GB10 Grace-Blackwell Superchip (DGX Spark / Project DIGITS).
        # ~1 PFLOP FP4 (sparse) → ~500 FP8 → ~250 BF16 TFLOPS dense-tensor.
        'Flops': 250,  # BF16 TFLOPS (~500 FP8, ~1000 FP4). Single Blackwell GPU.
        'Memory_size': 128,  # GB, LPDDR5X UNIFIED (coherent CPU+GPU); ~119 GiB usable to the OS.
        'Memory_BW': 273,  # GB/s — LPDDR5X unified bandwidth (the dominant decode bottleneck).
        'ICN': 100,  # NVLink-C2C CPU<->GPU / ConnectX clustering; unused at TP=1 (single GPU).
        'ICN_LL': 0.25,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'BLACKWELL',
        'generation': 'Grace Blackwell Superchip',
        'compute_capability': '12.1',  # sm_121
        'release_year': 2025,
        'tensor_cores': 'gen5',
        'rt_cores': 'gen4',
        'memory_type': 'LPDDR5X',
        'unified_memory': True,
        'aliases': ['GB10', 'DGX Spark', 'GB10 Grace Blackwell', 'NVIDIA GB10', 'Project DIGITS'],
        'cost': {
            'purchase_price_usd': 3999,
            'tdp_watts': 170,
        },
        # MEASURED inference calibration (real llama.cpp on GB10, qwen2.5 0.5B/3B q4, 14-point grid
        # of batch B∈{1,2,4,8} × context T∈{0.5k,4k,12k}). Read internally by get_inference_system →
        # the roofline is scaled by these efficiencies and the additive per-step/per-stream overheads
        # the analytic model omits. Decode RMS ~6.6%, prefill ~10%; refit via the inference
        # calibration harness with more model sizes. ONLY GB10 carries this block → every other
        # device is byte-identical (verified). eta_mem = effective decode BW (~66% of the 273 spec,
        # shared-LPDDR5X penalty); eta_compute = small-prefill MFU (~8%); t_launch = per-kernel launch
        # latency (ms); c_stream = per-layer per-extra-stream runtime overhead (sampling/scheduling).
        'inference_calibration': {
            # decode (memory-bound): eta_mem≈66% of peak BW (shared LPDDR5X) + ~4.6us/kernel launch
            # + ~0.016ms/layer per extra concurrent stream (sampling/scheduling). RMS ~6.6%.
            'decode':  {'eta_mem': 0.659, 't_launch_ms': 0.00456, 'c_stream_ms_per_layer': 0.0158},
            # prefill (compute-bound, small-model launch-heavy): eta_compute≈10% MFU on small GEMMs,
            # eta_mem≈0.74, ~0.21ms/GEMM-op launch floor. RMS ~5%. (No per-stream term: single pass.)
            'prefill': {'eta_compute': 0.096, 'eta_mem': 0.741, 't_launch_ms': 0.206},
        },
    },
    'B200_GPU': {
        'name': 'B200_GPU',
        'Flops': 2250,
        'Memory_size': 192,
        'Memory_BW': 8000,
        'ICN': 900,
        'ICN_LL': 0.25,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'BLACKWELL',
        'generation': 'Data Center GPU',
        'compute_capability': '10.0',
        'release_year': 2024,
        'tensor_cores': 'gen5',
        'rt_cores': None,
        'memory_type': 'HBM3e',
        'aliases': ['B200', 'B200-192GB', 'NVIDIA B200'],
        'cost': {
            'purchase_price_usd': 40000,
            'tdp_watts': 1000,
        },
    },
    'B300': {
        'name': 'B300',
        'Flops': 3750,  # BF16 TFLOPS (~15 PFLOPS dense FP4)
        'Memory_size': 288,
        'Memory_BW': 8000,
        'ICN': 900,
        'ICN_LL': 0.25,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'BLACKWELL',
        'generation': 'Data Center GPU',
        'compute_capability': '10.0',
        'release_year': 2025,
        'tensor_cores': 'gen5',
        'rt_cores': None,
        'memory_type': 'HBM3e',
        'aliases': ['B300', 'B300-288GB', 'NVIDIA B300', 'GB300'],
        'cost': {
            'purchase_price_usd': 55000,
            'tdp_watts': 1400,
        },
    },
    'L40S_48GB_GPU': {
        'name': 'L40S_48GB_GPU',
        'Flops': 362,
        'Memory_size': 48,
        'Memory_BW': 864,
        'ICN': 300,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'ADA_LOVELACE',
        'generation': 'Professional Data Center GPU',
        'compute_capability': '8.9',
        'release_year': 2024,
        'tensor_cores': 'gen4',
        'rt_cores': 'gen3',
        'memory_type': 'GDDR6',
        'interconnect': 'pcie4',
        'interconnect_bandwidth_gbps': 64,
        'pci_ids': ['26ba'],
        'aliases': ['L40S', 'L40S-48GB', 'NVIDIA L40S', 'NVIDIA-L40S', 'L40S 48GB'],
        'cost': {
            'aws_on_demand': 1.98,
            'gcp_on_demand': 1.70,
            'lambda_labs': 0.99,
            'coreweave': 1.14,
            'runpod': 0.99,
            'purchase_price_usd': 8000,
            'tdp_watts': 350,
        },
    },

    # NVIDIA Consumer GPUs (Ada Lovelace)
    'RTX4090_GPU': {
        'name': 'RTX4090_GPU',
        'Flops': 330,  # FP16 Tensor TFLOPS (82.6 FP32 TFLOPS)
        'Memory_size': 24,
        'Memory_BW': 1008,  # 21 Gbps × 384-bit
        'ICN': 64,  # PCIe 4.0 x16 (no NVLink)
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'ADA_LOVELACE',
        'generation': 'Consumer GPU',
        'compute_capability': '8.9',
        'release_year': 2022,
        'tensor_cores': 'gen4',
        'rt_cores': 'gen3',
        'memory_type': 'GDDR6X',
        'interconnect': 'pcie4',
        'interconnect_bandwidth_gbps': 64,
        'pci_ids': ['2684', '2717'],  # Standard and variants
        'tdp_watts': 450,
        'aliases': ['RTX4090', 'RTX 4090', 'rtx_4090', 'NVIDIA RTX 4090', 'GeForce RTX 4090']
    },
    'RTX4080_GPU': {
        'name': 'RTX4080_GPU',
        'Flops': 242,  # FP16 Tensor TFLOPS
        'Memory_size': 16,
        'Memory_BW': 717,  # 22.4 Gbps × 256-bit
        'ICN': 64,  # PCIe 4.0 x16
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'ADA_LOVELACE',
        'generation': 'Consumer GPU',
        'compute_capability': '8.9',
        'release_year': 2022,
        'tensor_cores': 'gen4',
        'rt_cores': 'gen3',
        'memory_type': 'GDDR6X',
        'interconnect': 'pcie4',
        'interconnect_bandwidth_gbps': 64,
        'pci_ids': ['2704'],
        'tdp_watts': 320,
        'aliases': ['RTX4080', 'RTX 4080', 'rtx_4080', 'NVIDIA RTX 4080', 'GeForce RTX 4080']
    },
    'RTX3080_GPU': {
        'name': 'RTX3080_GPU',
        'Flops': 119,  # FP16 Tensor TFLOPS (29.8 FP32 TFLOPS)
        'Memory_size': 10,
        'Memory_BW': 760,  # 19 Gbps × 320-bit
        'ICN': 64,  # PCIe 4.0 x16 (no NVLink)
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'AMPERE',
        'generation': 'Consumer GPU',
        'compute_capability': '8.6',
        'release_year': 2020,
        'tensor_cores': 'gen3',
        'rt_cores': 'gen2',
        'memory_type': 'GDDR6X',
        'interconnect': 'pcie4',
        'interconnect_bandwidth_gbps': 64,
        'pci_ids': ['2206'],
        'tdp_watts': 320,
        'aliases': ['RTX3080', 'RTX 3080', 'rtx_3080', 'NVIDIA RTX 3080', 'GeForce RTX 3080', 'RTX3080 10GB']
    },
    'RTX3080Ti_GPU': {
        'name': 'RTX3080Ti_GPU',
        'Flops': 136,  # FP16 Tensor TFLOPS (34.1 FP32 TFLOPS)
        'Memory_size': 12,
        'Memory_BW': 912,  # 19 Gbps × 384-bit
        'ICN': 64,  # PCIe 4.0 x16
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'AMPERE',
        'generation': 'Consumer GPU',
        'compute_capability': '8.6',
        'release_year': 2021,
        'tensor_cores': 'gen3',
        'rt_cores': 'gen2',
        'memory_type': 'GDDR6X',
        'interconnect': 'pcie4',
        'interconnect_bandwidth_gbps': 64,
        'pci_ids': ['2208'],
        'tdp_watts': 350,
        'aliases': ['RTX3080Ti', 'RTX 3080 Ti', 'rtx_3080_ti', 'NVIDIA RTX 3080 Ti', 'GeForce RTX 3080 Ti']
    },
    'RTX3070_GPU': {
        'name': 'RTX3070_GPU',
        'Flops': 81,  # FP16 Tensor TFLOPS (20.3 FP32 TFLOPS)
        'Memory_size': 8,
        'Memory_BW': 448,  # 14 Gbps × 256-bit
        'ICN': 64,  # PCIe 4.0 x16
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'AMPERE',
        'generation': 'Consumer GPU',
        'compute_capability': '8.6',
        'release_year': 2020,
        'tensor_cores': 'gen3',
        'rt_cores': 'gen2',
        'memory_type': 'GDDR6',
        'interconnect': 'pcie4',
        'interconnect_bandwidth_gbps': 64,
        'pci_ids': ['2484'],
        'tdp_watts': 220,
        'aliases': ['RTX3070', 'RTX 3070', 'rtx_3070', 'NVIDIA RTX 3070', 'GeForce RTX 3070']
    },
    'RTX3060_GPU': {
        'name': 'RTX3060_GPU',
        'Flops': 51,  # FP16 Tensor TFLOPS (12.7 FP32 TFLOPS)
        'Memory_size': 12,
        'Memory_BW': 360,  # 15 Gbps × 192-bit
        'ICN': 32,  # PCIe 4.0 x16
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'AMPERE',
        'generation': 'Consumer GPU',
        'compute_capability': '8.6',
        'release_year': 2021,
        'tensor_cores': 'gen3',
        'rt_cores': 'gen2',
        'memory_type': 'GDDR6',
        'interconnect': 'pcie4',
        'interconnect_bandwidth_gbps': 64,
        'pci_ids': ['2504'],
        'tdp_watts': 170,
        'aliases': ['RTX3060', 'RTX 3060', 'rtx_3060', 'NVIDIA RTX 3060', 'GeForce RTX 3060', 'RTX3060 12GB']
    },
    'A10_GPU': {
        'name': 'A10_GPU',
        'Flops': 125,  # FP16 Tensor TFLOPS
        'Memory_size': 24,
        'Memory_BW': 600,
        'ICN': 64,
        'tdp_watts': 150,  # NVIDIA A10 datasheet: 150W max power
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'AMPERE',
        'generation': 'Data Center GPU',
        'compute_capability': '8.6',
        'release_year': 2021,
        'tensor_cores': 'gen3',
        'rt_cores': None,
        'memory_type': 'GDDR6',
        'interconnect': 'pcie4',
        'interconnect_bandwidth_gbps': 64,
        'pci_ids': ['2236'],
        'aliases': ['A10', 'NVIDIA A10', 'A10-24GB'],
    },
    'A30_GPU': {
        'name': 'A30_GPU',
        'Flops': 165,  # FP16 Tensor TFLOPS
        'Memory_size': 24,
        'Memory_BW': 933,
        'ICN': 64,
        'tdp_watts': 165,  # NVIDIA A30 datasheet: 165W max power
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'AMPERE',
        'generation': 'Data Center GPU',
        'compute_capability': '8.0',
        'release_year': 2021,
        'tensor_cores': 'gen3',
        'rt_cores': None,
        'memory_type': 'HBM2e',
        'interconnect': 'pcie4',
        'interconnect_bandwidth_gbps': 64,
        'pci_ids': ['20b7'],
        'aliases': ['A30', 'NVIDIA A30', 'A30-24GB'],
    },
    'A40_GPU': {
        'name': 'A40_GPU',
        'Flops': 150,  # FP16 Tensor TFLOPS
        'Memory_size': 48,
        'Memory_BW': 696,
        'ICN': 64,
        'tdp_watts': 300,  # NVIDIA A40 datasheet: 300W max power
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'AMPERE',
        'generation': 'Professional GPU',
        'compute_capability': '8.6',
        'release_year': 2020,
        'tensor_cores': 'gen3',
        'rt_cores': 'gen2',
        'memory_type': 'GDDR6',
        'interconnect': 'pcie4',
        'interconnect_bandwidth_gbps': 64,
        'pci_ids': ['2235'],
        'aliases': ['A40', 'NVIDIA A40', 'A40-48GB'],
    },
    'A6000_GPU': {
        'name': 'A6000_GPU',
        'Flops': 155,  # FP16 Tensor TFLOPS (38.7 FP32)
        'Memory_size': 48,
        'Memory_BW': 768,
        'ICN': 64,
        'tdp_watts': 300,  # NVIDIA RTX A6000 datasheet: 300W max power
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'AMPERE',
        'generation': 'Professional GPU',
        'compute_capability': '8.6',
        'release_year': 2020,
        'tensor_cores': 'gen3',
        'rt_cores': 'gen2',
        'memory_type': 'GDDR6',
        'interconnect': 'pcie4',
        'interconnect_bandwidth_gbps': 64,
        'pci_ids': ['2230'],
        'aliases': ['A6000', 'RTX A6000', 'NVIDIA RTX A6000'],
    },
    'RTX4060_GPU': {
        'name': 'RTX4060_GPU',
        'Flops': 121,  # FP16 Tensor TFLOPS
        'Memory_size': 8,
        'Memory_BW': 272,  # 17 Gbps × 128-bit
        'ICN': 64,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'ADA_LOVELACE',
        'generation': 'Consumer GPU',
        'compute_capability': '8.9',
        'release_year': 2023,
        'tensor_cores': 'gen4',
        'rt_cores': 'gen3',
        'memory_type': 'GDDR6',
        'interconnect': 'pcie4',
        'interconnect_bandwidth_gbps': 64,
        'pci_ids': ['2803'],
        'tdp_watts': 115,
        'aliases': ['RTX4060', 'RTX 4060', 'rtx_4060', 'NVIDIA RTX 4060', 'GeForce RTX 4060'],
    },
    'RTX4060Ti_GPU': {
        'name': 'RTX4060Ti_GPU',
        'Flops': 177,  # FP16 Tensor TFLOPS
        'Memory_size': 16,
        'Memory_BW': 288,  # 18 Gbps × 128-bit
        'ICN': 64,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'ADA_LOVELACE',
        'generation': 'Consumer GPU',
        'compute_capability': '8.9',
        'release_year': 2023,
        'tensor_cores': 'gen4',
        'rt_cores': 'gen3',
        'memory_type': 'GDDR6',
        'interconnect': 'pcie4',
        'interconnect_bandwidth_gbps': 64,
        'pci_ids': ['2805'],
        'tdp_watts': 165,
        'aliases': ['RTX4060Ti', 'RTX 4060 Ti', 'rtx_4060_ti', 'NVIDIA RTX 4060 Ti', 'GeForce RTX 4060 Ti', 'RTX4060Ti 16GB'],
    },
    'RTX4070Ti_GPU': {
        'name': 'RTX4070Ti_GPU',
        'Flops': 186,  # FP16 Tensor TFLOPS
        'Memory_size': 12,
        'Memory_BW': 504,  # 21 Gbps × 192-bit
        'ICN': 64,  # PCIe 4.0 x16
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'ADA_LOVELACE',
        'generation': 'Consumer GPU',
        'compute_capability': '8.9',
        'release_year': 2023,
        'tensor_cores': 'gen4',
        'rt_cores': 'gen3',
        'memory_type': 'GDDR6X',
        'interconnect': 'pcie4',
        'interconnect_bandwidth_gbps': 64,
        'pci_ids': ['2782'],
        'tdp_watts': 285,
        'aliases': ['RTX4070Ti', 'RTX 4070 Ti', 'rtx_4070_ti', 'NVIDIA RTX 4070 Ti', 'GeForce RTX 4070 Ti']
    },
    'RTX4070_GPU': {
        'name': 'RTX4070_GPU',
        'Flops': 147,  # FP16 Tensor TFLOPS
        'Memory_size': 12,
        'Memory_BW': 504,  # 21 Gbps × 192-bit
        'ICN': 64,  # PCIe 4.0 x16
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'ADA_LOVELACE',
        'generation': 'Consumer GPU',
        'compute_capability': '8.9',
        'release_year': 2023,
        'tensor_cores': 'gen4',
        'rt_cores': 'gen3',
        'memory_type': 'GDDR6X',
        'interconnect': 'pcie4',
        'interconnect_bandwidth_gbps': 64,
        'pci_ids': ['2786'],
        'tdp_watts': 200,
        'aliases': ['RTX4070', 'RTX 4070', 'rtx_4070', 'NVIDIA RTX 4070', 'GeForce RTX 4070']
    },
    'RTX3090_GPU': {
        'name': 'RTX3090_GPU',
        'Flops': 142,  # FP16 Tensor TFLOPS (35.6 FP32)
        'Memory_size': 24,
        'Memory_BW': 936,  # 19.5 Gbps × 384-bit
        'ICN': 64,  # PCIe 4.0 x16 (NVLink available on some models)
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'NVIDIA',
        'architecture': 'AMPERE',
        'generation': 'Consumer GPU',
        'compute_capability': '8.6',
        'release_year': 2020,
        'tensor_cores': 'gen3',
        'rt_cores': 'gen2',
        'memory_type': 'GDDR6X',
        'interconnect': 'pcie4',
        'interconnect_bandwidth_gbps': 64,
        'pci_ids': ['2204'],
        'tdp_watts': 350,
        'aliases': ['RTX3090', 'RTX 3090', 'rtx_3090', 'NVIDIA RTX 3090', 'GeForce RTX 3090']
    },

    # Google TPUs
    # TPU interconnect is hierarchical:
    # - ICI (Inter-Chip Interconnect): High-bandwidth, low-latency within pod/slice
    # - DCN (Data Center Network): Lower-bandwidth, higher-latency across pods
    # ICN value represents effective ICI bandwidth per chip
    # Reference: https://cloud.google.com/tpu/docs/system-architecture-tpu-vm
    'TPUv6': {
        'name': 'TPUv6',
        'Flops': 926,
        'Memory_size': 32,
        'Memory_BW': 1640,
        'ICN': 200,  # Estimated ICI bandwidth (GB/s) per chip
        'ICI_bandwidth': 200,  # ICI: ~200 GB/s per chip
        'DCN_bandwidth': 50,   # DCN: ~50 GB/s cross-pod
        'ICI_latency': 3e-6,   # ICI latency: ~3µs
        'DCN_latency': 200e-6, # DCN latency: ~200µs
        'tdp_watts': 300,  # TPU v6e (Trillium) published per-chip TDP (300W; vs H100 700W)
        'real_values': True,
        'type': 'asic',
        'manufacturer': 'Google',
        'architecture': 'TPU_V6',
        'generation': 'Tensor Processing Unit v6 (Trillium)',
        'release_year': 2024,
        'tensor_cores': 'custom_matrix_units',
        'memory_type': 'HBM',
        'aliases': ['TPU_v6', 'TPU v6', 'tpu-v6', 'Google TPU v6', 'Trillium']
    },
    'TPUv5e': {
        'name': 'TPUv5e',
        'Flops': 197,
        'Memory_size': 16,
        'Memory_BW': 820,
        'ICN': 100,  # ICI bandwidth (GB/s) per chip
        'ICI_bandwidth': 100,  # ICI: ~100 GB/s per chip
        'DCN_bandwidth': 25,   # DCN: ~25 GB/s cross-pod
        'ICI_latency': 5e-6,   # ICI latency: ~5µs
        'DCN_latency': 300e-6, # DCN latency: ~300µs
        'tdp_watts': 170,  # TPU v5e (cost-optimized) published per-chip power ~170W
        'real_values': True,
        'type': 'asic',
        'manufacturer': 'Google',
        'architecture': 'TPU_V5E',
        'generation': 'Tensor Processing Unit v5e (cost-optimized)',
        'release_year': 2023,
        'tensor_cores': 'custom_matrix_units',
        'memory_type': 'HBM',
        'aliases': ['TPU_v5e', 'TPU v5e', 'tpu-v5e', 'Google TPU v5e']
    },
    'TPUv5p': {
        'name': 'TPUv5p',
        'Flops': 459,
        'Memory_size': 95,
        'Memory_BW': 2765,
        'ICN': 150,  # ICI bandwidth (GB/s) per chip - higher than v5e
        'ICI_bandwidth': 150,  # ICI: ~150 GB/s per chip
        'DCN_bandwidth': 50,   # DCN: ~50 GB/s cross-pod
        'ICI_latency': 4e-6,   # ICI latency: ~4µs
        'DCN_latency': 250e-6, # DCN latency: ~250µs
        'tdp_watts': 450,  # TPU v5p (performance-optimized) published per-chip TDP 450W (TSMC N5)
        'real_values': True,
        'type': 'asic',
        'manufacturer': 'Google',
        'architecture': 'TPU_V5P',
        'generation': 'Tensor Processing Unit v5p (performance-optimized)',
        'release_year': 2023,
        'tensor_cores': 'custom_matrix_units',
        'memory_type': 'HBM',
        'aliases': ['TPU_v5p', 'TPU v5p', 'tpu-v5p', 'Google TPU v5p']
    },
    'TPUv4': {
        'name': 'TPUv4',
        'Flops': 275,
        'Memory_size': 32,
        'Memory_BW': 1228,
        'ICN': 100,  # ICI bandwidth (GB/s) per chip - was 24, severely underspecified
        'ICI_bandwidth': 100,  # ICI: ~100 GB/s per chip (4.8 Tb/s total / 6 links / 8 bits)
        'DCN_bandwidth': 25,   # DCN: ~25 GB/s cross-pod (InfiniBand-class)
        'ICI_latency': 5e-6,   # ICI latency: ~5µs (very low)
        'DCN_latency': 300e-6, # DCN latency: ~300µs (cross-datacenter)
        'tdp_watts': 192,  # TPU v4 per-chip power, Jouppi et al. ISCA 2023 (192W measured; 250W spec)
        'real_values': True,
        'type': 'asic',
        'manufacturer': 'Google',
        'architecture': 'TPU_V4',
        'generation': 'Tensor Processing Unit v4',
        'release_year': 2021,
        'tensor_cores': 'custom_matrix_units',
        'memory_type': 'HBM',
        'aliases': ['TPU_v4', 'TPU v4', 'tpu-v4', 'Google TPU v4']
    },
    
    # AMD GPUs
    'MI300X': {
        'name': 'MI300X',
        'Flops': 1307,
        'Memory_size': 192,
        'Memory_BW': 5300,
        'ICN': 896,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'AMD',
        'architecture': 'CDNA3',
        'generation': 'Instinct MI300 Series',
        'release_year': 2023,
        'matrix_cores': 'gen3',
        'rt_cores': None,
        'memory_type': 'HBM3',
        'cost': {
            'aws_on_demand': 3.50,
            'gcp_on_demand': 3.50,
            'coreweave': 2.50,
            'purchase_price_usd': 15000,  # Bulk ~$10k (Samsung), retail estimated $12k-$18k
            'tdp_watts': 750,
        },
    },
    'MI325X': {
        'name': 'MI325X',
        'Flops': 1307,
        'Memory_size': 256,
        'Memory_BW': 6000,
        'ICN': 400,
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'AMD',
        'architecture': 'CDNA3',
        'generation': 'Instinct MI300 Series',
        'release_year': 2024,
        'matrix_cores': 'gen3',
        'rt_cores': None,
        'memory_type': 'HBM3',
        'cost': {
            'aws_on_demand': 4.00,
            'gcp_on_demand': 4.00,
            'coreweave': 3.00,
            'purchase_price_usd': 20000,  # Estimated, ~25-30% premium over MI300X
            'tdp_watts': 750,
        },
    },
    
    # Intel GPUs
    'MAX1550': {
        'name': 'MAX1550',
        'Flops': 45.2,  # FP16 TFLOPS estimate
        'Memory_size': 128,
        'Memory_BW': 3276,
        'ICN': 300,
        'tdp_watts': 600,  # Intel Data Center GPU Max 1550 product spec: 600W TDP
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'Intel',
        'architecture': 'XE_HPC',
        'generation': 'Ponte Vecchio',
        'release_year': 2022,
        'compute_units': 'Xe-cores',
        'memory_type': 'HBM2e',
        'pci_ids': ['0bd5'],
        'aliases': ['MAX 1550', 'Ponte Vecchio', 'PVC', 'Data Center GPU Max 1550']
    },
    'MAX1100': {
        'name': 'MAX1100',
        'Flops': 32.7,  # FP16 TFLOPS estimate
        'Memory_size': 48,
        'Memory_BW': 1640,
        'ICN': 200,
        'tdp_watts': 300,  # Intel Data Center GPU Max 1100 product spec: 300W PCIe TDP
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'Intel',
        'architecture': 'XE_HPC',
        'generation': 'Ponte Vecchio',
        'release_year': 2022,
        'compute_units': 'Xe-cores',
        'memory_type': 'HBM2e',
        'pci_ids': ['0bd9'],
        'aliases': ['MAX 1100', 'Data Center GPU Max 1100']
    },
    'ARC770': {
        'name': 'ARC770',
        'Flops': 17.2,  # FP16 TFLOPS estimate
        'Memory_size': 16,
        'Memory_BW': 560,
        'ICN': 50,
        'tdp_watts': 225,  # Intel Arc A770 total board power: 225W
        'real_values': True,
        'type': 'gpu',
        'manufacturer': 'Intel',
        'architecture': 'XE_HPG',
        'generation': 'Arc Alchemist',
        'release_year': 2022,
        'compute_units': 'Xe-cores',
        'rt_cores': 'gen1',
        'memory_type': 'GDDR6',
        'pci_ids': ['56c0'],
        'aliases': ['Arc A770', 'Intel Arc A770']
    },
    
    # Intel Accelerators
    'Gaudi3': {
        'name': 'Gaudi3',
        'Flops': 1835,
        'Memory_size': 128,
        'Memory_BW': 3675,
        'ICN': 300,
        'tdp_watts': 900,  # Intel Gaudi 3 HL-325L OAM product brief: 900W TDP (air-cooled)
        'real_values': True,
        'type': 'accelerator',
        'manufacturer': 'Intel',
        'architecture': 'GAUDI3',
        'generation': 'Habana Gaudi',
        'release_year': 2024,
        'matrix_cores': 'custom'
    },
    
    # Intel CPUs
    'SapphireRapids_CPU': {
        # R2-CPU3: reconcile to the sourced per-socket datasheet values (Xeon Platinum 8480+,
        # cpu_specs.py XEON_PLATINUM_8480PLUS): 56c x 2.0GHz x 16 x 2-socket AVX-512 FP32 = 57.3 TFLOPS;
        # per-socket DRAM = 8ch x DDR5-4800 (38.4 GB/s/ch) = 307.2 GB/s. The eta_mem DDR band (0.65)
        # derates this for decode realism. Was 33 / 180 (an under-valued disjoint source).
        'Flops': 57.3,
        'Memory_size': 300,
        'Memory_BW': 307.2,
        'ICN': 100,
        'Power': 434,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'Intel',
        'name': 'Intel 4th Gen Xeon Scalable (Sapphire Rapids)',
        'url': 'https://www.intel.com/content/www/us/en/products/docs/processors/xeon/4th-gen-xeon-scalable-processors.html',
        'description': 'Up to 56-core server CPU featuring Advanced Matrix Extensions (AMX) for accelerated BF16/INT8 AI operations.',
        'aliases': ['Sapphire Rapids', 'Xeon Sapphire Rapids', 'Intel Sapphire Rapids', 'Xeon Platinum 8480+']
    },
    'EmeraldRapids_CPU': {
        # R2-CPU3: reconcile to the sourced per-socket datasheet values (Xeon Platinum 8580,
        # cpu_specs.py XEON_PLATINUM_8580): 56c x 2.0GHz x 16 x 2-socket x 1.05 IPC = 61.4 TFLOPS;
        # per-socket DRAM = 8ch x DDR5-5600 (44.8 GB/s/ch) = 358.4 GB/s. Was 47 / 350.
        'Flops': 61.4,
        'Memory_size': 300,
        'Memory_BW': 358.4,
        'ICN': 125,
        'Power': 289,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'Intel',
        'name': 'Intel 5th Gen Xeon Scalable (Emerald Rapids)',
        'url': 'https://www.intel.com/content/www/us/en/newsroom/news/future-xeon-emerald-rapids.html',
        'description': 'Successor to Sapphire Rapids with more cores/cache, improved DDR5 speeds, and faster UPI links.',
        'aliases': ['Emerald Rapids', 'Xeon Emerald Rapids', 'Intel Emerald Rapids', 'Xeon Platinum 8580']
    },
    'GraniteRapids_CPU': {
        'Flops': 86,
        'Memory_size': 300,
        'Memory_BW': 500,
        'ICN': 175,
        'Power': 450,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'Intel',
        'name': 'Intel Granite Rapids (Next-Gen Xeon)',
        'url': 'https://www.intel.com/content/www/us/en/newsroom/news/intel-advances-ai-everywhere.html',
        'description': 'Expected to offer significant enhancements in core count, memory bandwidth, and AI acceleration.'
    },
    # Measured deployment part (Granite Rapids-AP). Memory_BW is the SOURCED datasheet peak:
    # 8 channels x DDR5-6400 (6.4 GT/s x 8 B) = 409.6 GB/s per socket. The two variants below make the
    # socket count EXPLICIT so TP=1 (1 socket) and TP=2 (2 sockets) are each verifiable against a
    # matching real vLLM run. Flops inherits the family FP32 projection (compute/TTFT path uncalibrated).
    # (If the DIMMs are actually MRDIMM-8800, peak scales to ~563/1126 GB/s; the fitted decode eta scales
    # inversely so predictions are unchanged — only the eta/BW split relabels.)
    'Xeon6767P_CPU': {                 # full 2-socket node (matches vLLM TP=2)
        'Flops': 86,
        'Memory_size': 512,
        'Memory_BW': 819,              # 2 sockets x 409.6 GB/s (DDR5-6400, 8ch/socket)
        'ICN': 175,
        'Power': 700,
        'real_values': True,
        'type': 'cpu',
        'manufacturer': 'Intel',
        'name': 'Intel Xeon 6767P (Granite Rapids-AP), 2-socket',
        'aliases': ['Xeon 6767P', 'Xeon 6767P 2S', '6767P', 'Granite Rapids 6767P']
    },
    'Xeon6767P_1socket_CPU': {         # single socket (matches vLLM TP=1 baseline)
        'Flops': 43,
        'Memory_size': 256,
        'Memory_BW': 409.6,            # 1 socket x 8ch DDR5-6400
        'ICN': 175,
        'Power': 350,
        'real_values': True,
        'type': 'cpu',
        'manufacturer': 'Intel',
        'name': 'Intel Xeon 6767P (Granite Rapids-AP), 1-socket',
        'aliases': ['Xeon 6767P 1S', '6767P 1socket']
    },
    'SierraForest_CPU': {
        'Flops': 47,
        'Memory_size': 300,
        'Memory_BW': 400,
        'ICN': 125,
        'Power': 256,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'Intel',
        'name': 'Intel Sierra Forest (E-Core Xeon)',
        'url': 'https://www.intel.com/content/www/us/en/newsroom/news/intel-advances-ai-everywhere.html',
        'description': 'First Intel Xeon based on E-cores, targeting cloud-native workloads with high core density and energy efficiency.'
    },
    
    # AMD CPUs
    'MilanX_CPU': {
        'Flops': 36,
        'Memory_size': 512,
        'Memory_BW': 205,
        'ICN': 80,
        'Power': 225,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'AMD',
        'name': 'AMD EPYC Milan-X (3rd Gen)',
        'url': 'https://www.amd.com/en/products/processors/server/epyc/3rd-generation.html',
        'description': 'Features 3D V-Cache technology providing up to 768MB L3 cache per CPU for data-intensive workloads.',
        'aliases': ['Milan-X', 'EPYC Milan-X', 'AMD Milan-X', 'AMD EPYC Milan-X', 'EPYC 7773X']
    },
    'Genoa_CPU': {
        'Flops': 60,
        'Memory_size': 300,
        'Memory_BW': 460,
        'ICN': 125,
        'Power': 360,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'AMD',
        'name': 'AMD EPYC Genoa (4th Gen)',
        'url': 'https://www.amd.com/en/products/processors/server/epyc/9004-series.html',
        'description': 'Up to 96 Zen 4 cores with support for DDR5, PCIe Gen5, and CXL 1.1+ for enhanced performance.',
        'aliases': ['Genoa', 'EPYC Genoa', 'AMD Genoa', 'AMD EPYC Genoa', 'EPYC 9654']
    },
    'GenoaX_CPU': {
        'Flops': 60,
        'Memory_size': 512,
        'Memory_BW': 460,
        'ICN': 125,
        'Power': 380,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'AMD',
        'name': 'AMD EPYC Genoa-X (4th Gen with 3D V-Cache)',
        'url': 'https://www.amd.com/en/products/processors/server/epyc/9004-series.html',
        'description': 'Genoa with 3D V-Cache technology, providing up to 1.2GB L3 cache for memory-intensive applications.'
    },
    'Bergamo_CPU': {
        'Flops': 45,
        'Memory_size': 300,
        'Memory_BW': 460,
        'ICN': 125,
        'Power': 360,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'AMD',
        'name': 'AMD EPYC Bergamo (4th Gen Cloud Native)',
        'url': 'https://www.amd.com/en/products/processors/server/epyc/9004-series.html',
        'description': 'Up to 128 Zen 4c cores optimized for cloud-native workloads with high core density.'
    },
    'Turin_CPU': {
        'Flops': 98,
        'Memory_size': 300,
        'Memory_BW': 600,
        'ICN': 175,
        'Power': 426,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'AMD',
        'name': 'AMD EPYC Turin (5th Gen)',
        'url': 'https://www.amd.com/en/products/processors/server/epyc.html',
        'description': 'Expected to feature Zen 5 cores with enhanced IPC, memory bandwidth, and AI acceleration capabilities.'
    },
    
    # ARM CPUs
    'Grace_CPU': {
        'Flops': 74,
        'Memory_size': 512,
        'Memory_BW': 500,
        'ICN': 200,
        'Power': 500,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'NVIDIA',
        'name': 'NVIDIA Grace CPU',
        'url': 'https://www.nvidia.com/en-us/data-center/grace-cpu/',
        'description': 'ARM-based server CPU with 72 cores, designed for AI and HPC workloads with LPDDR5X memory.',
        'aliases': ['Grace', 'Grace CPU', 'NVIDIA Grace', 'NVIDIA Grace CPU']
    },
    'Graviton3_CPU': {
        'Flops': 20,
        'Memory_size': 300,
        'Memory_BW': 307,
        'ICN': 100,
        'Power': 100,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'AWS',
        'name': 'AWS Graviton3',
        'url': 'https://aws.amazon.com/ec2/graviton/',
        'description': '64-core ARM Neoverse V1 CPU with DDR5 support, optimized for cloud workloads.',
        'aliases': ['Graviton3', 'Graviton 3', 'AWS Graviton3', 'AWS Graviton 3']
    },
    'Graviton4_CPU': {
        'Flops': 40,
        'Memory_size': 300,
        'Memory_BW': 450,
        'ICN': 150,
        'Power': 135,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'AWS',
        'name': 'AWS Graviton4',
        'url': 'https://aws.amazon.com/ec2/graviton/',
        'description': 'Next-generation ARM CPU with improved performance, memory bandwidth, and AI inference capabilities.'
    },
    'AmpereOne_CPU': {
        'Flops': 23,
        'Memory_size': 300,
        'Memory_BW': 307,
        'ICN': 100,
        'Power': 350,
        'real_values': False,
        'type': 'cpu',
        'manufacturer': 'Ampere',
        'name': 'Ampere One',
        'url': 'https://amperecomputing.com/products/processors/ampere-one',
        'description': 'Up to 192-core ARM CPU designed for cloud-native applications with high single-thread performance.'
    },
    
    # Additional Accelerators
    'Trainium1': {
        'Flops': 190,
        'Memory_size': 32,
        'Memory_BW': 820,
        'ICN': 800,
        'real_values': False,
        'type': 'asic',
        'manufacturer': 'AWS',
        'name': 'AWS Trainium1',
        'url': 'https://aws.amazon.com/machine-learning/trainium/',
        'description': 'Purpose-built for training machine learning models with high performance and efficiency.'
    },
    'Inferentia2': {
        'Flops': 190,
        'Memory_size': 32,
        'Memory_BW': 820,
        'ICN': 800,
        'real_values': False,
        'type': 'asic',
        'manufacturer': 'AWS',
        'name': 'AWS Inferentia2',
        'url': 'https://aws.amazon.com/machine-learning/inferentia/',
        'description': 'Optimized for large-scale ML inference workloads with high throughput and low latency.'
    },
    'Cerebras_WSE2': {
        'Flops': 7500,
        'Memory_size': 40,
        'Memory_BW': 20000,
        'ICN': 2000,
        'Power': 15000,
        'real_values': False,
        'type': 'asic',
        'manufacturer': 'Cerebras',
        'name': 'Cerebras WSE-2',
        'url': 'https://www.cerebras.net/product-system/',
        'description': 'Wafer-scale processor with 850,000 cores and 40GB on-chip SRAM for massive parallel processing.'
    },
    'Cerebras_WSE3': {
        'Flops': 125000,
        'Memory_size': 44,
        'Memory_BW': 21000,
        'ICN': 7000,
        'Power': 23000,
        'real_values': False,
        'type': 'asic',
        'manufacturer': 'Cerebras',
        'name': 'Cerebras WSE-3',
        'url': 'https://www.cerebras.net/press-release/cerebras-announces-third-generation-wafer-scale-engine',
        'description': 'Third-generation wafer-scale processor with 900,000 cores at 5nm technology.'
    },
    'Groq_LPU': {
        'Flops': 250,
        'Memory_size': 230,
        'Memory_BW': 80000,
        'ICN': 400,
        'Power': 500,
        'real_values': False,
        'type': 'asic',
        'manufacturer': 'Groq',
        'name': 'Groq Language Processing Unit',
        'url': 'https://groq.com/',
        'description': 'Tensor Streaming Processor optimized for sequential processing and LLM inference.'
    },
    'SambaNova_SN40L': {
        'Flops': 688,
        'Memory_size': 1500,
        'Memory_BW': 1638,
        'ICN': 400,
        'Power': 1000,
        'real_values': False,
        'type': 'asic',
        'manufacturer': 'SambaNova',
        'name': 'SambaNova SN40L',
        'url': 'https://sambanova.ai/products/sn40l/',
        'description': 'Reconfigurable dataflow architecture for AI training and inference at scale.'
    },
    
    # Merge specific CPU SKUs
    **CPU_CONFIGS,
}

def is_cpu_hardware(hardware_name: str) -> bool:
    """Check if hardware name refers to a CPU (in HARDWARE_CONFIGS or CPU_PRESETS).

    Checks the ``type`` field of HARDWARE_CONFIGS entries first, then falls
    back to a case-insensitive lookup in ``CPU_PRESETS``.
    """
    if hardware_name in HARDWARE_CONFIGS:
        return HARDWARE_CONFIGS[hardware_name].get('type') == 'cpu'
    # Fall back to CPU_PRESETS for preset names not merged into HARDWARE_CONFIGS
    try:
        from llm_memory_calculator.genz.cpu.cpu_configs import CPU_PRESETS
        return hardware_name.lower() in [k.lower() for k in CPU_PRESETS]
    except ImportError:
        return False


def get_hardware_names() -> list:
    """Get list of all hardware names."""
    return list(HARDWARE_CONFIGS.keys())

def get_hardware_by_manufacturer(manufacturer: str) -> Dict[str, Dict[str, Any]]:
    """Get all hardware from a specific manufacturer."""
    return {
        name: config 
        for name, config in HARDWARE_CONFIGS.items() 
        if config.get('manufacturer', '').lower() == manufacturer.lower()
    }

def get_hardware_by_type(hw_type: str) -> Dict[str, Dict[str, Any]]:
    """Get all hardware of a specific type (gpu, cpu, asic, accelerator)."""
    return {
        name: config 
        for name, config in HARDWARE_CONFIGS.items() 
        if config.get('type', '').lower() == hw_type.lower()
    }

# ==================================================================================================
# INFERENCE REALISM: achieved efficiency + fixed per-step overhead, DECLARED per hardware record
# ==================================================================================================
# Two things a pure roofline gets structurally wrong, and that no amount of model-side fixing can
# repair, because both are properties of the DEVICE + its host serving runtime, not of the model:
#
#   1. ACHIEVED EFFICIENCY.  `max(compute_time, memory_time)` against the DATASHEET peak assumes a
#      kernel sustains 100% of peak FLOPs and 100% of peak DRAM bandwidth. Nothing does. Measured on
#      an H100 (vLLM 0.9.0, in-cluster, tp=1): a 55.93 GB decode step against the 3350 GB/s HBM3 peak
#      has a 16.7 ms floor and lands at 20.197 ms — 83% achieved bandwidth. Peak is a ceiling, never
#      an operating point.
#
#   2. FIXED PER-STEP COST.  The roofline is a THROUGHPUT model: it charges bytes and FLOPs and
#      nothing else. A real engine step also pays kernel dispatch, the scheduler, sampling and
#      detokenisation. Measured on the same H100, Qwen3-0.6B TPOT never drops below ~2.1 ms however
#      small the shape, while the roofline predicts 0.506 ms — i.e. the roofline models the ~1.6 ms
#      floor as exactly zero. That floor does not shrink when the model shrinks, so it dominates
#      small models and is invisible on large ones.
#
# Both are declared here, as ordinary fields on a hardware record, so they are auditable, per-device
# overridable, and never model-specific. Every field is OPTIONAL, and nothing ever falls back to a
# silent 1.0 / 0.0:
#   * the three OVERHEADS fall back to the per-device-CLASS default table below;
#   * the two EFFICIENCIES fall back to the documented per-TECHNOLOGY bands (memory-type decode MBU,
#     tensor-core-generation large-GEMM MFU) that get_inference_system already applies. Those bands
#     are the principled default for a rate, they are keyed off fields every record already carries
#     (`memory_type`, `tensor_cores`), and they deliberately keep living in exactly one place rather
#     than being copied per record where the two copies could drift apart.
#
# Declarable keys on any HARDWARE_CONFIGS record (see INFERENCE_REALISM_KEYS):
#
#   'compute_efficiency'        achieved fraction of the record's peak dense FLOPs   (0, 1]
#   'memory_efficiency'         achieved fraction of the record's peak DRAM bandwidth (0, 1]
#   'kernel_launch_latency_ms'  host->device dispatch cost charged PER OPERATOR        >= 0
#   'step_overhead_ms'          fixed host cost charged ONCE PER ENGINE STEP           >= 0
#   'per_sequence_overhead_ms'  host cost charged PER SEQUENCE in the batch            >= 0
#
# Each value is either a scalar (both phases) or a per-phase dict, e.g.
#     'kernel_launch_latency_ms': {'prefill': 0.005, 'decode': 0.002}
#
# PRECEDENCE (highest first), resolved by resolve_inference_realism():
#   1. the record's MEASURED `inference_calibration[phase]` block (fitted against real benchmarks by
#      llm_memory_calculator.validation.inference_calibration) — a measurement always beats a prior;
#   2. the record's own declared key (scalar or per-phase);
#   3. the per-device-class default below (overheads) / the per-technology band (efficiencies).
#
# WHY THE OVERHEADS LIVE ON THE HARDWARE RECORD.  They are host-runtime costs, not silicon. They are
# declared here because the hardware record is this simulator's single device declaration, and the
# defaults are stated for the mainstream serving runtime (a vLLM-class Python engine driving an
# accelerator). A deployment on a different runtime declares its own values on the record rather than
# being silently mispredicted.
#
# THE TWO TERMS MUST NOT BE FITTED TO THE SAME RESIDUAL.  The "83% achieved bandwidth" above is what
# you get by attributing the WHOLE 20.197 ms to streaming: 55.93 GB / 20.197 ms = 2769 GB/s = 0.83 of
# peak. But that step also pays the fixed host cost. Take it out first — ~2.05 ms for a 64-layer model
# at the defaults below — and the remaining 18.1 ms of streaming implies ~0.92 of peak. The published
# HBM3 decode-MBU band (~0.85, applied by get_inference_system when a record declares nothing) sits
# between the two, so nothing here hardcodes 0.83 for one device: a single measurement that conflates
# a rate with a fixed cost is a worse prior than the published band. A device that measures the two
# SEPARATELY declares both, and its declaration wins.
#
# NOT A FUDGE FACTOR. These are additive, model-independent terms with their own physical units.
# They do not scale with weight bytes, so they cannot absorb (or be inflated by) an error in the
# model's parameter count: when a model-config defect is fixed and the weight term grows, the
# roofline term grows and these do not move. Anything that must be tuned per model belongs in the
# model, not here.

#: Fields a hardware record may declare to describe its achieved efficiency and fixed overheads.
INFERENCE_REALISM_KEYS = (
    'compute_efficiency',
    'memory_efficiency',
    'kernel_launch_latency_ms',
    'step_overhead_ms',
    'per_sequence_overhead_ms',
)

#: Efficiency fields, validated to (0, 1]. The rest are latencies in ms, validated to >= 0.
_EFFICIENCY_KEYS = ('compute_efficiency', 'memory_efficiency')

#: Key spelling used inside a MEASURED `inference_calibration[phase]` block -> declaration key.
#: `c_stream_ms_per_layer` is deliberately absent: it is a separate, layer-scaled legacy term kept
#: for the devices whose measured fit used it, and is resolved by get_inference_system, not here.
_CALIBRATION_ALIASES = {
    'eta_compute': 'compute_efficiency',
    'eta_mem': 'memory_efficiency',
    't_launch_ms': 'kernel_launch_latency_ms',
    'step_overhead_ms': 'step_overhead_ms',
    'per_sequence_overhead_ms': 'per_sequence_overhead_ms',
}

# --------------------------------------------------------------------------------------------------
# Per-device-class defaults. Principled, published, and identical for every device in a class — no
# device is special-cased, and none of these was fitted to the measurements quoted above.
# --------------------------------------------------------------------------------------------------
#
# kernel_launch_latency_ms — charged per operator in the model graph.
#   A CUDA kernel launch costs ~3-10 us of host+device dispatch when issued eagerly; replaying the
#   same kernel as a node of a captured CUDA Graph costs ~1-2 us (NVIDIA "Getting Started with CUDA
#   Graphs"; the same order on ROCm/HIP graphs). Production engines capture DECODE — its shapes are
#   static — and run PREFILL eagerly, because prompt length varies every request. So:
#     decode  0.002 ms/op  (graph replay, lower edge of the 1-2 us band)
#     prefill 0.005 ms/op  (eager launch, lower edge of the 3-10 us band)
#   Both take the conservative lower edge: under-charging a term is safer than inventing one.
#
# step_overhead_ms — charged once per engine step, in both phases.
#   The host-side work around one forward pass: schedule the batch, build the block tables and input
#   tensors, launch, sample, and hand the tokens to the detokeniser. It is Python-side and roughly
#   constant, which is exactly why it shows up as a floor. Reported at the ~1-3 ms/step scale for
#   vLLM-class engines (the motivation for vLLM's V1 rewrite, its async output processing, and the
#   CUDA-graph/"piecewise" work; TensorRT-LLM's in-flight batching makes the same argument). We take
#   1.0 ms — the lower edge of the reported band.
#
# per_sequence_overhead_ms — charged per sequence in the batch.  DEFAULT 0.0, BECAUSE IT WAS MEASURED.
#   The term exists for runtimes that do per-request host work INSIDE the engine step (per-sequence
#   sampling, block-table updates, stop checks, detokenisation) and is kept declarable for them. The
#   GPU-class default was 0.25 ms, inferred from a fit over batch 1 and batch 10 only. That fit was
#   contaminated: its batch-10 points came from a 2-replica deployment (5 sequences per engine),
#   identical prompts (vLLM's prefix cache let the batch share one KV copy) and requests stopping at
#   their own EOS (the batch drained mid-decode). All three made batch 10 look cheaper than it is,
#   and the fit split the error between a per-sequence term and an attention coefficient of ~0.35.
#
#   Re-measured with those artifacts removed — one engine addressed directly, a unique prompt per
#   request, ignore_eos with a fixed length, steady-state inter-token gap after every prefill has
#   finished, captured CUDA-graph batch sizes only — on an H100XM-80C (vLLM V1, tp=1):
#     Qwen3-0.6B   batch 1..32, context 2k..32k, 12 points    per-seq 0.25: 1.08-2.69x   0.0: 0.95-1.08x
#     gpt-oss-20b  batch 1..32, context 2k..32k, 10 points    per-seq 0.25: 0.93-2.09x   0.0: 0.86-0.99x
#   (gpt-oss with its MXFP4 experts sized at their stored precision; see genz/weight_precision.py)
#   The 0.25 default is not merely a worse fit, it is physically refuted: at 2k context and batch 32
#   the term alone would add 8.0 ms to a step that measures 4.75 ms IN TOTAL. Even charging the KV
#   read as free, the per-sequence cost is bounded by (4.75 - 1.0 step - 0.36 weights) / 32 = 0.10 ms.
#   That matches how vLLM V1 is built: sampling runs batched on the GPU and detokenisation runs in
#   the API-server process, outside the engine step. And step_overhead_ms is not being re-fitted to
#   absorb anything: 1.0 ms is independently confirmed by the same data (0.5 and 1.5 both fit worse).
#   A runtime that does per-request work inside its step declares its own value on the record.
#
# CPU (type == 'cpu') declares ZERO for all three, deliberately. The CPU decode path already clamps
# to a memory floor whose efficiency constant (_ETA_MEM_DECODE_CPU_X86 in llm_decode.py) was fitted
# against END-TO-END measured decode latency, i.e. with the host overhead already inside it. Adding
# these terms on top would double-count it. A CPU whose runtime overhead is measured SEPARATELY from
# that floor can declare them on its record.
_GPU_CLASS_REALISM = {
    'decode':  {'kernel_launch_latency_ms': 0.002,
                'step_overhead_ms': 1.0,
                'per_sequence_overhead_ms': 0.0},
    'prefill': {'kernel_launch_latency_ms': 0.005,
                'step_overhead_ms': 1.0,
                'per_sequence_overhead_ms': 0.0},
}
_ZERO_CLASS_REALISM = {
    'decode':  {'kernel_launch_latency_ms': 0.0, 'step_overhead_ms': 0.0, 'per_sequence_overhead_ms': 0.0},
    'prefill': {'kernel_launch_latency_ms': 0.0, 'step_overhead_ms': 0.0, 'per_sequence_overhead_ms': 0.0},
}

#: Fixed-overhead defaults by device class. Accelerators/ASICs/TPUs are host-driven the same way a
#: GPU is, so they share the GPU-class defaults rather than silently getting zero.
INFERENCE_REALISM_DEFAULTS: Dict[str, Dict[str, Dict[str, float]]] = {
    'gpu': _GPU_CLASS_REALISM,
    'accelerator': _GPU_CLASS_REALISM,
    'asic': _GPU_CLASS_REALISM,
    'tpu': _GPU_CLASS_REALISM,
    'cpu': _ZERO_CLASS_REALISM,
}

#: Device class used when a record does not declare `type`.
DEFAULT_DEVICE_CLASS = 'gpu'

INFERENCE_PHASES = ('prefill', 'decode')


def _realism_device_class(hardware: Any) -> str:
    """Device class of `hardware` ('gpu' / 'cpu' / ...), for the per-class defaults."""
    try:  # a CPUSystem carries no `type`, but is unambiguously a CPU
        from llm_memory_calculator.genz.cpu.cpu_system import CPUSystem
        if isinstance(hardware, CPUSystem):
            return 'cpu'
    except Exception:  # pragma: no cover - cpu extras absent
        pass
    if isinstance(hardware, dict):
        declared = hardware.get('type')
    else:
        declared = getattr(hardware, 'type', None)
    declared = str(declared or '').lower()
    return declared if declared in INFERENCE_REALISM_DEFAULTS else DEFAULT_DEVICE_CLASS


def _realism_record(hardware: Any) -> Any:
    """Resolve `hardware` to something whose declarations we can read.

    A name is looked up (aliases included); a dict / System / CPUSystem is used as-is.
    """
    if isinstance(hardware, str):
        try:
            from llm_memory_calculator.hardware.manager import get_hardware_config
            resolved = get_hardware_config(hardware)
        except Exception:  # pragma: no cover - manager import problems
            resolved = None
        if not resolved:  # get_hardware_config returns False (not None) for an unknown name
            resolved = HARDWARE_CONFIGS.get(hardware)
        if not resolved:
            # Same fallback chain get_inference_system uses, so a name it can resolve never dies here.
            try:
                from llm_memory_calculator.systems.system_configs import system_configs
                resolved = system_configs.get(hardware)
            except Exception:  # pragma: no cover - backward-compat shim unavailable
                resolved = None
        if not resolved:
            raise ValueError(
                f"Unknown hardware '{hardware}': cannot resolve its inference-realism declaration. "
                f"Pass a known hardware name or an explicit config dict."
            )
        return resolved
    return hardware


def _for_phase(value: Any, phase: str) -> Any:
    """Unwrap a declaration that may be a scalar or a {'prefill': .., 'decode': ..} dict."""
    if isinstance(value, dict):
        return value.get(phase)
    return value


def _declared(record: Any, key: str, phase: str) -> Any:
    """The value `record` DECLARES for `key` in `phase`, or None if it declares nothing.

    For a hardware-config dict, a declaration is simply the key. For a System / CPUSystem object
    there is no record to read, and its CONSTRUCTOR DEFAULTS must not be mistaken for declarations:
    ``System.kernel_launch_latency_ms`` is 0.0 and ``System.compute_efficiency`` is 1 on every
    freshly built System, and reading those back would re-assert exactly the 100%-of-peak,
    zero-overhead model this module exists to remove. So an object declares only via an explicit
    ``inference_realism`` mapping, plus an efficiency it was deliberately built with (!= 1, the same
    "already set" convention get_inference_system uses).
    """
    if isinstance(record, dict):
        return _for_phase(record.get(key), phase)
    explicit = getattr(record, 'inference_realism', None)
    if isinstance(explicit, dict):
        value = _for_phase(explicit.get(key), phase)
        if value is not None:
            return value
    if key in _EFFICIENCY_KEYS:
        value = getattr(record, key, None)
        try:
            if value is not None and float(value) != 1.0:
                return value
        except (TypeError, ValueError):
            return value
    return None


def _validate(key: str, value: Any, device: str) -> float:
    """Reject a nonsensical declaration loudly. A silently-clamped efficiency is how a confident
    wrong number gets shipped."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{device}: inference-realism '{key}' must be a number, got {value!r}")
    if value != value or value in (float('inf'), float('-inf')):
        raise ValueError(f"{device}: inference-realism '{key}' must be finite, got {value!r}")
    if key in _EFFICIENCY_KEYS:
        if not 0.0 < value <= 1.0:
            raise ValueError(
                f"{device}: inference-realism '{key}' is an ACHIEVED FRACTION OF PEAK and must lie in "
                f"(0, 1]; got {value!r}. A value above 1 would claim the device beats its datasheet."
            )
    elif value < 0.0:
        raise ValueError(f"{device}: inference-realism '{key}' is a latency in ms and must be >= 0, got {value!r}")
    return value


def resolve_inference_realism(hardware: Any, phase: str) -> Dict[str, Optional[float]]:
    """Achieved efficiency and fixed overheads declared for `hardware` in `phase`.

    Args:
        hardware: a hardware name, a HARDWARE_CONFIGS-shaped dict, or a System/CPUSystem object.
        phase: 'prefill' or 'decode'.

    Returns:
        A dict over INFERENCE_REALISM_KEYS. The two efficiencies are ``None`` when the record
        declares neither a measured calibration nor an explicit value — the caller then keeps the
        documented per-technology band (memory-type MBU / tensor-core-generation GEMM MFU) that
        get_inference_system already applies, which IS the principled default for those two and
        lives in one place. The three overheads are always a concrete number: their principled
        default is the per-device-class value in INFERENCE_REALISM_DEFAULTS.

    Raises:
        ValueError: for an unknown hardware name, an unknown phase, or an out-of-range declaration.
    """
    if phase not in INFERENCE_PHASES:
        raise ValueError(f"phase must be one of {INFERENCE_PHASES}, got {phase!r}")

    record = _realism_record(hardware)
    device_class = _realism_device_class(record)
    device = str((record.get('name') if isinstance(record, dict) else None)
                 or (hardware if isinstance(hardware, str) else device_class))
    calibration = (record.get('inference_calibration') if isinstance(record, dict)
                   else getattr(record, 'inference_calibration', None)) or {}
    measured = calibration.get(phase, {}) if isinstance(calibration, dict) else {}

    # A MEASURED calibration block is the authority for its phase: it was fitted against real
    # end-to-end benchmarks, so whatever overhead it does not name was already absorbed into the
    # terms it does name. Layering a class-default overhead on top of a measured fit would
    # double-count it and silently break that device's published residuals. So a calibrated phase
    # defaults its unstated overheads to 0; only an UNcalibrated phase gets the class priors.
    defaults = ({k: 0.0 for k in INFERENCE_REALISM_KEYS if k not in _EFFICIENCY_KEYS} if measured
                else INFERENCE_REALISM_DEFAULTS[device_class][phase])

    out: Dict[str, Optional[float]] = {}
    for key in INFERENCE_REALISM_KEYS:
        value = None
        # 1. measured calibration block (highest authority)
        for cal_key, decl_key in _CALIBRATION_ALIASES.items():
            if decl_key == key and isinstance(measured, dict) and cal_key in measured:
                value = measured[cal_key]
                break
        # 2. the record's own declaration
        if value is None:
            value = _declared(record, key, phase)
        # 3. per-device-class default (efficiencies have none here — see the docstring)
        if value is None:
            value = defaults.get(key)
        out[key] = None if value is None else _validate(key, value, device)
    return out


def apply_inference_realism(system: Any, realism: Dict[str, Optional[float]],
                            compute_efficiency: Optional[float] = None,
                            memory_efficiency: Optional[float] = None) -> Any:
    """Attach the resolved efficiencies and fixed overheads to a constructed System.

    The three overheads are additive terms applied by llm_prefill/llm_decode after the roofline, so
    they simply ride on the System object.

    The efficiencies are also stamped back when the caller resolved a concrete value, because
    get_inference_system treats ``efficiency == 1`` as "caller said nothing" and substitutes its
    per-technology band. That is the right default for an UNDECLARED device, but it would silently
    discard a device (or a caller) that deliberately declares 1.0 — which is exactly the audit case
    "declare no derating and no overhead, and reproduce the untouched roofline". Pass None for an
    efficiency to leave get_inference_system's band in place.
    """
    if compute_efficiency is not None:
        system.compute_efficiency = compute_efficiency
    if memory_efficiency is not None:
        system.memory_efficiency = memory_efficiency
    system.kernel_launch_latency_ms = realism.get('kernel_launch_latency_ms') or 0.0
    system.step_overhead_ms = realism.get('step_overhead_ms') or 0.0
    system.per_sequence_overhead_ms = realism.get('per_sequence_overhead_ms') or 0.0
    return system
