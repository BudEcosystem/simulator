# Hardware Schema v2 Documentation

## Overview

The hardware schema has been updated to support more granular pricing information:

1. **On-premise vendors** can now have individual pricing (lower/upper bounds)
2. **Cloud providers** can have multiple instance types, each with its own pricing
3. Backward compatibility is maintained for existing code

## New Data Structure

### On-Premise Vendors

Previously, all vendors shared the same pricing. Now each vendor can have different pricing:

```json
"on_prem_vendors": [
    {
        "name": "Dell",
        "price_lower": 10000,
        "price_upper": 12000
    },
    {
        "name": "HPE",
        "price_lower": 11000,
        "price_upper": 13000
    },
    "Lenovo"  // Vendor without specific pricing
]
```

### Cloud Support

Cloud providers now support multiple instance types with individual pricing:

```json
"clouds": [
    {
        "name": "AWS",
        "regions": ["us-east-1", "us-west-2"],
        "instances": [
            {
                "name": "p5.48xlarge",
                "price_lower": 3.5,
                "price_upper": 4.0
            },
            {
                "name": "p5.24xlarge", 
                "price_lower": 2.0,
                "price_upper": 2.5
            }
        ]
    }
]
```

## Database Schema Changes

### New Tables

1. **hardware_on_prem_vendors**
   - Stores vendor information with individual pricing
   - Links to hardware via hardware_id
   - Unique constraint on (hardware_id, vendor_name)

2. **hardware_clouds**
   - Stores cloud provider information
   - Links to hardware via hardware_id
   - Unique constraint on (hardware_id, cloud_name)

3. **hardware_cloud_instances**
   - Stores instance types with pricing
   - Links to cloud via cloud_id
   - Supports multiple instances per cloud

4. **hardware_cloud_regions** (updated)
   - Now links to cloud_id instead of hardware_id
   - Supports regions at the cloud level

### Removed Fields

The following fields have been removed from the main hardware table:
- `on_prem_price_lower`
- `on_prem_price_upper`

These are now stored in the vendor-specific tables.

## API Changes

### New Endpoints

1. **GET /api/hardware/{hardware_name}**
   - Returns detailed hardware info including vendor and cloud pricing
   - Response includes `vendors` and `clouds` arrays with full pricing details

2. **GET /api/hardware/vendors/{hardware_name}**
   - Returns vendor information with pricing for specific hardware

3. **GET /api/hardware/clouds/{hardware_name}**
   - Returns cloud availability with instance types and pricing

### Updated Request Models

When creating hardware via POST /api/hardware:

```python
{
    "name": "NewGPU",
    "type": "gpu",
    "flops": 1000,
    "memory_size": 80,
    "memory_bw": 2000,
    "on_prem_vendors": [
        {"name": "Vendor1", "price_lower": 10000, "price_upper": 12000},
        "Vendor2"  # Without pricing
    ],
    "clouds": [
        {
            "name": "AWS",
            "regions": ["us-east-1"],
            "instances": [
                {"name": "p5.xlarge", "price_lower": 3.0, "price_upper": 3.5}
            ]
        }
    ]
}
```

## Migration Guide

### For Existing Databases

Run the migration script:

```bash
python scripts/migrate_hardware_schema.py
```

This will:
1. Create a backup of your database
2. Migrate existing data to the new schema
3. Preserve all existing hardware information
4. Convert single pricing to vendor-specific pricing

### For New Installations

The new schema will be created automatically when you first use the BudHardware class.

### Backward Compatibility

The following features ensure backward compatibility:

1. **Simple vendor lists still work**: You can still pass a list of vendor names without pricing
2. **Old import format supported**: The import_from_json method handles both old and new formats
3. **GenZ compatibility maintained**: The get_config() method still returns the expected format

## Example Usage

### Python API

```python
from BudSimulator.src.hardware import BudHardware

hw = BudHardware()

# Add hardware with new pricing structure
hw.add_hardware({
    'name': 'H100_GPU',
    'type': 'gpu',
    'flops': 1979,
    'memory_size': 80,
    'memory_bw': 3350,
    'on_prem_vendors': [
        {'name': 'Dell', 'price_lower': 30000, 'price_upper': 35000},
        {'name': 'HPE', 'price_lower': 32000, 'price_upper': 36000}
    ],
    'clouds': [
        {
            'name': 'AWS',
            'regions': ['us-east-1', 'us-west-2'],
            'instances': [
                {'name': 'p5.48xlarge', 'price_lower': 8.0, 'price_upper': 10.0}
            ]
        }
    ]
})

# Get vendor pricing
vendors = hw.get_hardware_vendors('H100_GPU')
# Returns: [
#     {'vendor_name': 'Dell', 'price_lower': 30000, 'price_upper': 35000},
#     {'vendor_name': 'HPE', 'price_lower': 32000, 'price_upper': 36000}
# ]

# Get cloud pricing
clouds = hw.get_hardware_clouds('H100_GPU')
# Returns: [
#     {
#         'cloud_name': 'AWS',
#         'instance_name': 'p5.48xlarge',
#         'price_lower': 8.0,
#         'price_upper': 10.0,
#         'regions': ['us-east-1', 'us-west-2']
#     }
# ]
```

### REST API

```bash
# Get detailed hardware info
curl http://localhost:8000/api/hardware/H100_GPU

# Get vendor pricing
curl http://localhost:8000/api/hardware/vendors/H100_GPU

# Get cloud pricing
curl http://localhost:8000/api/hardware/clouds/H100_GPU
```

## Benefits

1. **More accurate pricing**: Different vendors often have different pricing for the same hardware
2. **Cloud instance flexibility**: Support for multiple instance types per cloud provider
3. **Better cost estimation**: Can calculate costs based on specific vendor or instance type
4. **Extensibility**: Easy to add more pricing dimensions in the future 