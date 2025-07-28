# Hardware Schema Update Summary

## Overview

The hardware data structure has been updated to support more granular pricing information:

1. **On-premise vendors** can now have individual pricing (lower/upper bounds)
2. **Cloud providers** can have multiple instance types, each with its own pricing
3. Full backward compatibility is maintained

## Key Changes Made

### 1. Database Schema Updates

**New Tables Created:**
- `hardware_on_prem_vendors` - Stores vendor-specific pricing
- `hardware_clouds` - Stores cloud provider relationships
- `hardware_cloud_instances` - Stores instance types with pricing
- `hardware_cloud_regions` (updated) - Now linked to clouds instead of hardware
- `hardware_legacy_pricing` - Stores old pricing data for migration

**Removed from hardware table:**
- `on_prem_price_lower`
- `on_prem_price_upper`

### 2. Code Updates

**Updated Files:**
- `src/db/hardware_schema.py` - New schema definition with version 2
- `src/hardware.py` - Updated BudHardware class to support new structure
- `apis/routers/hardware.py` - Updated API endpoints for new data structure
- `scripts/import_hardware_json.py` - Updated import script
- `scripts/migrate_hardware_schema.py` - New migration script

**New Test Files:**
- `tests/test_hardware_new_schema.py` - Comprehensive tests for new features

### 3. API Changes

**New Endpoints:**
- `GET /api/hardware/{hardware_name}` - Returns detailed info with vendor/cloud pricing
- `GET /api/hardware/vendors/{hardware_name}` - Get vendor pricing details
- `GET /api/hardware/clouds/{hardware_name}` - Get cloud instance pricing

**Updated Request Format:**
```json
{
    "name": "NewGPU",
    "type": "gpu",
    "flops": 1000,
    "memory_size": 80,
    "memory_bw": 2000,
    "on_prem_vendors": [
        {"name": "Vendor1", "price_lower": 10000, "price_upper": 12000},
        "Vendor2"  // Without pricing
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

### 4. Data Structure Examples

**On-Premise Vendors:**
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
    }
]
```

**Cloud Support:**
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

## Migration Instructions

### For Existing Users

1. **Backup your database** (automatic with migration script)
2. Run the migration script:
   ```bash
   python scripts/migrate_hardware_schema.py
   ```
3. Verify your data is migrated correctly
4. Optionally clean up old tables after verification

### For New Users

The new schema will be created automatically when first using BudHardware.

## Backward Compatibility

The following ensures backward compatibility:

1. **Simple vendor lists still work** - You can pass vendor names without pricing
2. **Old import format supported** - The import_from_json handles both formats
3. **GenZ compatibility maintained** - The get_config() method returns expected format
4. **System configs unchanged** - Original system_configs.py remains unmodified

## Testing

All tests pass with the new schema:
- 21 original hardware tests
- 6 new schema-specific tests
- Full backward compatibility verified

## Benefits

1. **Accurate Pricing** - Different vendors have different prices
2. **Cloud Flexibility** - Multiple instance types per cloud provider
3. **Better Cost Estimation** - Can calculate based on specific vendor/instance
4. **Future Extensibility** - Easy to add more pricing dimensions

## Import Results

Successfully imported 26/31 hardware items from hardware.json:
- 5 failures due to zero memory size or duplicate names
- All vendor and cloud information preserved
- Instance types and regions correctly mapped 