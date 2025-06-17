# Gated Model Support - Quick Implementation Guide

## What's Being Added

Support for gated HuggingFace models by allowing users to manually provide model configurations when the model requires access approval.

## Key Changes

### 1. Database
- New table: `user_model_configs` to store user-provided configurations
- Links to existing `models` table via `model_id`

### 2. API Changes

#### Enhanced ValidateModelResponse
```json
{
    "valid": false,
    "error": "Model is gated and requires access approval",
    "error_code": "MODEL_GATED",        // NEW
    "model_id": "meta-llama/Llama-3-70b", // NEW
    "requires_config": true,             // NEW
    "config_submission_url": "/api/models/config/submit" // NEW
}
```

#### New Endpoints
- `POST /api/models/config/submit` - Submit user config (requires `model_uri` and `config`)
- `GET /api/models/config/check/{model_id}` - Check if config exists

### 3. Backend Logic Flow
```
1. User initiates action (calculate/analyze/compare)
2. Check user_model_configs table first
3. If not found, check models table
4. If not found, try HuggingFace
5. If gated, return special error code
6. Frontend shows config input (stores pending action)
7. User submits config with model_uri
8. Config validated and saved
9. Frontend automatically continues with original action
10. Results shown immediately (no re-submission needed)
```

### 4. Frontend Changes
- Detect `error_code: "MODEL_GATED"`
- Store the pending action (calculate/analyze/compare) and parameters
- Show textarea for config input
- Submit config with `model_uri` (from original input) and pasted `config`
- On success, automatically continue with stored pending action
- Results shown immediately without user re-submission

## Implementation Order

1. **Database Migration** - Add `user_model_configs` table
2. **Update ModelManager** - Add `get_user_config()` and `save_user_config()`
3. **Update API Schemas** - Add new fields to `ValidateModelResponse`
4. **Add New Endpoints** - Config submission and checking
5. **Update Existing Endpoints** - Check user configs first
6. **Frontend Updates** - Add config input UI and logic

## Key Files to Modify

- `src/db/schema.py` - Add new table definition
- `src/db/model_manager.py` - Add user config methods
- `apis/schemas.py` - Update response models
- `apis/routers/models.py` - Update endpoints
- `src/bud_models.py` - Update HuggingFaceConfigLoader
- `frontend/src/components/ModelDetails.tsx` - Add config input UI

## Testing Checklist

- [ ] User can submit config for gated model
- [ ] Config is saved to database
- [ ] Saved config is used for calculations
- [ ] All existing functionality still works
- [ ] Invalid configs are rejected
- [ ] Multiple users can save different configs 