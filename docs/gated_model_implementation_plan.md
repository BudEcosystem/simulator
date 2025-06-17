# Gated Model Support Implementation Plan

## Overview

This document outlines the implementation plan for supporting gated models in BudSimulator. When a model on HuggingFace requires access approval (gated model), the system will:

1. Detect and communicate the gated status to the frontend
2. Allow users to manually provide model configurations
3. Store user-provided configurations for future use
4. Seamlessly integrate user configs with existing functionality

## User Experience Flow

From the user's perspective, the flow will be:

1. User enters a gated model URL (e.g., "meta-llama/Llama-3-70b") and clicks "Calculate Memory" (or any action)
2. System detects the model is gated and shows a message with config input area
3. User obtains config.json from HuggingFace (after getting access) and pastes it
4. User clicks "Submit Configuration"
5. System validates and saves the configuration
6. **Immediately continues with the original action** (memory calculation/analysis/etc.)
7. Results are shown as if the model was never gated
8. Future requests for the same model automatically use the saved config

**Key Point**: The user's workflow is not interrupted - they get their calculation results immediately after submitting the config, without needing to re-initiate the action.

## Design Principles

- **Backward Compatibility**: All existing APIs and functionality must continue to work unchanged
- **Seamless Integration**: User-provided configs should work identically to fetched configs
- **Security**: Validate all user inputs to prevent injection attacks
- **Performance**: Cache user configs to avoid repeated database queries
- **User Experience**: Clear error messages and intuitive UI flow

## System Architecture

### 1. Database Schema Changes

#### New Table: `user_model_configs`

```sql
CREATE TABLE IF NOT EXISTS user_model_configs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT UNIQUE NOT NULL,        -- e.g., "meta-llama/Llama-3-70b"
    config_json TEXT NOT NULL,            -- User-provided configuration
    source TEXT DEFAULT 'user_provided',  -- Source of the config
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_verified BOOLEAN DEFAULT 0,        -- Whether config has been verified
    FOREIGN KEY (model_id) REFERENCES models(model_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_user_configs_model_id ON user_model_configs(model_id);
```

### 2. API Response Modifications

#### Enhanced Error Response for Gated Models

Current error response:
```json
{
    "valid": false,
    "error": "Model is gated and requires access approval"
}
```

New error response:
```json
{
    "valid": false,
    "error": "Model is gated and requires access approval",
    "error_code": "MODEL_GATED",
    "model_id": "meta-llama/Llama-3-70b",
    "requires_config": true,
    "config_submission_url": "/api/models/config/submit"
}
```

### 3. New API Schemas

#### Request/Response Models

```python
# In schemas.py

class ConfigSubmitRequest(BaseModel):
    """Request model for submitting a model configuration."""
    model_uri: str = Field(..., description="HuggingFace model URI", example="meta-llama/Llama-3-70b")
    config: Dict[str, Any] = Field(..., description="Model configuration from config.json")

class ConfigValidation(BaseModel):
    """Validation details for a submitted config."""
    valid: bool = Field(..., description="Whether the config is valid")
    model_type: str = Field(..., description="Detected model type", example="decoder-only")
    attention_type: Optional[str] = Field(None, description="Detected attention type", example="gqa")
    parameter_count: Optional[int] = Field(None, description="Calculated parameter count", example=70000000000)

class ConfigSubmitResponse(BaseModel):
    """Response model for config submission."""
    success: bool = Field(..., description="Whether the config was saved successfully")
    message: str = Field(..., description="Success or error message")
    model_id: str = Field(..., description="The model ID for future reference")
    validation: Optional[ConfigValidation] = Field(None, description="Validation details if successful")
    error_code: Optional[str] = Field(None, description="Error code if failed", example="INVALID_CONFIG")
    missing_fields: Optional[List[str]] = Field(None, description="List of missing required fields")

class ConfigCheckResponse(BaseModel):
    """Response model for checking if a user config exists."""
    has_config: bool = Field(..., description="Whether a user config exists")
    model_id: str = Field(..., description="The model ID")
    submitted_at: Optional[str] = Field(None, description="When the config was submitted")
    is_verified: bool = Field(False, description="Whether the config has been verified")

# Enhanced ValidateModelResponse
class ValidateModelResponse(BaseModel):
    """Response model for model validation."""
    valid: bool = Field(..., description="Whether the model URL is valid")
    error: Optional[str] = Field(None, description="Error message if invalid")
    error_code: Optional[str] = Field(None, description="Error code for specific errors", example="MODEL_GATED")
    model_id: Optional[str] = Field(None, description="Model identifier for reference")
    requires_config: Optional[bool] = Field(None, description="Whether user needs to provide config")
    config_submission_url: Optional[str] = Field(None, description="URL to submit config")
```

### 4. New API Endpoints

#### POST `/api/models/config/submit`

Submit a user-provided model configuration and immediately validate it.

**Request:**
```json
{
    "model_uri": "meta-llama/Llama-3-70b",  // HuggingFace URI the user originally entered
    "config": {
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 8192,
        "num_hidden_layers": 80,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "intermediate_size": 28672,
        "vocab_size": 128256,
        "max_position_embeddings": 8192,
        "rope_theta": 500000.0,
        "hidden_act": "silu"
    }
}
```

**Response (Success):**
```json
{
    "success": true,
    "message": "Configuration saved and validated successfully",
    "model_id": "meta-llama/Llama-3-70b",
    "validation": {
        "valid": true,
        "model_type": "decoder-only",
        "attention_type": "gqa",
        "parameter_count": 70000000000
    }
}
```

**Response (Invalid Config):**
```json
{
    "success": false,
    "message": "Invalid configuration: missing required field 'hidden_size'",
    "error_code": "INVALID_CONFIG",
    "required_fields": ["hidden_size", "num_hidden_layers", "num_attention_heads"]
}
```

#### GET `/api/models/config/check/{model_id}`

Check if a user-provided config exists for a model.

**Response:**
```json
{
    "has_config": true,
    "model_id": "meta-llama/Llama-3-70b",
    "submitted_at": "2024-12-17T10:30:00Z",
    "is_verified": false
}
```

### 4. Modified Existing Endpoints

All model-related endpoints will be modified to check for user configs first:

1. `POST /api/models/validate`
2. `GET /api/models/{model_id}/config`
3. `POST /api/models/calculate`
4. `POST /api/models/analyze`
5. `POST /api/models/compare`

#### Modified Flow (example for `/validate`):

```python
@router.post("/validate", response_model=ValidateModelResponse)
async def validate_model(request: ValidateModelRequest):
    try:
        # NEW: Check user-provided configs first
        user_config = model_manager.get_user_config(request.model_url)
        if user_config:
            return ValidateModelResponse(valid=True, error=None)
        
        # Check if it's already in the database
        db_model = model_manager.get_model(request.model_url)
        if db_model:
            return ValidateModelResponse(valid=True, error=None)
        
        # Try to fetch from HuggingFace
        try:
            config = hf_loader.fetch_model_config(request.model_url)
            return ValidateModelResponse(valid=True, error=None)
        except Exception as e:
            error_msg = str(e)
            if "gated" in error_msg.lower():
                # NEW: Return enhanced error for gated models
                return ValidateModelResponse(
                    valid=False,
                    error="Model is gated and requires access approval",
                    error_code="MODEL_GATED",
                    model_id=request.model_url,
                    requires_config=True,
                    config_submission_url="/api/models/config/submit"
                )
            # ... rest of error handling
```

### 5. Frontend Implementation

#### Detecting Gated Models and Continuing Flow

```typescript
interface ValidateResponse {
    valid: boolean;
    error?: string;
    error_code?: string;
    model_id?: string;
    requires_config?: boolean;
    config_submission_url?: string;
}

// Store the original action the user was trying to perform
interface PendingAction {
    type: 'calculate' | 'analyze' | 'compare';
    params: any;
}

const [pendingAction, setPendingAction] = useState<PendingAction | null>(null);

const handleCalculateMemory = async (params: CalculateParams) => {
    try {
        // First validate the model
        const validateResponse = await api.post<ValidateResponse>('/models/validate', {
            model_url: params.model_id
        });
        
        if (!validateResponse.data.valid && validateResponse.data.error_code === 'MODEL_GATED') {
            // Store the action to continue after config submission
            setPendingAction({ type: 'calculate', params });
            setShowConfigInput(true);
            setGatedModelId(validateResponse.data.model_id);
            return; // Wait for config submission
        }
        
        // If valid, continue with calculation
        const result = await api.post('/models/calculate', params);
        setCalculationResult(result.data);
    } catch (error) {
        handleError(error);
    }
};
```

#### Config Input UI

```tsx
{showConfigInput && (
    <div className="mt-4 p-4 border-2 border-yellow-400 rounded-lg bg-yellow-50">
        <h3 className="text-lg font-semibold mb-2">
            Model Configuration Required
        </h3>
        <p className="text-sm text-gray-600 mb-3">
            This model is gated and requires access approval. You can paste the model's 
            config.json below to proceed with your {pendingAction?.type || 'analysis'}.
        </p>
        <textarea
            className="w-full h-64 p-2 border rounded-md font-mono text-sm"
            placeholder="Paste the model's config.json here..."
            value={configJson}
            onChange={(e) => setConfigJson(e.target.value)}
        />
        <div className="mt-3 flex gap-2">
            <button
                onClick={submitConfig}
                className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
                disabled={!configJson.trim()}
            >
                Submit Configuration
            </button>
            <button
                onClick={handleCancel}
                className="px-4 py-2 bg-gray-300 text-gray-700 rounded hover:bg-gray-400"
            >
                Cancel
            </button>
        </div>
    </div>
)}

// Config submission handler
const submitConfig = async () => {
    try {
        const config = JSON.parse(configJson);
        
        const response = await api.post('/models/config/submit', {
            model_uri: gatedModelId,
            config: config
        });
        
        if (response.data.success) {
            // Hide config input
            setShowConfigInput(false);
            setConfigJson('');
            
            // Continue with the pending action
            if (pendingAction) {
                switch (pendingAction.type) {
                    case 'calculate':
                        const calcResult = await api.post('/models/calculate', pendingAction.params);
                        setCalculationResult(calcResult.data);
                        break;
                    case 'analyze':
                        const analyzeResult = await api.post('/models/analyze', pendingAction.params);
                        setAnalysisResult(analyzeResult.data);
                        break;
                    case 'compare':
                        const compareResult = await api.post('/models/compare', pendingAction.params);
                        setComparisonResult(compareResult.data);
                        break;
                }
                setPendingAction(null);
            }
        } else {
            setError(response.data.message);
        }
    } catch (error) {
        if (error instanceof SyntaxError) {
            setError('Invalid JSON format. Please paste a valid config.json');
        } else {
            setError(error.response?.data?.message || 'Failed to submit configuration');
        }
    }
};
```

### 6. Backend Service Implementation

#### New API Endpoint Implementation

```python
@router.post("/config/submit", response_model=ConfigSubmitResponse)
async def submit_model_config(request: ConfigSubmitRequest):
    """
    Submit and validate a user-provided model configuration.
    """
    try:
        # Validate the config has minimum required fields
        required_fields = ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 'vocab_size']
        missing_fields = [field for field in required_fields if field not in request.config]
        
        if missing_fields:
            raise HTTPException(
                status_code=400,
                detail={
                    "success": False,
                    "message": f"Invalid configuration: missing required fields",
                    "error_code": "INVALID_CONFIG",
                    "required_fields": required_fields,
                    "missing_fields": missing_fields
                }
            )
        
        # Save the config
        success = model_manager.save_user_config(request.model_uri, request.config)
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to save configuration"
            )
        
        # Validate and analyze the config
        calc = ModelMemoryCalculator()
        model_type = calc.detect_model_type(request.config)
        attention_type = calc.detect_attention_type(request.config)
        
        # Calculate parameter count if not provided
        if 'num_parameters' not in request.config:
            param_count = calc._calculate_transformer_params(request.config)
        else:
            param_count = request.config['num_parameters']
        
        return ConfigSubmitResponse(
            success=True,
            message="Configuration saved and validated successfully",
            model_id=request.model_uri,
            validation={
                "valid": True,
                "model_type": model_type,
                "attention_type": attention_type,
                "parameter_count": param_count
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error submitting config for {request.model_uri}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )
```

#### ModelManager Updates

```python
class ModelManager:
    # ... existing methods ...
    
    def get_user_config(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get user-provided config for a model."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT config_json FROM user_model_configs WHERE model_id = ?",
                (model_id,)
            )
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None
    
    def save_user_config(self, model_id: str, config: Dict[str, Any]) -> bool:
        """Save a user-provided model configuration."""
        try:
            # Validate config has minimum required fields
            required_fields = ['hidden_size', 'num_hidden_layers', 'num_attention_heads']
            if not all(field in config for field in required_fields):
                raise ValueError("Configuration missing required fields")
            
            config_json = json.dumps(config)
            
            with self.get_connection() as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO user_model_configs 
                    (model_id, config_json, updated_at)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (model_id, config_json))
                conn.commit()
            
            # Also add to main models table if not exists
            self._ensure_model_exists(model_id, config)
            
            return True
        except Exception as e:
            logging.error(f"Failed to save user config for {model_id}: {e}")
            return False
    
    def _ensure_model_exists(self, model_id: str, config: Dict[str, Any]):
        """Ensure model exists in main models table."""
        # Extract model name from ID
        model_name = model_id.split('/')[-1] if '/' in model_id else model_id
        
        # Detect model characteristics
        calc = ModelMemoryCalculator()
        model_type = calc.detect_model_type(config)
        attention_type = calc.detect_attention_type(config)
        param_count = calc._calculate_transformer_params(config)
        
        # Insert if not exists
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO models 
                (model_id, model_name, source, config_json, model_type, attention_type, parameter_count)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id, 
                model_name, 
                'user_provided',
                json.dumps(config),
                model_type,
                attention_type,
                param_count
            ))
            conn.commit()
```

#### HuggingFaceConfigLoader Updates

```python
class HuggingFaceConfigLoader:
    # ... existing methods ...
    
    def get_model_config(
        self, 
        model_id: str,
        add_param_count: bool = True,
        check_user_configs: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch and prepare model config, checking user configs first.
        """
        # NEW: Check for user-provided config first
        if check_user_configs:
            from src.db import ModelManager
            mm = ModelManager()
            user_config = mm.get_user_config(model_id)
            if user_config:
                # Map to our format
                config = self.map_config_to_memory_format(user_config)
                config['_source'] = f"user_provided:{model_id}"
                return config
        
        # Original flow continues...
        hf_config = self.fetch_model_config(model_id)
        config = self.map_config_to_memory_format(hf_config)
        
        if add_param_count and 'num_parameters' not in config:
            param_count = self.extract_parameter_count(model_id, config)
            if param_count:
                config['num_parameters'] = param_count
        
        config['_source'] = f"huggingface:{model_id}"
        return config
```

### 7. Testing Strategy

#### Unit Tests
- Test user config validation
- Test database operations for user configs
- Test API endpoint responses with gated models
- Test config priority (user > database > huggingface)

#### Integration Tests
- Full flow from gated error to config submission
- Verify saved configs work with all endpoints
- Test error handling for invalid configs

#### Frontend Tests
- Test gated model detection
- Test config input UI appears/disappears correctly
- Test config submission flow

### 8. Migration Steps

1. **Database Migration**
   - Add new `user_model_configs` table
   - Update schema version

2. **Backend Updates** (in order)
   - Update ModelManager with new methods
   - Update HuggingFaceConfigLoader
   - Update API schemas
   - Implement new endpoints
   - Update existing endpoints

3. **Frontend Updates**
   - Add config input component
   - Update model validation flow
   - Add config submission logic

4. **Testing & Deployment**
   - Run all tests
   - Deploy backend first
   - Deploy frontend after backend is stable

### 9. Security Considerations

- **Input Validation**: Validate all user-provided JSON configs
- **Size Limits**: Limit config size to prevent DoS
- **Sanitization**: Ensure configs don't contain executable code
- **Rate Limiting**: Limit config submissions per IP

### 10. Future Enhancements

- Allow users to edit existing configs
- Add config verification status
- Share configs between users (with permission)
- Auto-detect config format and convert
- Config versioning for updates

## Summary

This implementation provides a seamless solution for handling gated models while maintaining backward compatibility. The key benefits are:

1. **User-Friendly**: Clear error messages and intuitive UI for config submission
2. **Persistent**: User configs are saved and reused automatically
3. **Compatible**: All existing APIs continue to work unchanged
4. **Secure**: Input validation prevents malicious configs
5. **Efficient**: Configs are cached to minimize database queries
6. **Seamless Flow**: Users get their results immediately after config submission

The implementation follows a phased approach, allowing for incremental deployment and testing. The system gracefully falls back to existing behavior when user configs are not available, ensuring no disruption to current functionality.

## Key Implementation Details

1. **Config Submission API** requires:
   - `model_uri`: The HuggingFace URI the user originally entered
   - `config`: The complete config.json content

2. **Frontend Flow**:
   - Stores the pending action when gated model is detected
   - After config submission, automatically continues with the stored action
   - No need for user to re-submit or re-click anything

3. **Backend Flow**:
   - Validates config has required fields
   - Calculates model characteristics (type, attention, parameters)
   - Returns validation info immediately for frontend to use
   - Saved configs are prioritized over HuggingFace fetches

This ensures the user experience remains smooth and uninterrupted, with gated models becoming as easy to use as any other model after the initial config submission. 