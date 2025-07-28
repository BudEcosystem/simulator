# GitHub Copilot Instructions for BudSimulator

## Project Overview

BudSimulator is a comprehensive AI model benchmarking platform that helps users:
- Analyze LLM performance on various hardware
- Compare different AI models
- Get hardware recommendations
- Understand computational requirements

## Code Completion Guidelines

### Python Backend

When suggesting Python code:
- Import statements should follow the project's existing patterns
- Use type hints for function parameters and returns
- Follow the GenZ framework conventions for operators and systems
- Respect the existing class hierarchies

Example patterns:
```python
from typing import Dict, List, Optional
from BudSimulator.GenZ import System, ParallelismConfig
```

### React Frontend

When suggesting React code:
- Use TypeScript with proper type definitions
- Follow the existing component structure
- Implement loading and error states
- Use the established API service patterns

Example patterns:
```typescript
import { useState, useEffect } from 'react';
import { Hardware, Usecase } from '../types';
```

### API Endpoints

When suggesting API code:
- Follow FastAPI patterns with Pydantic models
- Include proper error handling
- Use consistent response formats
- Implement input validation

## Common Tasks

### Adding a Model
Suggest code that follows the pattern in `BudSimulator/GenZ/Models/Model_sets/`

### Creating API Endpoints
Follow the structure in `BudSimulator/apis/routers/`

### Frontend Components
Match the style in `BudSimulator/frontend/src/components/`

## Important Conventions

1. **Database IDs**: Always use `unique_id` for usecases, not numeric `id`
2. **Error Handling**: Wrap operations in try-except blocks
3. **Logging**: Use appropriate logging levels
4. **Testing**: Suggest test cases for new functionality

## Performance Considerations

- For large model computations, consider memory constraints
- Use efficient data structures
- Implement caching where appropriate
- Optimize database queries

## Security Notes

- Validate all user inputs
- Sanitize data before database operations
- Use environment variables for sensitive configuration
- Never commit API keys or secrets

## Testing Patterns

Suggest tests that:
- Cover edge cases
- Use pytest fixtures
- Follow existing test structure
- Include both unit and integration tests

## Documentation

When suggesting code:
- Include docstrings for functions
- Add inline comments for complex logic
- Update README files when adding features
- Follow the existing documentation style