# API Configuration Guide

This guide explains how to configure the API backend URL for the BudSimulator frontend.

## Configuration Options

The frontend now supports configurable API endpoints through environment variables.

### Development Mode

In development mode (when running `npm start`), the application uses the proxy configuration from `package.json` by default. This means:
- API calls to `/api/*` are automatically proxied to `http://localhost:8000`
- No additional configuration is needed for local development

### Production Mode

For production deployments, you can configure the API URL using the `REACT_APP_API_URL` environment variable.

## Configuration Methods

### 1. Using .env File (Recommended)

Create or edit the `.env` file in the frontend directory:

```bash
# For default development setup (uses proxy)
REACT_APP_API_URL=

# For production
REACT_APP_API_URL=https://api.budsimulator.com

# For custom backend URL
REACT_APP_API_URL=http://localhost:3001
```

### 2. Using Environment Variables

Set the environment variable when starting the application:

```bash
# Linux/macOS
REACT_APP_API_URL=https://api.budsimulator.com npm start

# Windows
set REACT_APP_API_URL=https://api.budsimulator.com && npm start
```

### 3. Using .env Files for Different Environments

Create environment-specific files:
- `.env.development` - for development
- `.env.production` - for production
- `.env.local` - for local overrides (not committed to git)

## How It Works

1. The configuration is managed in `src/config/api.ts`
2. If `REACT_APP_API_URL` is set, it will be used as the base URL
3. If not set or empty, the app uses relative paths (`/api`), which work with:
   - The proxy in development
   - Same-domain API in production

## Examples

### Local Development (Default)
No configuration needed. The proxy in `package.json` handles routing.

### Production with Same Domain
```
REACT_APP_API_URL=
```
API calls will go to `/api` on the same domain.

### Production with Different Domain
```
REACT_APP_API_URL=https://api.budsimulator.com
```
API calls will go to `https://api.budsimulator.com/api`

### Custom Local Backend
```
REACT_APP_API_URL=http://localhost:3001
```
API calls will go to `http://localhost:3001/api`

## Troubleshooting

1. **CORS Issues**: If you're getting CORS errors, ensure your backend is configured to accept requests from your frontend domain.

2. **Proxy Not Working**: The proxy only works in development mode (`npm start`). It doesn't work with production builds.

3. **Environment Variables Not Loading**: 
   - Ensure variable names start with `REACT_APP_`
   - Restart the development server after changing `.env` files
   - Check that `.env` file is in the frontend root directory

## Files Modified

The following files have been updated to support configurable API URLs:
- `src/config/api.ts` - Central API configuration
- `src/services/hardwareAPI.ts` - Hardware API service
- `src/services/usecaseAPI.ts` - Usecase API service
- `src/components/ModelDetails.tsx` - Model details component
- `src/AIMemoryCalculator.tsx` - Main calculator component