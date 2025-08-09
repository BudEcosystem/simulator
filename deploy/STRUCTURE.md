# Final Deployment Structure

## Overview
The deployment files are organized for clarity and maintainability, with deployment configuration in the `deploy/` folder and Docker build files in their required locations.

## Directory Structure

```
simulator/
│
├── deploy/                              # All deployment configuration
│   ├── deploy.sh                       # Main deployment script
│   ├── docker-compose.yml              # Service orchestration
│   ├── nginx.conf                      # HTTPS reverse proxy config
│   ├── ssl-manager.sh                  # SSL certificate management
│   ├── ssl/                           # SSL certificates
│   │   ├── fullchain.pem              # Certificate chain
│   │   └── privkey.pem                # Private key
│   ├── README.md                       # Deployment documentation
│   ├── DEPLOYMENT_PRODUCTION.md        # Detailed production guide
│   └── STRUCTURE.md                    # This file
│
├── Dockerfile                           # Backend container build (Python API)
│
├── BudSimulator/
│   ├── Dockerfile.frontend.prod        # Frontend production build (React)
│   └── nginx.conf                      # Frontend nginx config
│
├── deployment-backup/                   # Archived old deployment files
│   ├── docker-compose.yml              # Old dev compose
│   ├── docker-compose.prod.yml         # Old prod compose
│   ├── nginx.conf                      # Old nginx config
│   ├── build.sh                        # Old build script
│   ├── BudSimulator/Dockerfile         # Unused backend Dockerfile
│   └── BudSimulator/Dockerfile.frontend # Dev frontend Dockerfile
│
└── deploy.sh -> deploy/deploy.sh       # Symlink for convenience
```

## File Purposes

### In deploy/ folder
- **deploy.sh** - Orchestrates the entire deployment process
- **docker-compose.yml** - Defines and configures all services
- **nginx.conf** - Reverse proxy for HTTPS and routing
- **ssl/** - SSL certificates for HTTPS

### In root directory
- **Dockerfile** - Builds the backend container with Python/FastAPI
  - Must be in root to access both `BudSimulator/` and `llm-memory-calculator/`

### In BudSimulator/ folder
- **Dockerfile.frontend.prod** - Builds optimized React frontend
  - Must be here to access `BudSimulator/frontend/`
- **nginx.conf** - Config for nginx inside frontend container

## Why This Structure?

1. **Docker Build Context**: Docker can only access files in the build context (the directory specified in the build command). Since we build from the root, Dockerfiles need access to their respective source directories.

2. **Separation of Concerns**: 
   - Deployment configuration (compose, SSL, proxy) in `deploy/`
   - Build configuration (Dockerfiles) near their source code

3. **Clean Organization**: Only production files remain, development files are backed up

## Deployment Commands

```bash
# From root directory
./deploy.sh                    # Using symlink

# Or from deploy directory
cd deploy && ./deploy.sh

# Manual deployment
cd deploy
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f [service]

# Stop services
docker compose down
```

## Services

1. **budsimulator-backend** - FastAPI backend (port 8000 internal)
2. **budsimulator-frontend** - React app served by nginx (port 80 internal)
3. **budsimulator-nginx** - Reverse proxy (ports 80/443 external)

## Data Persistence

- Backend database: Docker volume `backend-data`
- SSL certificates: Mounted from `deploy/ssl/`