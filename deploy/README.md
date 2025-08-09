# Deployment Directory

This directory contains all files necessary for deploying the LLM Performance Simulator to production.

## Files

- **deploy.sh** - Main deployment script that builds images and starts services
- **docker-compose.yml** - Docker services orchestration configuration
- **nginx.conf** - Nginx reverse proxy configuration for HTTPS and routing
- **ssl-manager.sh** - SSL certificate management utility
- **ssl/** - SSL certificates directory (fullchain.pem and privkey.pem)
- **DEPLOYMENT_PRODUCTION.md** - Comprehensive deployment documentation

## Quick Start

From the project root directory:

```bash
# Deploy the application
./deploy.sh

# Or from within the deploy directory
cd deploy
./deploy.sh
```

## Services

The deployment runs three Docker containers:
1. **budsimulator-backend** - FastAPI backend service
2. **budsimulator-frontend** - React frontend served by nginx
3. **budsimulator-nginx** - Reverse proxy with SSL termination

## Management Commands

```bash
# Check service status
docker compose ps

# View logs
docker compose logs -f [service-name]

# Stop all services
cd deploy && docker compose down

# Restart a specific service
docker compose restart [service-name]
```

## SSL Certificates

Place your SSL certificates in the `ssl/` subdirectory:
- `ssl/fullchain.pem` - Certificate chain
- `ssl/privkey.pem` - Private key

You can obtain free SSL certificates from Let's Encrypt:
```bash
sudo certbot certonly --standalone -d simulator.bud.studio
```

## Notes

### Docker Build Files (Must remain in their locations for build context)
- **Root Directory:**
  - `Dockerfile` - Backend container build configuration
  
- **BudSimulator Directory:**
  - `BudSimulator/Dockerfile.frontend.prod` - Production frontend build
  - `BudSimulator/nginx.conf` - Nginx configuration for frontend container

### Why These Files Can't Move
Docker requires build files to have access to the build context. The backend Dockerfile needs access to both `BudSimulator/` and `llm-memory-calculator/` directories, so it must be in the root. Similarly, the frontend Dockerfile needs access to `BudSimulator/frontend/`, so it must be in the BudSimulator directory.

### Deployment Configuration
All deployment-specific configuration (docker-compose, nginx proxy, SSL) is contained in this `deploy/` directory for organization.