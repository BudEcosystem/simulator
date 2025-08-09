# Production Deployment Guide

## Overview
This guide covers the production deployment of the LLM Performance Simulator at https://simulator.bud.studio

## Required Files

The deployment files are organized in the `deploy/` folder:

```
simulator/
├── deploy/                      # All deployment files
│   ├── deploy.sh               # Main deployment script
│   ├── docker-compose.yml      # Docker services configuration
│   ├── nginx.conf              # Nginx reverse proxy configuration
│   ├── ssl-manager.sh          # SSL certificate management script
│   └── ssl/                    # SSL certificates
│       ├── fullchain.pem       # SSL certificate chain
│       └── privkey.pem         # SSL private key
├── Dockerfile                   # Backend container build
└── BudSimulator/
    ├── Dockerfile.frontend.prod # Production frontend build
    └── nginx.conf              # Frontend nginx configuration
```

## Prerequisites

1. **Server Requirements**
   - Ubuntu/Debian Linux server
   - Docker and Docker Compose installed
   - Domain name pointing to server IP
   - Ports 80 and 443 open

2. **SSL Certificates**
   - Place SSL certificates in `deploy/ssl/` directory
   - Files needed: `fullchain.pem` and `privkey.pem`
   - Can use Let's Encrypt with: `sudo certbot certonly --standalone -d simulator.bud.studio`

## Deployment Steps

### 1. Initial Setup
```bash
# Clone repository
git clone <repository-url>
cd simulator

# Ensure SSL certificates are in place
ls -la deploy/ssl/
```

### 2. Build Docker Images
```bash
# Build backend image (if not exists)
docker build -t budsimulator .

# Build frontend production image
docker build -f BudSimulator/Dockerfile.frontend.prod -t budsimulator-frontend-prod .
```

### 3. Deploy Application
```bash
# Run the deployment script
./deploy/deploy.sh

# Or manually with docker-compose from deploy folder
cd deploy
docker compose up -d
```

### 4. Verify Deployment
```bash
# Check container status
docker compose ps

# View logs
docker compose logs -f

# Test endpoints
curl -k https://simulator.bud.studio/api/health
curl -k https://simulator.bud.studio
```

## Service Architecture

### Services
1. **Backend** (budsimulator-backend)
   - FastAPI application on port 8000
   - Handles API requests and model calculations
   - SQLite database for data persistence

2. **Frontend** (budsimulator-frontend)
   - React application served by nginx
   - Static files optimized for production
   - Communicates with backend via API

3. **Nginx** (budsimulator-nginx)
   - Reverse proxy and SSL termination
   - Routes traffic to frontend and backend
   - Handles HTTPS redirect and security headers

### Network Configuration
- All services communicate on internal `app-network`
- Only nginx exposes ports 80/443 externally
- Backend API accessible at `/api/*` path

## Common Operations

### View Logs
```bash
docker compose logs -f [service-name]
```

### Restart Services
```bash
docker compose restart [service-name]
```

### Stop Services
```bash
docker compose down
```

### Update Application
```bash
# Pull latest code
git pull

# Rebuild images
docker build -t budsimulator .
docker build -f BudSimulator/Dockerfile.frontend.prod -t budsimulator-frontend-prod .

# Redeploy
./deploy.sh
```

### SSL Certificate Renewal
```bash
# Renew certificates
sudo certbot renew

# Copy to project directory
cp /etc/letsencrypt/live/simulator.bud.studio/fullchain.pem ssl/
cp /etc/letsencrypt/live/simulator.bud.studio/privkey.pem ssl/

# Restart nginx
docker compose restart nginx
```

## Troubleshooting

### Backend Not Responding
1. Check logs: `docker compose logs backend`
2. Verify database exists: `docker exec budsimulator-backend ls -la /app/BudSimulator/data/`
3. Test from inside container: `docker exec budsimulator-backend curl http://localhost:8000/api/health`

### Frontend Not Loading
1. Check nginx logs: `docker compose logs nginx`
2. Verify build succeeded: `docker images | grep frontend`
3. Check nginx configuration syntax

### SSL Issues
1. Verify certificates exist: `ls -la ssl/`
2. Check certificate validity: `openssl x509 -in ssl/fullchain.pem -text -noout`
3. Ensure nginx can read certificates

## Environment Variables

Configure in `docker-compose.yml`:

### Backend
- `BACKEND_PORT`: API port (default: 8000)
- `DATABASE_URL`: Database connection string

### Frontend
- `REACT_APP_API_URL`: Backend API URL

## Security Considerations

1. **SSL/TLS**: Always use HTTPS in production
2. **Headers**: Security headers configured in nginx.conf
3. **Firewall**: Only expose necessary ports (80, 443)
4. **Updates**: Keep Docker images and dependencies updated
5. **Secrets**: Never commit SSL certificates or sensitive data

## Monitoring

### Health Checks
- Backend: `https://simulator.bud.studio/api/health`
- Frontend: `https://simulator.bud.studio`

### Resource Usage
```bash
docker stats
```

### Disk Usage
```bash
docker system df
```

## Backup

### Database Backup
```bash
docker exec budsimulator-backend cp /app/BudSimulator/data/prepopulated.db /app/BudSimulator/data/backup-$(date +%Y%m%d).db
docker cp budsimulator-backend:/app/BudSimulator/data/backup-*.db ./backups/
```

## Support

For issues or questions:
- Check application logs first
- Review this documentation
- Contact system administrator