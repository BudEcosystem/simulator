# Deployment Guide

This guide explains how to deploy the LLM Performance Simulator application in production environments.

## Deployment Options

### 1. Docker Deployment (Recommended)

Docker is the recommended deployment method for production environments because it:
- Encapsulates all dependencies
- Ensures consistency between development and production
- Simplifies scaling and management
- Provides better security through containerization

#### Simple Docker Deployment

```bash
# Build the Docker image
docker build -t budsimulator .

# Run the container
docker run -d \
  --name budsimulator \
  -p 8000:8000 \
  -p 3000:3000 \
  budsimulator
```

#### Docker Compose Deployment

For a more production-ready setup with separate services:

```bash
# Start all services
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Stop services
docker-compose -f docker-compose.prod.yml down
```

### 2. Domain-Specific Deployment

For deploying to a specific domain like `simulator.bud.studio`:

```bash
# Use the domain deployment script
./deploy-domain.sh
```

This will:
1. Check for existing Docker images
2. Build images if needed
3. Guide you through SSL certificate setup
4. Start services with domain-specific configuration

#### SSL Certificate Setup

For production deployments with HTTPS, you'll need SSL certificates. Here are two options:

1. **Let's Encrypt with Certbot** (recommended):
```bash
# Install certbot
sudo apt-get update
sudo apt-get install certbot

# Obtain certificates
sudo certbot certonly --standalone -d simulator.bud.studio

# Copy certificates to the ssl directory
sudo cp /etc/letsencrypt/live/simulator.bud.studio/fullchain.pem ssl/
sudo cp /etc/letsencrypt/live/simulator.bud.studio/privkey.pem ssl/
sudo chown $USER:$USER ssl/*.pem
```

2. **Copy existing certificates**:
```bash
# Place your certificates in the ssl directory
cp /path/to/your/fullchain.pem ssl/
cp /path/to/your/privkey.pem ssl/
```

### 3. Traditional Deployment with Process Manager

If you prefer not to use Docker, you can deploy using a process manager like PM2, but this requires more manual setup.

#### Backend Deployment with PM2

```bash
# Install PM2 globally
npm install -g pm2

# Navigate to the backend directory
cd BudSimulator

# Start the backend API
pm2 start run_api.py --name "budsimulator-backend" --interpreter python

# Save the PM2 configuration
pm2 save
```

#### Frontend Deployment

For the frontend, you would need to:
1. Build the React application
2. Serve it with a web server like Nginx

```bash
# Build the frontend
cd BudSimulator/frontend
npm run build

# Serve with a web server (example with serve)
npx serve -s build -l 3000
```

## Environment Configuration

### Database Configuration

The application uses SQLite by default. For production, you might want to:

1. Use a persistent volume for the database file
2. Consider migrating to PostgreSQL for better performance

### SSL/TLS Configuration

For production deployments, configure SSL/TLS in the nginx configuration:

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;
    
    ssl_certificate /etc/nginx/ssl/certificate.crt;
    ssl_certificate_key /etc/nginx/ssl/private.key;
    
    # ... rest of configuration
}
```

## Scaling Considerations

### Horizontal Scaling

For high-traffic deployments:
1. Use a load balancer in front of multiple instances
2. Consider using Redis for session storage
3. Use a shared database (PostgreSQL) instead of SQLite

### Resource Requirements

Minimum requirements per instance:
- CPU: 2 cores
- RAM: 4GB
- Disk: 10GB (for dependencies and data)

## Monitoring and Maintenance

### Health Checks

The Docker image includes a health check:
```bash
curl -f http://localhost:8000/api/health
```

### Logs

View Docker container logs:
```bash
docker logs budsimulator
```

Or with docker-compose:
```bash
docker-compose -f docker-compose.prod.yml logs -f
```

### Updates

To update the application:
```bash
# Pull the latest code
git pull

# Rebuild and restart containers
docker-compose -f docker-compose.prod.yml down
docker-compose -f docker-compose.prod.yml up --build -d
```

## Backup and Recovery

### Database Backup

Regularly backup the SQLite database:
```bash
# In the container
cp /app/BudSimulator/data/prepopulated.db /backup/prepopulated.db.backup

# Or from the host
docker exec budsimulator cp /app/BudSimulator/data/prepopulated.db /backup/prepopulated.db.backup
```

### Configuration Backup

Backup important configuration files:
- `.env` files
- Custom hardware configurations
- nginx configuration

## Troubleshooting

### Common Issues

1. **Port conflicts**: Ensure ports 8000 and 3000 are available
2. **Permission errors**: Check file permissions in volumes
3. **Database errors**: Ensure the database file is accessible

### Debugging

```bash
# Check running containers
docker ps

# View container logs
docker logs <container-id>

# Execute commands in container
docker exec -it <container-id> /bin/bash
```

## Security Considerations

1. Run containers as non-root users
2. Use environment variables for sensitive configuration
3. Regularly update base images
4. Implement proper firewall rules
5. Use HTTPS in production