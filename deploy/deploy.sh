#!/bin/bash

# Production Deployment script for simulator.bud.studio

DOMAIN="simulator.bud.studio"
PROJECT_DIR="/home/budadmin/simulator"
DEPLOY_DIR="$PROJECT_DIR/deploy"

echo "🚀 Deploying LLM Performance Simulator to $DOMAIN (Production)"

# Check if Docker is installed
if ! command -v docker &> /dev/null
then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker compose &> /dev/null
then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

# Build Docker images
echo "🏗️  Building Docker images..."
cd $PROJECT_DIR

# Build backend image (always rebuild to ensure latest fixes)
echo "📦 Building backend image..."
docker build -t budsimulator .

# Build production frontend image
echo "🎨 Building production frontend image..."
docker build -f BudSimulator/Dockerfile.frontend.prod -t budsimulator-frontend-prod .

# Check SSL certificates
echo "🔑 Checking SSL certificates..."
if [ ! -f "$DEPLOY_DIR/ssl/fullchain.pem" ] || [ ! -f "$DEPLOY_DIR/ssl/privkey.pem" ]; then
    echo "❌ SSL certificates not found in $DEPLOY_DIR/ssl/"
    echo "   Please obtain SSL certificates for $DOMAIN"
    echo "   Options:"
    echo "   1. Use Let's Encrypt with Certbot:"
    echo "      sudo certbot certonly --standalone -d $DOMAIN"
    echo "   2. Copy existing certificates to $DEPLOY_DIR/ssl/"
    echo "      - fullchain.pem (certificate chain)"
    echo "      - privkey.pem (private key)"
    exit 1
fi

echo "✅ SSL certificates found"

# Stop any existing containers
echo "⏹️  Stopping existing containers..."
cd $DEPLOY_DIR
docker compose down

# Start services with docker-compose
echo "🚀 Starting services..."
docker compose up -d

if [ $? -eq 0 ]; then
    echo "✅ Services started successfully!"
    echo ""
    echo "🌐 Your application will be available at:"
    echo "   https://$DOMAIN"
    echo ""
    echo "📋 To check service status:"
    echo "   docker compose ps"
    echo ""
    echo "📋 To view logs:"
    echo "   docker compose logs -f"
    echo ""
    echo "🛑 To stop services:"
    echo "   docker compose down"
else
    echo "❌ Failed to start services"
    exit 1
fi

# Wait a few seconds for services to start
echo "⏳ Waiting for services to initialize..."
sleep 10

# Check if services are responding
echo "🔍 Checking service health..."

# Check backend health
if curl -k -f https://$DOMAIN/api/health > /dev/null 2>&1; then
    echo "✅ Backend is responding"
else
    echo "⚠️  Backend health check failed"
fi

echo "🎉 Deployment completed!"
echo "   Access your application at: https://$DOMAIN"