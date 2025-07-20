#!/bin/bash

# SlackToDoc Production Deployment Script
# Deploys the complete SlackToDoc production system

set -e

echo "🚀 Starting SlackToDoc Production Deployment..."

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "❌ Docker daemon is not running. Please start Docker."
    exit 1
fi

# Create directories
mkdir -p logs backups ssl

# Check environment file
if [[ ! -f ".env.production" ]]; then
    echo "⚠️ Creating environment template..."
    cat > .env.production << EOF
ENVIRONMENT=production
LOG_LEVEL=INFO
JWT_SECRET=your-jwt-secret-here
SLACK_BOT_TOKEN=xoxb-your-token
SLACK_SIGNING_SECRET=your-secret
NOTION_TOKEN=secret_your-token
OPENAI_API_KEY=sk-your-key
POSTGRES_PASSWORD=secure-password
EOF
    echo "📝 Please edit .env.production with your configuration"
    exit 1
fi

# Build application
echo "🔨 Building application..."
docker-compose -f docker-compose.production.yml build

# Start services
echo "🚀 Starting services..."
docker-compose -f docker-compose.production.yml up -d

# Wait for health check
echo "⏳ Waiting for services to be ready..."
sleep 30

# Verify deployment
echo "✅ Verifying deployment..."
if curl -f http://localhost:8000/health &>/dev/null; then
    echo "✅ SlackToDoc is running successfully!"
    echo "📊 Health: http://localhost:8000/health"
    echo "📈 Metrics: http://localhost:8000/metrics"
    echo "🔍 Prometheus: http://localhost:9090"
    echo "📊 Grafana: http://localhost:3000"
else
    echo "❌ Health check failed"
    docker-compose -f docker-compose.production.yml logs
    exit 1
fi

echo "🎉 Deployment complete!"