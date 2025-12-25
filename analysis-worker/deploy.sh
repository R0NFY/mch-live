#!/bin/bash
# Automated deploy script for Yandex Serverless Container
# Usage: ./deploy.sh

set -e  # Exit on any error

# ============ CONFIGURATION ============
# Fill these in once, then just run ./deploy.sh
REGISTRY_ID="crpv9gnnri1vqg1cof2b"           # e.g., crp1234567890abcdef
CONTAINER_NAME="analysis-worker"     # e.g., analysis-worker
SERVICE_ACCOUNT_ID="aje7j75f1c9al9uusakq"  # e.g., aje1234567890abcdef
IMAGE_NAME="analysis-worker"
MEMORY="4GB"
TIMEOUT="3600s"
# ========================================

# Validate config
if [[ -z "$REGISTRY_ID" || -z "$CONTAINER_NAME" || -z "$SERVICE_ACCOUNT_ID" ]]; then
    echo "❌ Error: Missing required configuration."
    echo ""
    echo "Set these environment variables or edit this script:"
    echo "  export REGISTRY_ID=crp..."
    echo "  export CONTAINER_NAME=analysis-worker"
    echo "  export SERVICE_ACCOUNT_ID=aje..."
    exit 1
fi

IMAGE_URI="cr.yandex/${REGISTRY_ID}/${IMAGE_NAME}:latest"

echo "🔨 Building Docker image..."
docker build -t $IMAGE_NAME .

echo "🏷️  Tagging image..."
docker tag $IMAGE_NAME $IMAGE_URI

echo "📤 Pushing to Yandex Container Registry..."
docker push $IMAGE_URI

echo "🚀 Deploying new revision..."
yc serverless container revision deploy \
    --container-name $CONTAINER_NAME \
    --image $IMAGE_URI \
    --service-account-id $SERVICE_ACCOUNT_ID \
    --memory $MEMORY \
    --execution-timeout $TIMEOUT \
    --environment "$(cat .env.local 2>/dev/null | grep -v '^#' | xargs | tr ' ' ',' || echo '')" \
    --cores 1

echo "✅ Deploy complete!"
echo "   Image: $IMAGE_URI"
