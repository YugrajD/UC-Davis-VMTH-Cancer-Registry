#!/usr/bin/env bash
# Build and push the backend and/or ml-worker images to ECR.
#
# Usage: scripts/push-images.sh [backend|ml-worker|all] [env]
#   env defaults to "prod" (the CDK envName; repos are cancer-registry-<env>-<name>).
#
# The ECR repos are created by the foundation stack, so this works before the
# app stack is deployed. Uses the current AWS credentials/region
# (AWS_PROFILE, AWS_REGION or the CLI default). Tags :<git-sha> and :latest.
set -euo pipefail

WHAT="${1:-all}"
ENV_NAME="${2:-prod}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-$(aws configure get region || true)}}"
REGION="${REGION:-us-east-1}"

cd "$(dirname "$0")/.."
ACCOUNT="$(aws sts get-caller-identity --query Account --output text)"
REGISTRY="${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com"
SHA="$(git rev-parse --short HEAD)"

aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$REGISTRY"

push() {
  local name="$1" dockerfile="$2" context="$3"
  local repo="${REGISTRY}/cancer-registry-${ENV_NAME}-${name}"
  echo "==> ${name}: building ${repo}:${SHA}"
  # Fargate tasks are x86_64 by default; force it so Apple Silicon builds work.
  docker buildx build --platform linux/amd64 --provenance=false \
    -f "$dockerfile" -t "${repo}:${SHA}" -t "${repo}:latest" --push "$context"
}

case "$WHAT" in
  backend)   push backend backend/Dockerfile backend ;;
  ml-worker) push ml-worker ml-worker/Dockerfile.batch . ;;
  all)       push backend backend/Dockerfile backend
             push ml-worker ml-worker/Dockerfile.batch . ;;
  *) echo "usage: $0 [backend|ml-worker|all] [env]" >&2; exit 1 ;;
esac
