#!/usr/bin/env bash
set -euo pipefail

# Run on a Linux EC2 instance after installing Docker and cloning the repository.
# Restrict inbound port 8501 to the intended audience in the EC2 security group.
cd "$(dirname "$0")/.."
docker build -f deployment/Dockerfile -t xrayguide-ai .
docker stop xrayguide-ai 2>/dev/null || true
docker rm xrayguide-ai 2>/dev/null || true

docker_args=(--restart unless-stopped --name xrayguide-ai -p 8501:8501)
if [[ -n "${S3_BUCKET_NAME:-}" ]]; then
  docker_args+=(-e S3_BUCKET_NAME)
fi
if [[ -n "${AWS_REGION:-}" ]]; then
  docker_args+=(-e AWS_REGION)
fi

docker run -d "${docker_args[@]}" xrayguide-ai
docker ps --filter name=xrayguide-ai
