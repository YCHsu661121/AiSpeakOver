#!/usr/bin/env bash
# rebuild.sh — 完整重建並啟動所有服務（含 NeMo）
set -e

cd "$(dirname "$0")"

echo "==> 停止所有容器..."
docker compose --profile nemo down --remove-orphans

echo "==> 重新 build 並啟動（含 NeMo）..."
docker compose --profile nemo up -d --build

echo "==> 等待服務就緒..."
for i in $(seq 1 30); do
  STATUS=$(docker inspect --format='{{.State.Health.Status}}' aispeakover-aispeakover-1 2>/dev/null || echo "starting")
  if [ "$STATUS" = "healthy" ]; then
    echo "==> aispeakover: healthy"
    break
  fi
  echo "    aispeakover: $STATUS ($i/30)..."
  sleep 3
done

echo ""
docker compose --profile nemo ps
echo ""
echo "==> 完成。開啟 https://localhost:4445"
