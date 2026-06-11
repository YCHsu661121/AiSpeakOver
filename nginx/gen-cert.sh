#!/bin/sh
# Generate a self-signed certificate if not already present.
# Called once at container start by the nginx Docker entrypoint override.
set -e

CERT_DIR=/etc/nginx/certs
KEY="$CERT_DIR/server.key"
CRT="$CERT_DIR/server.crt"

if [ -f "$KEY" ] && [ -f "$CRT" ]; then
  echo "Certificate already exists, skipping generation."
  exit 0
fi

mkdir -p "$CERT_DIR"
openssl req -x509 -nodes -days 3650 \
  -newkey rsa:2048 \
  -keyout "$KEY" \
  -out    "$CRT" \
  -subj   "/CN=ai/O=AiSpeakOver" \
  -addext "subjectAltName=DNS:ai,DNS:localhost,IP:127.0.0.1"

echo "Self-signed certificate generated at $CRT"
