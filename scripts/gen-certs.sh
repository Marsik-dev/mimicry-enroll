#!/usr/bin/env bash
#
# Генерирует self-signed TLS сертификат для mimicry-enroll.
# В SAN добавляются: localhost, 127.0.0.1, LAN IP (для доступа с телефона).
#
# Usage:
#   ./scripts/gen-certs.sh              # автоопределение LAN IP
#   LAN_IP=192.168.1.50 ./scripts/gen-certs.sh
#   EXTRA_IPS=10.0.0.5,10.0.0.6 ./scripts/gen-certs.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CERT_DIR="$SCRIPT_DIR/../certs"
mkdir -p "$CERT_DIR"

# Авто-детект LAN IP
if [ -z "${LAN_IP:-}" ]; then
    LAN_IP="$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}' \
              || hostname -I 2>/dev/null | awk '{print $1}' \
              || echo '')"
fi

SAN="DNS:localhost,DNS:mimicry-enroll,IP:127.0.0.1"
if [ -n "$LAN_IP" ]; then
    SAN="$SAN,IP:$LAN_IP"
fi
if [ -n "${EXTRA_IPS:-}" ]; then
    for ip in $(echo "$EXTRA_IPS" | tr ',' ' '); do
        SAN="$SAN,IP:$ip"
    done
fi

echo "→ Генерирую сертификат с SAN: $SAN"

openssl req -x509 -newkey rsa:4096 -sha256 -days 365 -nodes \
    -keyout "$CERT_DIR/key.pem" \
    -out   "$CERT_DIR/cert.pem" \
    -subj "/CN=mimicry-enroll" \
    -addext "subjectAltName=$SAN" \
    2>/dev/null

chmod 600 "$CERT_DIR/key.pem"
chmod 644 "$CERT_DIR/cert.pem"

echo "✓ Готово:"
echo "  $CERT_DIR/cert.pem"
echo "  $CERT_DIR/key.pem"
echo ""
echo "Перезапусти сервер: docker compose restart app"
echo ""
echo "Откройте на телефоне:"
echo "  https://${LAN_IP:-<LAN-IP>}:8502"
echo ""
echo "В первый раз браузер покажет 'Соединение не защищено' —"
echo "нажмите «Дополнительно» → «Перейти на сайт (небезопасно)»."
