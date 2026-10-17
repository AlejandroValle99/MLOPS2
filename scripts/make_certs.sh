#!/usr/bin/env bash
# A throwaway development CA, the model server's certificate, and one client certificate
# (for mutual TLS, which is optional). certs/ is gitignored: never commit what this writes.
#
#   ./scripts/make_certs.sh            # writes certs/
#   ./scripts/make_certs.sh <dir>      # writes <dir> (the tests use a temporary one)
#
# Every name a client uses to reach the model server must be in the server certificate's
# subjectAltName: model-server inside compose, localhost from the laptop and the healthcheck.
set -euo pipefail
out="${1:-certs}"
mkdir -p "$out" && cd "$out"
if [[ -f ca.crt && -z "${FORCE:-}" ]]; then
  echo "$out/ca.crt already exists; FORCE=1 $0 to replace every certificate" >&2
  exit 0
fi
days=365
openssl req -x509 -newkey rsa:2048 -nodes -days "$days" -subj "/CN=stroke-dev-ca" \
  -keyout ca.key -out ca.crt 2>/dev/null
openssl req -newkey rsa:2048 -nodes -subj "/CN=model-server" \
  -keyout server.key -out server.csr 2>/dev/null
printf "subjectAltName=DNS:model-server,DNS:localhost,IP:127.0.0.1\n" > server.ext
openssl x509 -req -in server.csr -CA ca.crt -CAkey ca.key -CAcreateserial -days "$days" \
  -extfile server.ext -out server.crt 2>/dev/null
openssl req -newkey rsa:2048 -nodes -subj "/CN=stroke-client" \
  -keyout client.key -out client.csr 2>/dev/null
openssl x509 -req -in client.csr -CA ca.crt -CAkey ca.key -CAcreateserial -days "$days" \
  -out client.crt 2>/dev/null
rm -f server.csr client.csr server.ext ca.srl
# The CA key signs certificates: only its owner reads it. The services run as a non-root
# user that isn't you, and a bind mount keeps your file mode, so the leaf keys must be
# readable to it. Acceptable for a dev CA that lives a year and never leaves this machine.
chmod 600 ca.key
chmod 644 server.key client.key
echo "wrote $out/: ca.crt, server.crt/.key (model-server, localhost), client.crt/.key"
