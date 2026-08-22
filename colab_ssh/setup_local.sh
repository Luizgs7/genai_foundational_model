#!/usr/bin/env bash
# colab_ssh/setup_local.sh
#
# Prepara a máquina local para conectar via SSH na sessão compartilhada do
# Colab do grupo (túnel Cloudflare), sem precisar de sudo/root local.
#
# O que faz:
#   1. Baixa o binário do cloudflared (standalone, sem instalação de pacote)
#      em ~/.local/bin.
#   2. Gera uma chave SSH dedicada (~/.ssh/colab_gcm) — só para esse túnel,
#      não mexe nas suas outras chaves.
#   3. Adiciona um bloco `Host *.trycloudflare.com` no ~/.ssh/config, sem
#      duplicar se você rodar de novo.
#
# Uso:
#   bash colab_ssh/setup_local.sh
#
# Depois de rodar, ver colab_ssh/README.md para os próximos passos (mandar
# sua chave pública pro time e conectar pelo VSCode Remote-SSH).

set -euo pipefail

BIN_DIR="$HOME/.local/bin"
KEY_PATH="$HOME/.ssh/colab_gcm"
SSH_CONFIG="$HOME/.ssh/config"

mkdir -p "$BIN_DIR" "$HOME/.ssh"
chmod 700 "$HOME/.ssh"

echo "==> Baixando cloudflared..."
OS="$(uname -s)"
ARCH="$(uname -m)"
case "$OS-$ARCH" in
  Linux-x86_64)   CF_ASSET="cloudflared-linux-amd64" ;;
  Linux-aarch64)  CF_ASSET="cloudflared-linux-arm64" ;;
  Darwin-x86_64)  CF_ASSET="cloudflared-darwin-amd64.tgz" ;;
  Darwin-arm64)   CF_ASSET="cloudflared-darwin-arm64.tgz" ;;
  *)
    echo "SO/arquitetura não reconhecido automaticamente: $OS-$ARCH."
    echo "Baixe manualmente em https://github.com/cloudflare/cloudflared/releases,"
    echo "coloque o binário executável em $BIN_DIR/cloudflared e rode este script de novo."
    exit 1
    ;;
esac

URL="https://github.com/cloudflare/cloudflared/releases/latest/download/$CF_ASSET"
if [[ "$CF_ASSET" == *.tgz ]]; then
  curl -fsSL "$URL" -o /tmp/cloudflared.tgz
  tar -xzf /tmp/cloudflared.tgz -C "$BIN_DIR"
  rm -f /tmp/cloudflared.tgz
else
  curl -fsSL "$URL" -o "$BIN_DIR/cloudflared"
fi
chmod +x "$BIN_DIR/cloudflared"
CF_PATH="$(cd "$BIN_DIR" && pwd)/cloudflared"
echo "    cloudflared em $CF_PATH ($("$CF_PATH" --version))"

if [[ -f "$KEY_PATH" ]]; then
  echo "==> Chave $KEY_PATH já existe — não vou sobrescrever."
else
  echo "==> Gerando chave SSH dedicada ($KEY_PATH)..."
  ssh-keygen -t ed25519 -f "$KEY_PATH" -N "" -C "colab-team-$(whoami)" >/dev/null
fi
chmod 600 "$KEY_PATH"
chmod 644 "$KEY_PATH.pub"

touch "$SSH_CONFIG"
chmod 600 "$SSH_CONFIG"
if grep -q 'Host \*.trycloudflare.com' "$SSH_CONFIG" 2>/dev/null; then
  echo "==> ~/.ssh/config já tem o bloco *.trycloudflare.com — não vou duplicar."
else
  echo "==> Adicionando bloco ao ~/.ssh/config..."
  {
    echo ""
    echo "Host *.trycloudflare.com"
    echo "    HostName %h"
    echo "    User root"
    echo "    Port 22"
    echo "    IdentityFile $KEY_PATH"
    echo "    IdentitiesOnly yes"
    echo "    ProxyCommand $CF_PATH access ssh --hostname %h"
  } >> "$SSH_CONFIG"
fi

echo ""
echo "============================================================"
echo " Pronto! Sua CHAVE PÚBLICA (pode compartilhar, não é segredo):"
echo "============================================================"
cat "$KEY_PATH.pub"
echo "============================================================"
echo ""
echo "Próximo passo: adiciona essa linha em colab_ssh/authorized_keys"
echo "neste repo (commit/PR, ou peça pra alguém do time adicionar) —"
echo "ver colab_ssh/README.md."
