#!/usr/bin/env bash
# Baixa do Colab, em ordem de prioridade, os caches compactos gerados por
# pipeline/common/gerar_caches_download.py. Arquivo a arquivo, com verificação
# de tamanho e gravação em .parcial (uma queda nunca deixa arquivo truncado).
# Uso: scripts/puxar_caches_colab.sh <host>
set -u
H="$1"; cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=200 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR $H"
ARQS="runs/hm_clientes/caches/pca32_h3.npz
runs/hm_clientes/caches/h10_fusao_d_completo_s0/predicoes_test.parquet
runs/hm_clientes/caches/h10_fusao_d_completo_s1/predicoes_test.parquet
runs/hm_clientes/caches/h10_fusao_d_completo_s2/predicoes_test.parquet
runs/hm_clientes/caches/pca32_final_s0.npz"
for arq in $ARQS; do
  while true; do
    [ -f "$arq" ] && { echo "[$(date +%H:%M)] já existe $arq"; break; }
    tam=$(timeout 400 $S "test -f /content/genai_model/$arq.pronto && stat -c %s /content/genai_model/$arq" 2>/dev/null)
    if [ -z "$tam" ]; then sleep 300; continue; fi
    mkdir -p "$(dirname "$arq")"
    if timeout 7200 $S "cat /content/genai_model/$arq" > "$arq.parcial" && [ "$(stat -c %s "$arq.parcial")" = "$tam" ]; then
      mv "$arq.parcial" "$arq"; echo "[$(date +%H:%M)] ok $arq ($tam bytes)"; break
    fi
    rm -f "$arq.parcial"; echo "[$(date +%H:%M)] falha $arq — tentando de novo"; sleep 60
  done
done
echo "[$(date +%H:%M)] caches baixados"
