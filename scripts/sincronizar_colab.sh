#!/usr/bin/env bash
# Puxa do Colab, a cada N segundos, os artefatos PEQUENOS do run H&M
# (relatórios .json, métricas .csv, logs e checkpoints melhor.pt das
# ablações) para a máquina local. O runtime do Colab é efêmero: já se
# perderam resultados duas vezes quando o túnel caiu.
#
# Uso: scripts/sincronizar_colab.sh <host-do-tunel> [intervalo_s=300]
# (rodar em segundo plano; para com Ctrl+C / kill)
set -u
H="$1"; INT="${2:-300}"
cd "$(dirname "$0")/.."
while true; do
  timeout 2400 ssh -o ConnectTimeout=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$H" '[ -f /content/.ultimo_sync ] || touch -d @0 /content/.ultimo_sync; cd /content/genai_model && find runs/hm_clientes \( -name "*.json" -o -name "*.csv" -o -name "*.png" -o -path "*abl_*/pretreino/checkpoints/melhor.pt" -o -path "*final*/checkpoints/melhor.pt" -o -path "*h10_*/predicoes_test*.parquet" \) -size -800M -newer /content/.ultimo_sync 2>/dev/null | tar cf - -T - 2>/dev/null; touch /content/.ultimo_sync_novo' 2>/dev/null | tar xf - 2>/dev/null \
    && timeout 60 ssh -o ConnectTimeout=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$H" 'mv /content/.ultimo_sync_novo /content/.ultimo_sync' 2>/dev/null \
    && echo "[$(date +%H:%M)] sincronizado" || echo "[$(date +%H:%M)] falha de conexão"
  for f in /content/h4_h9.log /content/h6_h7.log /content/h8.log /content/fila.log /content/hibrido.log /content/extracao.log /content/h5_xgb_ltv.log /content/final.log /content/final_gbdt.log /content/h10_gpu.log /content/h10_cpu.log /content/h7_xgb.log /content/h7_resto.log /content/h10_xgb.log /content/h10_boot.log /content/certeza_gpu.log /content/certeza_cpu1.log /content/certeza_cpu2.log /content/certeza_boot.log /content/fila_certeza.log; do
    timeout 60 ssh -o ConnectTimeout=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$H" "cat $f 2>/dev/null" > "logs/colab_$(basename $f)" 2>/dev/null || true
  done
  sleep "$INT"
done
