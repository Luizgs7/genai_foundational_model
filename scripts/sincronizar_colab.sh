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
  timeout 1200 ssh -o ConnectTimeout=180 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$H" '[ -f /content/.ultimo_sync ] || touch -d @0 /content/.ultimo_sync; cd /content/genai_model && find runs/hm_clientes runs/hm_clientes_f3h1/ \( -name "*.json" -o -path "*f3h1_fusao_s*/predicoes_*.parquet" -o -path "*v2_fusao_s*/predicoes_*.parquet" -o -path "*v2_pretreino/melhor.pt" -o -name "*.csv" -o -name "*.png" -o -path "*abl_*/pretreino/checkpoints/melhor.pt" -o -path "*final*/checkpoints/melhor.pt" -o -path "*f2h10_e2/pretreino/checkpoints/melhor.pt" -o -path "*f2h10_e3a/adaptador/*" \) -size -200M -newer /content/.ultimo_sync 2>/dev/null | tar cf - -T - 2>/dev/null; touch /content/.ultimo_sync_novo' 2>/dev/null | tar xf - 2>/dev/null \
    && timeout 300 ssh -o ConnectTimeout=180 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$H" 'mv /content/.ultimo_sync_novo /content/.ultimo_sync' 2>/dev/null \
    && echo "[$(date +%H:%M)] sincronizado" || echo "[$(date +%H:%M)] falha de conexão"
  for f in /content/f3.log /content/f2h11_tabicl_hibrido_n50000.log /content/f2h10.log /content/f2h11.log; do
    # grava num temporário: se a conexão falhar, a última cópia boa do log fica intacta
    d="logs/colab_$(basename $f)"
    timeout 300 ssh -o ConnectTimeout=180 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$H" "cat $f 2>/dev/null" > "$d.tmp" 2>/dev/null && [ -s "$d.tmp" ] && mv "$d.tmp" "$d" || rm -f "$d.tmp"
  done
  sleep "$INT"
done
