#!/usr/bin/env bash
# Traz do Colab, arquivo a arquivo, as predições de TESTE da H10 necessárias
# ao bootstrap que ainda falta (GBDT LightGBM, fusão completa e backbone final).
# Retomável: pula arquivo que já existe localmente com o mesmo tamanho, e grava
# em .parcial antes de renomear (uma queda não deixa arquivo truncado).
# Uso: scripts/puxar_predicoes_colab.sh <host> [intervalo_s=600]
set -u
H="$1"; INT="${2:-600}"
cd "$(dirname "$0")/.."
S="ssh -o ConnectTimeout=180 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR $H"
while true; do
  lista=$(timeout 300 $S 'cd /content/genai_model && find runs/hm_clientes/gbdt/h10_*lgbm* runs/hm_clientes/h10_fusao_d_completo_s* runs/hm_clientes/h10_final_fusao_s* -name "predicoes_test*.parquet" -mmin +2 -printf "%s %p\n" 2>/dev/null')
  while read -r tam arq; do
    [ -z "$arq" ] && continue
    [ -f "$arq" ] && [ "$(stat -c %s "$arq")" = "$tam" ] && continue
    mkdir -p "$(dirname "$arq")"
    if timeout 1800 $S "cat /content/genai_model/$arq" > "$arq.parcial" && [ "$(stat -c %s "$arq.parcial")" = "$tam" ]; then
      mv "$arq.parcial" "$arq"; echo "[$(date +%H:%M)] ok $arq ($tam bytes)"
    else
      rm -f "$arq.parcial"; echo "[$(date +%H:%M)] falha $arq"
    fi
  done <<< "$lista"
  sleep "$INT"
done
