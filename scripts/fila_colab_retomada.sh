#!/usr/bin/env bash
# Fila do Colab após a perda do runtime de 2026-10-18 (H3 e H4/H9 já estão
# concluídos e trazidos para a máquina local):
#   1) extração dos embeddings da H3 (GPU) — insumo da H6/H7;
#   2) em paralelo: H6/H7 + XGBoost LTV da H5 (CPU) e as 5 ablações restantes
#      da H8 (GPU). Cada etapa pula o que já tem relatório.
cd /content/genai_model
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
C=configs/hm_clientes.yaml
if [ ! -f runs/hm_clientes/fusao/embeddings_eventos_index.parquet ]; then
  echo "[$(date)] extraindo embeddings da H3"
  python3 -u pipeline/common/tarefa13_extrair_embeddings.py $C --pretreino pretreino --saida fusao --dtype float16 \
    >> /content/extracao.log 2>&1 || { echo "[$(date)] extração FALHOU"; exit 1; }
fi
echo "[$(date)] extração ok"
(
  THREADS=${THREADS:-6} bash scripts/rodar_h6_h7.sh >> /content/h6_h7.log 2>&1 && echo "[$(date)] H6/H7 ok" || echo "[$(date)] H6/H7 FALHOU"
  python3 -c "import json,sys;sys.exit(0 if 'ltv_90d' in json.load(open('runs/hm_clientes/gbdt/trad_xgb/relatorio.json'))['tarefas'] else 1)" 2>/dev/null \
    || python3 -u pipeline/common/tarefa14_gbdt.py $C --features trad --modelo xgb --tarefas ltv_90d --n_trials 25 --threads ${THREADS:-6} >> /content/h5_xgb_ltv.log 2>&1
  echo "[$(date)] H5 XGB LTV terminado ($?)"
) &
python3 -u pipeline/common/experimento_ablacao.py $C ffn_swiglu norm_rms drop_0 attn_gqa2 attn_mqa \
  >> /content/h8.log 2>&1 && echo "[$(date)] H8 ok" || echo "[$(date)] H8 FALHOU"
wait
echo "[$(date)] fila concluída"
