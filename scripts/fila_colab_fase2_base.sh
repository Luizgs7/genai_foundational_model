#!/usr/bin/env bash
# Base da fase 2 no Colab: extração dos embeddings do H3 (insumo de H1/H2) e
# a tarefa que faltou do híbrido com embedding bruto (LTV, H7 da fase 1).
cd /content/genai_model
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
C=configs/hm_clientes.yaml
R=runs/hm_clientes
[ -f $R/fusao/embeddings_eventos_index.parquet ] || python3 -u pipeline/common/tarefa13_extrair_embeddings.py $C --pretreino pretreino --saida fusao --dtype float16
echo "[$(date)] extração ok"
python3 -c "import json,sys;sys.exit(0 if 'ltv_90d' in json.load(open('$R/gbdt/hibrido_bruto_lgbm/relatorio.json'))['tarefas'] else 1)" 2>/dev/null || \
  python3 -u pipeline/common/tarefa14_gbdt.py $C --emb fusao --pca_k 32 --modelo lgbm --amostra_hpo 300000 --threads 12 \
    --features hibrido_bruto --saida hibrido_bruto_lgbm --tarefas ltv_90d --n_trials 12
echo "[$(date)] base fase 2 concluída"
