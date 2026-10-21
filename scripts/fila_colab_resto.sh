#!/usr/bin/env bash
# O que faltava da fila de certeza depois que o plano B local fechou o bootstrap
# de tradicional/híbrido/só-embedding/cadastro/fusão (h10_bootstrap_local.json):
#   1) híbrido com o backbone final na H10 + bootstrap contra o tradicional
#   2) variantes cortadas: embedding bruto multiclasse (H6) e híbrido bruto (H7)
cd /content/genai_model
C=configs/hm_clientes.yaml
R=runs/hm_clientes
B="python3 pipeline/common/avaliacao.py $C --split test --B 300 --saida $R/h10_bootstrap_v2.json"
python3 -u pipeline/common/tarefa14_gbdt.py $C --avaliar_teste --seeds 0,1,2 --threads 12 --pca_k 32 --modelo lgbm \
  --features hibrido --emb final_s0/embeddings --params_de final_hibrido_lgbm --saida h10_final_hibrido_lgbm
for t in churn proxima_online next_category ltv_90d; do $B --tarefa $t --a gbdt/h10_trad_lgbm --b gbdt/h10_final_hibrido_lgbm; done
echo "[$(date)] híbrido final ok"
V="python3 -u pipeline/common/tarefa14_gbdt.py $C --emb fusao --pca_k 32 --modelo lgbm --amostra_hpo 300000 --threads 12"
tem() { python3 -c "import json,sys;sys.exit(0 if '$2' in json.load(open('$R/gbdt/$1/relatorio.json'))['tarefas'] else 1)" 2>/dev/null; }
tem emb_bruto_lgbm next_category && tem emb_bruto_lgbm ltv_90d || $V --features emb --saida emb_bruto_lgbm --tarefas next_category,ltv_90d --n_trials 12
echo "[$(date)] emb bruto ok"
tem hibrido_bruto_lgbm proxima_online || $V --features hibrido_bruto --saida hibrido_bruto_lgbm --tarefas churn,proxima_online --n_trials 20
tem hibrido_bruto_lgbm ltv_90d || $V --features hibrido_bruto --saida hibrido_bruto_lgbm --tarefas next_category,ltv_90d --n_trials 12
echo "[$(date)] fila resto concluída"
