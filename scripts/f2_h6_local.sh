#!/usr/bin/env bash
# F2-H6 (custo de uma tarefa nova), parte local: caminhos GBDT tradicional e
# híbrido para a tarefa proxima_cesta_alta, cronometrados. Mesmo orçamento de
# HPO para os dois (40 tentativas), 3 sementes, teste avaliado uma vez.
cd "$(dirname "$0")/.."
C=configs/hm_clientes_f2h6.yaml
G="scripts/rodar_limitado.sh -m 9G .venv/bin/python -u pipeline/common/tarefa14_gbdt.py $C --tarefas proxima_cesta_alta --modelo lgbm --n_trials 40 --seeds 0,1,2 --threads 10 --avaliar_teste --pca_k 32"
for cam in trad hibrido; do
  ini=$(date +%s)
  if [ $cam = trad ]; then $G --features trad --saida f2h6_trad_lgbm
  else $G --features hibrido --emb fusao --pca_cache runs/hm_clientes/caches/pca32_h3.npz --saida f2h6_hibrido_lgbm; fi
  echo "TEMPO $cam $(( $(date +%s) - ini )) s"
done
