#!/usr/bin/env bash
# F2-H1 (eficiência de rótulos), parte local: GBDT tradicional e híbrido
# treinados com 1%, 5%, 10% e 25% dos CLIENTES de treino rotulados (amostras
# aninhadas, 3 sementes de amostra — as mesmas da fusão no Colab). Validação e
# teste completos. HPO com orçamento da H7 (20 tentativas binárias, 12 multiclasse).
cd "$(dirname "$0")/.."
C=configs/hm_clientes.yaml
until grep -q "TEMPO hibrido" logs/f2_h6_local.log 2>/dev/null; do sleep 120; done
G="scripts/rodar_limitado.sh -m 9G .venv/bin/python -u pipeline/common/tarefa14_gbdt.py $C --modelo lgbm --seeds 0 --threads 10 --avaliar_teste --pca_k 32"
completo() { python3 -c "import json,sys;r=json.load(open('runs/hm_clientes/gbdt/$1/relatorio.json'));sys.exit(0 if len(r['tarefas'])==4 else 1)" 2>/dev/null; }
for FR in 0.01 0.05 0.10 0.25; do
  for S in 0 1 2; do
    for F in trad hibrido; do
      out=f2h1_${F}_f${FR}_s$S
      completo $out && continue
      extra=""; [ $F = hibrido ] && extra="--emb fusao --pca_cache runs/hm_clientes/caches/pca32_h3.npz"
      $G --features $F $extra --frac_treino $FR --semente_amostra $S --saida $out --tarefas churn,proxima_online --n_trials 20
      $G --features $F $extra --frac_treino $FR --semente_amostra $S --saida $out --tarefas next_category,ltv_90d --n_trials 12
      echo "[$(date)] $out ok"
    done
  done
done
echo "[$(date)] F2-H1 local concluída"
