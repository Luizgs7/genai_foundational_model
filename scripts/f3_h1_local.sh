#!/usr/bin/env bash
# F3-H1 (propensão a desconto), parte local: coluna derivada, rótulos, features e
# os caminhos GBDT tradicional e híbrido, cronometrados. Mesmo orçamento de HPO
# para os dois (40 tentativas), 3 sementes, teste avaliado uma vez.
cd "$(dirname "$0")/.."
C=configs/hm_clientes_f3h1.yaml
B=runs/hm_clientes_f3h1
L="scripts/rodar_limitado.sh -m 8G"
PY=".venv/bin/python -u"
[ -f runs/hm_clientes/eventos_canonicos_f3h1.parquet ] || $L $PY pipeline/common/f3h1_desconto.py configs/hm_clientes.yaml
mkdir -p $B
for d in splits discretizacao vocabulario serializacao pretreino caches fusao; do
  [ -e $B/$d ] || [ ! -e runs/hm_clientes/$d ] || ln -s ../hm_clientes/$d $B/$d
done
ini=$(date +%s)
[ -f $B/rotulos_downstream/rotulos.parquet ] || $L $PY pipeline/common/tarefa8_rotulos.py $C
echo "TEMPO rotulos $(( $(date +%s) - ini )) s"; ini=$(date +%s)
[ -f $B/features_tradicionais/features.parquet ] || scripts/rodar_limitado.sh -m 10G $PY pipeline/common/features_tradicionais.py $C   # as features precisam de mais memória que o resto
echo "TEMPO features $(( $(date +%s) - ini )) s"
G="$L $PY pipeline/common/tarefa14_gbdt.py $C --tarefas proxima_com_desconto --modelo lgbm --n_trials 40 --seeds 0,1,2 --threads 10 --avaliar_teste --pca_k 32"
for cam in trad hibrido; do
  ini=$(date +%s)
  if [ $cam = trad ]; then $G --features trad --saida f3h1_trad_lgbm
  else $G --features hibrido --emb fusao --pca_cache runs/hm_clientes/caches/pca32_h3.npz --saida f3h1_hibrido_lgbm; fi
  echo "TEMPO $cam $(( $(date +%s) - ini )) s"
done
echo "FIM $(date)"
