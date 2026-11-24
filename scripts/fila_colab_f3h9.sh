#!/usr/bin/env bash
# F3-H9 — ajuste fino conjunto do backbone v2 com a cabeça DCNv2, 5 tarefas da H&M, 3 sementes.
# Desenho e critério em PLANO_FASE3.md (registrados antes de rodar). Idempotente.
cd /content/genai_model
source scripts/colab_persistencia.sh
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
C=configs/hm_clientes.yaml
C1=configs/hm_clientes_f3h1.yaml
R=runs/hm_clientes
B=runs/hm_clientes_f3h1
t() { echo "== $* $(date)"; }

# lista de âncoras: o índice da extração dos vetores do v2 (refazê-la leva poucos minutos)
[ -f $R/v2_fusao/embeddings_eventos_index.parquet ] || { t "extracao v2"; python3 -u pipeline/common/v2_extrair.py $C \
    --pretreino v2_pretreino --saida v2_fusao; }
mkdir -p $B
for d in splits discretizacao vocabulario serializacao pretreino caches fusao v2 v2_pretreino v2_fusao; do
  [ -e $B/$d ] || ln -s ../hm_clientes/$d $B/$d
done

for S in 0 1 2; do
  if [ ! -f $B/f3h9_ft_s$S/predicoes_test.parquet ]; then
    t "ajuste conjunto s$S"
    APAGAR_CHECKPOINT=1 python3 -u pipeline/common/v2_ajuste_conjunto.py $C1 --emb v2_fusao --v2 v2 --pretreino v2_pretreino \
        --saida f3h9_ft_s$S --seed $S --epocas 3 --avaliar_teste || exit 1
  fi
  ln -sfn ../hm_clientes_f3h1/f3h9_ft_s$S $R/f3h9_ft_s$S
  persistir
done

# bootstrap por cliente: E ajustado contra T (critério principal) e contra o vetor congelado (secundário)
S=$B/f3h9/bootstrap_test.json; mkdir -p $B/f3h9; rm -f $S
t "bootstrap"
A="python3 -u pipeline/common/avaliacao.py"
for tf in churn proxima_online next_category ltv_90d; do
  $A $C --tarefa $tf --split test --B 300 --saida $S --a gbdt/h10_trad_lgbm --b "f3h9_ft_s*" --b "v2_fusao_s*"
  $A $C --tarefa $tf --split test --B 300 --saida $S --a "v2_fusao_s*" --b "f3h9_ft_s*"
done
$A $C1 --tarefa proxima_com_desconto --split test --B 300 --saida $S --a gbdt/f3h1_trad_lgbm --b "f3h9_ft_s*" --b "f3h5_desc_s*"
$A $C1 --tarefa proxima_com_desconto --split test --B 300 --saida $S --a "f3h5_desc_s*" --b "f3h9_ft_s*"
persistir
echo "FIM $(date)"
