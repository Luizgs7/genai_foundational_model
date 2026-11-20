#!/usr/bin/env bash
# F3-H6, etapa A (corte atual, motor v2): os braços H que faltam nos 4 problemas e a clusterização com o v2.
# Desenho em PLANO_FASE3.md. A pontuação do v2 (GPU) roda em paralelo com os LightGBM (CPU).
cd /content/genai_model
source scripts/colab_persistencia.sh
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
C=configs/hm_clientes.yaml
C1=configs/hm_clientes_f3h1.yaml
R=runs/hm_clientes
B=runs/hm_clientes_f3h1
PY="python3 -u"
t() { echo "== $* $(date)"; }
( while sleep 600; do persistir; done ) & PERS=$!

# 1. recomendação: pontuação dos candidatos pelo v2 (GPU, em paralelo)
[ -f $R/f3h6/pontuar_v2.json ] || { t "pontuar v2"; $PY pipeline/common/f3h6_pontuar_v2.py $C > /content/f3h6_pontuar.log 2>&1; t "pontuar v2 fim"; } &
PONT=$!

# 2. H em churn e nas tarefas antigas: LightGBM com as 97 features e o PCA-32 do v2
G="$PY pipeline/common/tarefa14_gbdt.py"
[ -f $R/gbdt/f3h6_hibrido_v2_lgbm/relatorio.json ] || { t "H v2 tarefas antigas"; $G $C --tarefas churn,next_category,ltv_90d,proxima_online \
    --modelo lgbm --n_trials 40 --seeds 0,1,2 --threads 12 --avaliar_teste --pca_k 32 --features hibrido --emb v2_fusao \
    --pca_cache $R/caches/pca32_v2.npz --saida f3h6_hibrido_v2_lgbm; }
persistir
# 3. H em desconto (run da F3-H1, com o histórico de desconto nas features)
[ -f $B/gbdt/f3h6_hibrido_v2_lgbm/relatorio.json ] || { t "H v2 desconto"; $G $C1 --tarefas proxima_com_desconto \
    --modelo lgbm --n_trials 40 --seeds 0,1,2 --threads 12 --avaliar_teste --pca_k 32 --features hibrido --emb v2_fusao \
    --pca_cache $R/caches/pca32_v2_f3h1.npz --saida f3h6_hibrido_v2_lgbm; }
persistir

# 4. clusterização com o v2 (escolhas de forma na validação; teste uma vez)
for f in val test; do
  [ -f $R/f3h6_cluster/resultado_$f.json ] || { t "cluster $f"; $PY pipeline/common/f3h2_clusterizacao.py $C --fase $f \
      --emb_cache caches/pca32_v2.npz --saida f3h6_cluster; }
done
persistir

# 5. ranqueador híbrido com o v2 (espera a pontuação)
wait $PONT
for f in val test; do
  [ -f $R/f2h3/resultado_${f}_v2.json ] || { t "ranqueador v2 $f"; $PY pipeline/common/f2h3_ranquear.py $C --fase $f --rep v2; }
done

# 6. bootstrap por cliente: H − T (e E da F3-H5 como referência)
S=$R/f3h6/bootstrap_test.json; rm -f $S
A="$PY pipeline/common/avaliacao.py"
for tf in churn next_category ltv_90d proxima_online; do
  $A $C --tarefa $tf --split test --B 300 --saida $S --a gbdt/h10_trad_lgbm --b gbdt/f3h6_hibrido_v2_lgbm
done
$A $C1 --tarefa proxima_com_desconto --split test --B 300 --saida $S --a gbdt/f3h1_trad_lgbm --b gbdt/f3h6_hibrido_v2_lgbm
kill $PERS 2>/dev/null
persistir
echo "FIM $(date)"
