#!/usr/bin/env bash
# F3-H6, etapa A, complemento: H do v2 nas tarefas antigas que faltam (LTV e próxima categoria; a próxima
# online já está no Drive) e o bootstrap delas contra o LightGBM tradicional. Retoma por tarefa.
cd /content/genai_model
source scripts/colab_persistencia.sh
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
C=configs/hm_clientes.yaml
R=runs/hm_clientes
PY="python3 -u"
t() { echo "== $* $(date)"; }
( while sleep 600; do persistir; done ) & PERS=$!
[ -f $R/v2_fusao/embeddings_eventos_index.parquet ] || { t "extracao v2"; $PY pipeline/common/v2_extrair.py $C --pretreino v2_pretreino --saida v2_fusao; }
G="$PY pipeline/common/tarefa14_gbdt.py"
GH="--modelo lgbm --n_trials 40 --seeds 0,1,2 --threads 12 --avaliar_teste --pca_k 32 --features hibrido --emb v2_fusao"
S=$R/f3h6/bootstrap_complemento.json
for tf in proxima_online ltv_90d next_category; do
  [ -f $R/gbdt/f3h6_hibrido_v2_lgbm/predicoes_test_${tf}_seed2.parquet ] || { t "H v2 $tf"; $G $C --tarefas $tf $GH \
      --pca_cache $R/caches/pca32_v2.npz --saida f3h6_hibrido_v2_lgbm; }
  persistir
  t "bootstrap $tf"
  python3 - "$S" "$tf" <<'EOF' || $PY pipeline/common/avaliacao.py $C --tarefa $tf --split test --B 300 --saida $S --a gbdt/h10_trad_lgbm --b gbdt/f3h6_hibrido_v2_lgbm
import json, os, sys
s, t = sys.argv[1:]
sys.exit(0 if os.path.exists(s) and any(r["tarefa"] == t for r in json.load(open(s))) else 1)
EOF
done
kill $PERS 2>/dev/null
persistir
echo "FIM $(date)"
