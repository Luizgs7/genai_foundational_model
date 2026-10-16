#!/usr/bin/env bash
# Roda no Colab (CPU, 52 GB de RAM) depois da extração da H4:
#   H6 — embedding do modelo fundacional como feature no GBDT
#   H7 — híbrido (features tradicionais + embedding)
#   H9 — híbrido sem cadastro
# 1) sondas com parâmetros fixos escolhem o k do PCA (16/32/64) pela AUC de
#    churn na validação (tarefa principal); 2) HPO completa nas variantes.
# Embedding bruto (384 dims) e híbridos têm HPO menor (20/12 tentativas,
# amostra de 300 mil) — custo por tentativa ~5x o das features tradicionais.
set -euo pipefail
C=configs/hm_clientes.yaml
G="python3 -u pipeline/common/tarefa14_gbdt.py $C --emb fusao --threads ${THREADS:-8}"
R=runs/hm_clientes/gbdt

for K in 16 32 64; do
  [ -f $R/sonda_pca$K/relatorio.json ] || $G --features emb_pca --pca_k $K --modelo lgbm --n_trials 0 --seeds 0 --saida sonda_pca$K
done
[ -f $R/sonda_bruto/relatorio.json ] || $G --features emb --modelo lgbm --n_trials 0 --seeds 0 --saida sonda_bruto

K=$(python3 - <<'PY'
import json
r = {k: json.load(open(f"runs/hm_clientes/gbdt/sonda_pca{k}/relatorio.json"))["tarefas"]["churn"]["val_resumo"]["auc"]["media"] for k in (16, 32, 64)}
print(max(r, key=r.get))
PY
)
echo "k escolhido pela sonda (AUC churn val): $K"

roda() {  # $1=features $2=modelo $3=saida  [extras...]
  local f=$1 m=$2 s=$3; shift 3
  [ -f $R/$s/relatorio.json ] && python3 -c "import json,sys;r=json.load(open('$R/$s/relatorio.json'));sys.exit(0 if len(r['tarefas'])==4 else 1)" && return
  $G --features $f --modelo $m --pca_k $K --saida $s --amostra_hpo 300000 --tarefas churn,proxima_online --n_trials 20 "$@"
  $G --features $f --modelo $m --pca_k $K --saida $s --amostra_hpo 300000 --tarefas next_category,ltv_90d --n_trials 12 "$@"
}
roda trad lgbm trad_lgbm_orcamento_h7   # mesmo orçamento de HPO dos híbridos (comparação justa na H7)
roda emb_pca lgbm emb_pca_lgbm          # H6
roda emb lgbm emb_bruto_lgbm            # H6
roda hibrido lgbm hibrido_lgbm          # H7
roda hibrido_bruto lgbm hibrido_bruto_lgbm   # H7
roda hibrido xgb hibrido_xgb            # H7 (robustez ao algoritmo)
roda hibrido lgbm hibrido_lgbm_sem_cadastro --sem_cadastro   # H9
echo "H6/H7 concluídas (k=$K)"
