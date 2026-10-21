#!/usr/bin/env bash
# Plano B local da fila de certeza (as sessões do Colab caem antes de um GBDT
# terminar): roda aqui os GBDTs da H10 com hiperparâmetros congelados — usando a
# projeção PCA baixada do Colab (scripts/puxar_caches_colab.sh) — e o bootstrap
# de cada comparação assim que as duas predições existem.
# Resultado: runs/hm_clientes/h10_bootstrap_local.json
set -u
cd "$(dirname "$0")/.."
C=configs/hm_clientes.yaml
R=runs/hm_clientes
PY=".venv/bin/python -u"
LIM="scripts/rodar_limitado.sh -m 9G"
G="$LIM $PY pipeline/common/tarefa14_gbdt.py $C --avaliar_teste --seeds 0,1,2 --threads 10 --modelo lgbm --pca_k 32"
B="$LIM $PY pipeline/common/avaliacao.py $C --split test --B 300 --saida $R/h10_bootstrap_local.json"
T="churn proxima_online next_category ltv_90d"
espera() { until [ -f "$1" ]; do sleep 120; done; }
tem_comp() { python3 -c "import json,sys,os;f='$R/h10_bootstrap_local.json';r=json.load(open(f)) if os.path.exists(f) else [];sys.exit(0 if any(x['tarefa']=='$1' and x['a'].endswith('$2') and x['b'].endswith('$3') for x in r) else 1)"; }
comp() { tem_comp "$1" "$2" "$3" || $B --tarefa $1 --a "$2" --b "$3"; }
gb_ok() { [ -f $R/gbdt/$1/predicoes_test_ltv_90d_seed2.parquet ] && [ -f $R/gbdt/$1/predicoes_test_proxima_online_seed2.parquet ]; }

espera $R/gbdt/h10_trad_lgbm/predicoes_test_proxima_online_seed2.parquet   # tradicional (rodando à parte)
espera $R/caches/pca32_h3.npz
gb_ok h10_hibrido_lgbm || $G --features hibrido --emb fusao --pca_cache $R/caches/pca32_h3.npz --params_de hibrido_lgbm --saida h10_hibrido_lgbm
for t in $T; do comp $t gbdt/h10_trad_lgbm gbdt/h10_hibrido_lgbm; done; echo "[$(date)] híbrido ok"
gb_ok h10_emb_pca_lgbm || $G --features emb_pca --emb fusao --pca_cache $R/caches/pca32_h3.npz --params_de emb_pca_lgbm --saida h10_emb_pca_lgbm
for t in $T; do comp $t gbdt/h10_trad_lgbm gbdt/h10_emb_pca_lgbm; done; echo "[$(date)] só-embedding ok"
gb_ok h10_hibrido_lgbm_sem_cadastro || $G --features hibrido --emb fusao --sem_cadastro --pca_cache $R/caches/pca32_h3.npz --params_de hibrido_lgbm_sem_cadastro --saida h10_hibrido_lgbm_sem_cadastro
for t in $T; do comp $t gbdt/h10_hibrido_lgbm_sem_cadastro gbdt/h10_hibrido_lgbm; done; echo "[$(date)] cadastro no GBDT ok"
espera $R/caches/h10_fusao_d_completo_s2/predicoes_test.parquet
for t in $T; do comp $t gbdt/h10_trad_lgbm "caches/h10_fusao_d_completo_s*"; done; echo "[$(date)] tradicional x fusão ok"
espera $R/caches/pca32_final_s0.npz
gb_ok h10_final_hibrido_lgbm || $G --features hibrido --emb final_s0/embeddings --pca_cache $R/caches/pca32_final_s0.npz --params_de final_hibrido_lgbm --saida h10_final_hibrido_lgbm
for t in $T; do comp $t gbdt/h10_trad_lgbm gbdt/h10_final_hibrido_lgbm; done; echo "[$(date)] híbrido final ok"
echo "[$(date)] certeza local concluída"
