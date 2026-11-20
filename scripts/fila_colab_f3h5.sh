#!/usr/bin/env bash
# F3-H5 — embedding congelado do v2 com cabeça leve de TAREFA ÚNICA, contra o LightGBM de cada pergunta.
# Churn (família dos alvos do pré-treino) e canal e desconto (fora dos alvos). 3 sementes. Desenho em PLANO_FASE3.md.
cd /content/genai_model
source scripts/colab_persistencia.sh
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
C=configs/hm_clientes.yaml
R=runs/hm_clientes
B=runs/hm_clientes_f3h1
F="python3 -u pipeline/common/tarefa13_treinar_fusao.py"
t() { echo "== $* $(date)"; }
falta() { local d=$1 p=$2 S; for S in 0 1 2; do [ -f $d/${p}_s$S/predicoes_test.parquet ] || return 0; done; return 1; }

# embeddings do v2 (os de todas as âncoras não vão para o Drive; refazê-los leva alguns minutos)
if { falta $R f3h5_churn || falta $R f3h5_online || falta $B f3h5_desc; } && [ ! -f $R/v2_fusao/embeddings_eventos_index.parquet ]; then
  t "extracao v2"; python3 -u pipeline/common/v2_extrair.py $C --pretreino v2_pretreino --saida v2_fusao
fi
[ -e $B/v2_fusao ] || ln -s ../hm_clientes/v2_fusao $B/v2_fusao

for S in 0 1 2; do
  [ -f $R/f3h5_churn_s$S/predicoes_test.parquet ] || { t "churn s$S"; $F configs/hm_clientes_f3h5_churn_fusao.yaml --emb v2_fusao \
      --seed $S --saida f3h5_churn_s$S --dispositivo auto --avaliar_teste; }
  [ -f $R/f3h5_online_s$S/predicoes_test.parquet ] || { t "online s$S"; $F configs/hm_clientes_f3h5_online_fusao.yaml --emb v2_fusao \
      --seed $S --saida f3h5_online_s$S --dispositivo auto --avaliar_teste; }
  [ -f $B/f3h5_desc_s$S/predicoes_test.parquet ] || { t "desconto s$S"; $F configs/hm_clientes_f3h1_fusao.yaml --emb v2_fusao \
      --seed $S --saida f3h5_desc_s$S --dispositivo auto --avaliar_teste; }
  persistir
done

# bootstrap por cliente (E − T; a fusão conjunta da F3-H3 e a fusão v1 da F3-H1 como referência)
S=$R/f3h5/bootstrap_test.json; mkdir -p $R/f3h5; rm -f $S
t "bootstrap"
A="python3 -u pipeline/common/avaliacao.py"
$A $C --tarefa churn --split test --B 300 --saida $S --a gbdt/h10_trad_lgbm --b "f3h5_churn_s*" --b "v2_fusao_s*"
$A $C --tarefa proxima_online --split test --B 300 --saida $S --a gbdt/h10_trad_lgbm --b "f3h5_online_s*" --b "v2_fusao_s*"
$A configs/hm_clientes_f3h1.yaml --tarefa proxima_com_desconto --split test --B 300 --saida $S \
   --a gbdt/f3h1_trad_lgbm --b "f3h5_desc_s*" --b "f3h1_fusao_s*"
persistir
echo "FIM $(date)"
