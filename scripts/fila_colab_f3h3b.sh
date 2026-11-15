#!/usr/bin/env bash
# F3-H3b (PLANO_FASE3.md): v2 com alvos de composição por atributo, na H&M. Roda em paralelo com a
# fila f3b; a trava impede que o pré-treino da base grande (janela 8192) dispute a GPU com este.
cd /content/genai_model
source scripts/colab_persistencia.sh
restaurar
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
C=configs/hm_clientes.yaml; R=runs/hm_clientes
t() { echo "== $* $(date)"; }
touch /content/trava_gpu
trap 'rm -f /content/trava_gpu' EXIT
if [ ! -f $R/v2b_pretreino/v2_relatorio.json ] || ! python3 -c "import json,sys; sys.exit(0 if len(json.load(open('$R/v2b_pretreino/v2_relatorio.json'))['historico']) >= 8 else 1)"; then
  t "F3-H3b pre-treino"
  ( while sleep 600; do persistir; done ) & PERS=$!
  python3 -u pipeline/common/v2_pretreino.py $C --saida v2b_pretreino --epocas 8 --alvos_atributos --retomar
  kill $PERS 2>/dev/null
fi
rm -f /content/trava_gpu; persistir
[ -f $R/v2b_fusao/embeddings_eventos_index.parquet ] || { t "F3-H3b extracao"; python3 -u pipeline/common/v2_extrair.py $C --pretreino v2b_pretreino --saida v2b_fusao; }
for S in 0 1 2; do
  [ -f $R/v2b_fusao_s$S/predicoes_test.parquet ] || { t "F3-H3b fusao s$S"; python3 -u pipeline/common/tarefa13_treinar_fusao.py $C --emb v2b_fusao --seed $S \
      --saida v2b_fusao_s$S --dispositivo auto --avaliar_teste; }
done
persistir
S=$R/f3h3/bootstrap_v2b.json; rm -f $S
for tt in churn next_category ltv_90d proxima_online; do
  python3 -u pipeline/common/avaliacao.py $C --tarefa $tt --split test --B 300 --saida $S --a "caches/h10_fusao_d_completo_s*" --b "v2b_fusao_s*" --b "v2_fusao_s*"
done 2>&1 | grep -E "auc|accuracy"
persistir
echo "FIM $(date)"
