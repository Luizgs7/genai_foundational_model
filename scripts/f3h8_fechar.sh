#!/usr/bin/env bash
# F3-H8 (base sintética grande, janela 8192): fecha a hipótese depois do pré-treino da fila f3b.
#   forma: cobertura do contexto e custo de um passo de treino em função do tamanho do pedido
#   estrutura: ARI dos segmentos (rfm, trad, E) contra o gabarito; recomendação direta contra R1/POP
#              e recuperação das regras de compra conjunta plantadas
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
source scripts/colab_persistencia.sh
C=configs/beleza_b2b_grande.yaml; R=runs/beleza_b2b_grande
t() { echo "== $* $(date)"; }
while [ ! -f $R/emb_v2/embeddings_eventos_index.parquet ]; do sleep 60; done
mkdir -p $R/f3h8
[ -f $R/f3h8/custo.json ] || { t "custo e cobertura"; python3 -u pipeline/common/f3h8_custo.py $C --pretreino pre_v2 --saida custo; }
t "segmentos"; python3 -u pipeline/common/f3h7_segmentos.py $C --espacos rfm trad E:emb_v2 --saida segmentos
D=2025-01-06
[ -f $R/rec_v2/recomendacoes_$D.parquet ] || { t "recomendacao v2"; python3 -u pipeline/common/v2_extrair.py $C --dados v2 --pretreino pre_v2 \
    --recomendar $D --k 12 --saida rec_v2; }
t "recomendacao: medida"; python3 -u pipeline/common/f3h7_recomendacao.py $C --data $D --modelo v2=rec_v2 --saida recomendacao
persistir
echo "FIM $(date)"
