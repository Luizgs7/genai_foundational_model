#!/usr/bin/env bash
# Prepara a base sintética de beleza B2B de um tamanho para o motor v2 e a bateria
# da F3-H7/F3-H8: gera os dados (determinístico), adapter, splits, discretização,
# rótulos, features tradicionais (com as fontes extras) e as serializações do v2:
# todas as fontes e a retirada de uma por vez (navegação, crédito, incentivo e o
# campo de desconto das linhas). Idempotente.
#
# Uso: scripts/preparar_beleza.sh pequeno|grande [janela] [so_completo]
#   janela: sobrepõe a do config (a F3-H8 usa uma janela que cabe o maior cliente inteiro)
#   so_completo: só a serialização com todas as fontes (sem as variantes da ablação)
set -e
cd "$(dirname "$0")/.."
T=$1; J=${2:-}; SO=${3:-}
C=configs/beleza_b2b_$T.yaml
R=runs/beleza_b2b_$T
PY="${PY:-python3} -u"
[ -f $C ] || sed "s/beleza_b2b_pequeno/beleza_b2b_$T/g; s#dados_beleza_b2b/pequeno#dados_beleza_b2b/$T#g; s/tamanho pequeno/tamanho $T/" \
    configs/beleza_b2b_pequeno.yaml > $C
[ -f dados_beleza_b2b/$T/gabarito/clientes_latentes.parquet ] || $PY gerar_dados_sinteticos_beleza_b2b.py --tamanho $T
[ -f $R/eventos_canonicos.parquet ] || $PY adapters/adapter_beleza_b2b.py $T
[ -f $R/splits/splits.parquet ] || $PY pipeline/common/tarefa7_splits.py $C
[ -f $R/discretizacao/discretizado.parquet ] || $PY pipeline/common/tarefa4_discretizacao.py $C
[ -f $R/rotulos_downstream/rotulos.parquet ] || $PY pipeline/common/tarefa8_rotulos.py $C
[ -f $R/features_tradicionais/features.parquet ] || $PY pipeline/common/features_tradicionais.py $C
JA=${J:+--janela $J}
S="$PY pipeline/common/v2_serializacao.py $C $JA"
[ -f $R/v2/meta.json ] || $S --dados v2
[ "$SO" = so_completo ] || {
[ -f $R/v2_sem_navegacao/meta.json ] || $S --dados v2_sem_navegacao --sem_fonte navegacao
[ -f $R/v2_sem_credito/meta.json ] || $S --dados v2_sem_credito --sem_fonte credito
[ -f $R/v2_sem_incentivo/meta.json ] || $S --dados v2_sem_incentivo --sem_fonte incentivo
[ -f $R/v2_sem_desconto/meta.json ] || $S --dados v2_sem_desconto --sem_campo pedido.desconto
}
python3 -c "import json; m=json.load(open('$R/v2/meta.json')); print('$T:', m['n_entidades'], 'entidades,', m['n_posicoes'], 'posições,', m['n_linhas'], 'linhas; janela', m['janela'])"
