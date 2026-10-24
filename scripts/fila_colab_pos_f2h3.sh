#!/usr/bin/env bash
# Espera a pontuação da F2-H3 terminar e então dispara, em paralelo, o
# ranqueamento da F2-H3 (CPU) e a fila de GPU da F2-H10.
cd /content/genai_model
until grep -q "^FIM" /content/f2h3.log; do sleep 60; done
while ps -eo args= | grep -q "^python3 -u pipeline/common/f2h3_ranquear.py"; do sleep 60; done
setsid nohup bash -c "([ -f runs/hm_clientes/f2h3/resultado_val.json ] || python3 -u pipeline/common/f2h3_ranquear.py configs/hm_clientes.yaml --fase val) && python3 -u pipeline/common/f2h3_ranquear.py configs/hm_clientes.yaml --fase test" > /content/f2h3_rank.log 2>&1 < /dev/null &
setsid nohup bash scripts/fila_colab_f2h10.sh > /content/f2h10.log 2>&1 < /dev/null &
echo "disparado $(date)"
