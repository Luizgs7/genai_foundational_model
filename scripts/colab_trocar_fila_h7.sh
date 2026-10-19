# interrompe a fila antiga da H6/H7 (casando os argumentos exatos, sem pkill -f) e lança rodar_h7_resto.sh
cd /content/genai_model
ps -eo pid=,args= | awk '$2=="bash" && ($3=="scripts/fila_colab_retomada.sh" || $3=="scripts/rodar_h6_h7.sh") {print $1}' | xargs -r kill
ps -eo pid=,args= | awk '$2=="python3" && /--saida emb_bruto_lgbm/ {print $1}' | xargs -r kill
sleep 3
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
setsid nohup bash scripts/rodar_h7_resto.sh >> /content/h7_resto.log 2>&1 < /dev/null &
