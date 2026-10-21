# encerra as filas de certeza em andamento (casando argumentos exatos, sem pkill -f) e lança fila_colab_resto.sh
cd /content/genai_model
ps -eo pid=,args= | awk '$2=="bash" && $3=="scripts/fila_colab_certeza.sh" {print $1}' | xargs -r kill
sleep 2
ps -eo pid=,args= | awk '$2=="python3" && ($4=="pipeline/common/tarefa14_gbdt.py" || $3=="pipeline/common/tarefa14_gbdt.py" || $3=="pipeline/common/avaliacao.py") {print $1}' | xargs -r kill
sleep 3
setsid nohup bash scripts/fila_colab_resto.sh >> /content/certeza_cpu.log 2>&1 < /dev/null &
echo trocado
