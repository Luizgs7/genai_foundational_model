#!/usr/bin/env bash
# Vigia um log no Colab e termina quando (a) o padrão aparece no log ou
# (b) a conexão falha 3 vezes seguidas — o túnel caiu ou o runtime morreu.
# Em 13/11/2026 um monitor que só procurava erros no log não percebeu a queda
# do túnel; por isso a falha de conexão também encerra a vigília.
#
# Uso: scripts/vigiar_colab.sh <host-do-tunel> <log-remoto> <regex> [intervalo_s=300]
H="$1"; LOG="$2"; PADRAO="$3"; INT="${4:-300}"
O="-o ConnectTimeout=60 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
falhas=0
while true; do
  if saida=$(timeout 120 ssh $O "$H" "grep -E '$PADRAO' $LOG | tail -5; echo __OK__" 2>/dev/null) && echo "$saida" | grep -q __OK__; then
    falhas=0
    if echo "$saida" | grep -v __OK__ | grep -q .; then
      echo "PADRÃO ENCONTRADO ($(date +%H:%M)):"; echo "$saida" | grep -v __OK__; break
    fi
  else
    falhas=$((falhas + 1)); echo "falha de conexão $falhas ($(date +%H:%M))"
    [ $falhas -ge 3 ] && { echo "CONEXÃO PERDIDA ($(date +%H:%M)): túnel caiu ou runtime morreu"; break; }
  fi
  sleep "$INT"
done
timeout 120 ssh $O "$H" "grep -E '^==|Traceback|Killed|^FIM' $LOG | tail -15; tail -3 $LOG | cut -c1-300" 2>/dev/null
