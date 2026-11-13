#!/usr/bin/env bash
# F2-H11 no Colab (L4): TabICL e TabPFN-3.5 com contexto de N âncoras de treino da
# amostra (trad e híbrido). Espera a fila da F2-H10 liberar a GPU. TabPFN só roda
# com TABPFN_TOKEN em /content/tabpfn_token (licença aceita em ux.priorlabs.ai).
cd /content/genai_model
source scripts/colab_persistencia.sh
until grep -q "^FIM" /content/f2h10.log 2>/dev/null; do sleep 120; done
restaurar
R=runs/hm_clientes/gbdt
for M in tabicl tabpfn; do
  if [ $M = tabpfn ]; then
    [ -f /content/tabpfn_token ] || { echo "sem token do TabPFN: pulando"; continue; }
    export TABPFN_TOKEN=$(cat /content/tabpfn_token)
    # a chave sozinha não basta: a licença do TabPFN-3.5 precisa estar aceita na conta (ux.priorlabs.ai, aba Licenses)
    python3 -c "
import os, sys
from tabpfn import browser_auth as b
from tabpfn.settings import settings
lic = b._get_license_name('tabpfn_3_5')
sys.exit(0 if b.check_license_accepted(os.environ['TABPFN_TOKEN'], settings.tabpfn.auth_api_url, lic) else 1)" 2>/dev/null \
      || { echo "licença do TabPFN-3.5 não aceita na conta: pulando TabPFN (rodar depois)"; continue; }
  fi
  for H in "" "--hibrido"; do for N in 10000 50000 300000; do
    nome=f2h11_${M}_$([ -n "$H" ] && echo hibrido || echo trad)_n$N
    [ -f $R/$nome/relatorio.json ] && continue
    echo "== $nome $(date)"; python3 -u pipeline/common/f2h11_tabular.py configs/hm_clientes.yaml --modelo $M --n $N $H
    persistir
  done; done
done

# avaliação final no teste (uma vez), na amostra de clientes da F2-H10
A="python3 -u pipeline/common/f2h10_avaliar.py configs/hm_clientes.yaml"
[ -f runs/hm_clientes/f2h10/resultado_test.json ] || { echo "== avaliacao F2-H10 $(date)"; $A --saida f2h10/resultado_test.json \
    --braco T=gbdt/f2h10_trad --braco "E0=f2h10_fusao_e0_s*" --braco "E1=f2h10_e1_s*" --braco "E2=f2h10_fusao_e2_s*" \
    --braco "E3a0=f2h10_fusao_e3a0_s*" --braco "E3a=f2h10_fusao_e3a_s*" --ref E0 --ref T; }
BR="--braco T=gbdt/f2h10_trad"
for d in $R/f2h11_*; do [ -f $d/relatorio.json ] && BR="$BR --braco $(basename $d | sed 's/^f2h11_//')=gbdt/$(basename $d)"; done
REFS=$(for d in $R/f2h11_lgbm_*; do echo -n " --ref $(basename $d | sed 's/^f2h11_//')"; done)
ls -d $R/f2h11_tabpfn_* >/dev/null 2>&1 || [ -f /content/sem_tabpfn ] || { persistir; echo "F2-H11: aguardando TabPFN para avaliar"; echo "FIM $(date)"; exit 0; }
[ -f runs/hm_clientes/f2h11/resultado_test.json ] || { echo "== avaliacao F2-H11 $(date)"; $A --saida f2h11/resultado_test.json --B 300 $BR --ref T $REFS; }
persistir
echo "FIM $(date)"
