#!/usr/bin/env bash
# F2-H11 (TabICL e TabPFN-3.5) e E3b da F2-H10, reordenados em 14/11/2026: o Colab
# encerra a sessão com 24 h, então rodam primeiro os contextos pequenos dos dois
# modelos (onde o critério principal se decide) e por último os de 199 mil âncoras.
# Idempotente e com cópia de cada etapa no Drive (scripts/colab_persistencia.sh):
# num runtime novo, basta relançar.
#
# Uso: setsid nohup scripts/fila_colab_f2h11_v2.sh > /content/f2h11.log 2>&1 &
cd /content/genai_model
source scripts/colab_persistencia.sh
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
C=configs/hm_clientes.yaml
R=runs/hm_clientes
G=$R/gbdt
t() { echo "== $* $(date)"; }
restaurar

# espera um job de GPU já em andamento (ex.: o TabICL de 199 mil âncoras lançado pela fila anterior)
while ps -eo args= | awk '/pipeline\/common\/f2h11_tabular.py/ && !/awk/' | grep -q .; do sleep 60; done

tabpfn_ok() {
  [ -f /content/tabpfn_token ] || { echo "sem token do TabPFN"; return 1; }
  export TABPFN_TOKEN=$(cat /content/tabpfn_token)
  python3 -c "
import os, sys
from tabpfn import browser_auth as b
from tabpfn.settings import settings
lic = b._get_license_name('tabpfn_3_5')
sys.exit(0 if b.check_license_accepted(os.environ['TABPFN_TOKEN'], settings.tabpfn.auth_api_url, lic) else 1)" 2>/dev/null \
    || { echo "licença do TabPFN-3.5 não aceita na conta"; return 1; }
}
rodar() {  # modelo, "" ou --hibrido, N
  local nome=f2h11_$1_$([ -n "$2" ] && echo hibrido || echo trad)_n$3
  [ -f $G/$nome/relatorio.json ] && return 0
  if [ $1 = tabpfn ]; then tabpfn_ok || { echo "pulando $nome"; return 0; }; fi
  t $nome; python3 -u pipeline/common/f2h11_tabular.py $C --modelo $1 --n $3 $2
  persistir
}

rodar tabicl "" 10000;  rodar tabicl "" 50000;  rodar tabicl "" 300000     # já feitos ou em andamento
rodar tabpfn "" 10000;  rodar tabpfn "" 50000
rodar tabicl --hibrido 10000;  rodar tabicl --hibrido 50000
rodar tabpfn --hibrido 10000;  rodar tabpfn --hibrido 50000
rodar tabpfn "" 300000
rodar tabicl --hibrido 300000
rodar tabpfn --hibrido 300000

# avaliação final da F2-H11 (uma vez), só com os 12 braços prontos ou com /content/fechar_f2h11
A="python3 -u pipeline/common/f2h10_avaliar.py $C"
prontos=$(ls $G/f2h11_tabicl_*/relatorio.json $G/f2h11_tabpfn_*/relatorio.json 2>/dev/null | wc -l)
if [ ! -f $R/f2h11/resultado_test.json ] && { [ "$prontos" -ge 12 ] || [ -f /content/fechar_f2h11 ]; }; then
  BR="--braco T=gbdt/f2h10_trad"
  for d in $G/f2h11_*; do [ -f $d/relatorio.json ] && BR="$BR --braco $(basename $d | sed 's/^f2h11_//')=gbdt/$(basename $d)"; done
  REFS=$(for d in $G/f2h11_lgbm_*; do echo -n " --ref $(basename $d | sed 's/^f2h11_//')"; done)
  t "avaliacao F2-H11 ($prontos braços de modelo tabular)"; $A --saida f2h11/resultado_test.json --B 300 $BR --ref T $REFS
  persistir
else
  echo "F2-H11: $prontos de 12 braços prontos; avaliação adiada"
fi

# E3b da F2-H10: Qwen3.5-0.8B com LoRA por próximo token e embedding pela média dos estados.
# Roda porque o E3a ficou a menos de 0,005 do E0 (−0,0009), como previa o desenho.
QB=Qwen/Qwen3.5-0.8B-Base
F="python3 -u pipeline/common/tarefa13_treinar_fusao.py"
if [ ! -f $R/f2h10_e3b/relatorio_lm.json ]; then
  t "E3b ensaio"
  if python3 -u pipeline/common/f2h10_llm.py $C lm --modelo $QB --saida f2h10_e3b_ensaio --lote 4 --max_tokens 128 --passos 2 --max_clientes 200 --estado_a_cada 0; then
    rm -rf $R/f2h10_e3b_ensaio
    t "E3b treino"
    ( while sleep 300; do persistir; done ) & PERS=$!
    python3 -u pipeline/common/f2h10_llm.py $C lm --modelo $QB --saida f2h10_e3b --lote 32 --max_tokens 384 --passos 1000 --retomar
    kill $PERS 2>/dev/null
  else
    echo "E3b: o ensaio falhou; treino não iniciado"
  fi
fi
if [ -f $R/f2h10_e3b/relatorio_lm.json ]; then
  [ -f $R/f2h10_e3b/embeddings_eventos_index.parquet ] || { t "E3b extracao"; python3 -u pipeline/common/f2h10_llm.py $C extrair --modelo $QB --causal_lm \
      --adaptador f2h10_e3b/adaptador --pooling media --sem_instrucao --saida f2h10_e3b; }
  persistir
  for S in 0 1 2; do
    [ -f $R/f2h10_fusao_e3b_s$S/predicoes_test.parquet ] || { t "fusao e3b s$S"; $F $C --emb f2h10_e3b --seed $S --min_passos 4000 \
        --saida f2h10_fusao_e3b_s$S --dispositivo auto --avaliar_teste; }
  done
  persistir
  [ -f $R/f2h10/resultado_test_e3b.json ] || { t "avaliacao E3b"; $A --saida f2h10/resultado_test_e3b.json \
      --braco T=gbdt/f2h10_trad --braco "E0=f2h10_fusao_e0_s*" --braco "E3a0=f2h10_fusao_e3a0_s*" --braco "E3a=f2h10_fusao_e3a_s*" \
      --braco "E3b=f2h10_fusao_e3b_s*" --ref E0 --ref T --ref E3a; }
  persistir
fi
echo "FIM $(date)"
