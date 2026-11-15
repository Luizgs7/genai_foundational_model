#!/usr/bin/env bash
# Fila do Colab de 15/11/2026: fecha a etapa 0 e começa a etapa 2 da fase 3.
#   1. F2-H11: refaz o TabICL híbrido de 50 mil (falhou no início; log guardado) e
#      fecha a avaliação com os braços prontos. Os TabICL de 199 mil âncoras ficam
#      fora: morreram duas vezes por falta de RAM (52 GB), um limite prático.
#   2. F3-H1, braço E: embeddings do backbone H3 em todas as âncoras e a fusão de
#      tarefa única para proxima_com_desconto (3 sementes, teste uma vez).
#   3. F3-H3: pré-treino do motor v2 na H&M, extração nas âncoras do v1 e a mesma
#      fusão da H4 (3 sementes).
#   4. E3b da F2-H10 (Qwen gerativo com LoRA), por último: é a etapa mais longa.
#   5. TabPFN de 199 mil âncoras, se sobrar sessão.
# Idempotente e com cópia de cada etapa no Drive: num runtime novo, basta relançar.
#
# Uso: setsid nohup scripts/fila_colab_f3.sh > /content/f3.log 2>&1 &
cd /content/genai_model
source scripts/colab_persistencia.sh
export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH:-}
C=configs/hm_clientes.yaml
R=runs/hm_clientes
G=$R/gbdt
F="python3 -u pipeline/common/tarefa13_treinar_fusao.py"
A="python3 -u pipeline/common/f2h10_avaliar.py $C"
t() { echo "== $* $(date)"; }
restaurar

# ---- 1. F2-H11 ----
nome=f2h11_tabicl_hibrido_n50000
if [ ! -f $G/$nome/relatorio.json ]; then
  t $nome
  python3 -u pipeline/common/f2h11_tabular.py $C --modelo tabicl --n 50000 --hibrido > /content/$nome.log 2>&1 \
    || { echo "$nome falhou; fim do log:"; tail -30 /content/$nome.log; }
  persistir
fi
if [ ! -f $R/f2h11/resultado_test.json ]; then
  prontos=$(ls $G/f2h11_tabicl_*/relatorio.json $G/f2h11_tabpfn_*/relatorio.json 2>/dev/null | wc -l)
  BR="--braco T=gbdt/f2h10_trad"
  for d in $G/f2h11_tabicl_* $G/f2h11_tabpfn_* $G/f2h11_lgbm_*; do
    [ -f $d/relatorio.json ] && BR="$BR --braco $(basename $d | sed 's/^f2h11_//')=gbdt/$(basename $d)"
  done
  REFS=$(for d in $G/f2h11_lgbm_*; do [ -f $d/relatorio.json ] && echo -n " --ref $(basename $d | sed 's/^f2h11_//')"; done)
  t "avaliacao F2-H11 ($prontos braços de modelo tabular)"; $A --saida f2h11/resultado_test.json --B 300 $BR --ref T $REFS
  persistir
fi

# ---- 2. F3-H1, braço E ----
[ -f $R/fusao/embeddings_eventos_index.parquet ] || { t "extracao H3 (todas as ancoras)"; python3 -u pipeline/common/tarefa13_extrair_embeddings.py $C \
    --pretreino pretreino --saida fusao --dtype float16; }
B=runs/hm_clientes_f3h1
mkdir -p $B
for d in splits discretizacao vocabulario serializacao pretreino caches fusao; do
  [ -e $B/$d ] || [ ! -e $R/$d ] || ln -s ../hm_clientes/$d $B/$d
done
for S in 0 1 2; do
  [ -f $B/f3h1_fusao_s$S/predicoes_test.parquet ] || { t "F3-H1 fusao s$S"; $F configs/hm_clientes_f3h1_fusao.yaml --emb fusao --seed $S \
      --saida f3h1_fusao_s$S --dispositivo auto --avaliar_teste; }
  persistir
done

# ---- 3. F3-H3: motor v2 ----
if [ -f $R/v2/meta.json ]; then
  if [ ! -f $R/v2_pretreino/v2_relatorio.json ] || ! python3 -c "import json,sys; sys.exit(0 if len(json.load(open('$R/v2_pretreino/v2_relatorio.json'))['historico']) >= 3 else 1)"; then
    t "F3-H3 pre-treino v2"
    ( while sleep 600; do persistir; done ) & PERS=$!
    python3 -u pipeline/common/v2_pretreino.py $C --saida v2_pretreino --epocas 3 --retomar
    kill $PERS 2>/dev/null
    persistir
  fi
  [ -f $R/v2_fusao/embeddings_eventos_index.parquet ] || { t "F3-H3 extracao v2"; python3 -u pipeline/common/v2_extrair.py $C \
      --pretreino v2_pretreino --saida v2_fusao; }
  for S in 0 1 2; do
    [ -f $R/v2_fusao_s$S/predicoes_test.parquet ] || { t "F3-H3 fusao v2 s$S"; $F $C --emb v2_fusao --seed $S \
        --saida v2_fusao_s$S --dispositivo auto --avaliar_teste; }
    persistir
  done
else
  echo "F3-H3: sem $R/v2 (serialização do v2 não enviada); pulando"
fi

# ---- 4. E3b da F2-H10 ----
QB=Qwen/Qwen3.5-0.8B-Base
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

# ---- 5. TabPFN de 199 mil âncoras (se sobrar sessão; fora da avaliação já feita) ----
if [ -f /content/tabpfn_token ]; then
  export TABPFN_TOKEN=$(cat /content/tabpfn_token)
  for h in "" --hibrido; do
    nome=f2h11_tabpfn_$([ -n "$h" ] && echo hibrido || echo trad)_n300000
    [ -f $G/$nome/relatorio.json ] || { t $nome; python3 -u pipeline/common/f2h11_tabular.py $C --modelo tabpfn --n 300000 $h; persistir; }
  done
fi
echo "FIM $(date)"
