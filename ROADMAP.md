# Roadmap — Customer Foundation Model

Fases detalhadas de implementação, na sequência do `README.md` (kickoff). Numeração das tarefas mantida igual à do plano técnico original (rastreabilidade), **não** é a ordem de execução — a ordem recomendada está ao final deste documento.

Ver `.claude/skills/` para os guias especialistas consultados em cada tarefa.

---

## ✅ Tarefa 7 — Splits de treino/validação/teste (concluída)

**Objetivo**: corte temporal (treino/validação/teste) + separação de clientes elegíveis para downstream (≥2 transações) vs. cold-start (1 transação), sem leakage.

**Resultado**: `pipeline/tarefa7_splits.py` → `pipeline/artifacts/splits.csv`. Corte: treino até 2026-01-28, validação 2026-01-29 a 2026-04-28, teste a partir de 2026-04-29. 82.446 / 8.003 / 9.551 linhas (treino/val/teste). 10.527 de 15.000 clientes elegíveis para downstream.

---

## ✅ Tarefa 4 — Discretização (bucketing por quantil por categoria) (concluída)

**Objetivo**: calcular os limites de bucket de `valor_total` e `desconto`, separadamente por `categoria_produto`, usando apenas o split de treino (`pipeline/artifacts/splits.csv`).

**Definition of Done**:
- Artefato JSON com os limites de quantil por categoria (10 buckets cada), calculados só sobre linhas com `split=train`.
- Função de mapeamento valor→bucket testada nos 3 splits (aplicando os limites do treino em val/test, nunca recalculando).
- Relatório de ocupação dos buckets por categoria no treino.

**Resultado**: `pipeline/tarefa4_discretizacao.py` → `pipeline/artifacts/buckets.json` + `pipeline/artifacts/discretizado.csv` (100.000 linhas). Ocupação dos 10 buckets balanceada (~10% cada) nas 4 categorias, para `valor_total` e `desconto`. Limites calibrados só no treino e reaplicados em val/teste — os 2 splits cobrem os 10 buckets (0-9) sem recalcular.

**Dependências**: Tarefa 7 (concluída).
**Skill de apoio**: `customer-sequence-serialization`.

---

## ✅ Tarefa 5 — Vocabulário e tokenizer fechado (concluída)

**Objetivo**: tokenizer de vocabulário fechado (sem BPE), cobrindo campos categóricos + buckets (Tarefa 4) + tokens estruturais (`BOS`, `EOS`, `EVT`, `PAD`, `UNK`).

**Definition of Done**:
- Vocabulário salvo em JSON (token → id), tamanho documentado (esperado ~300-400).
- `encode`/`decode` implementados e testados.
- 100% dos valores categóricos do dataset mapeiam para um token válido.

**Resultado**: `pipeline/tarefa5_vocabulario.py` → `pipeline/artifacts/vocab.json` com **312 tokens**. `encode`/`decode` testados (`CATEGORIA_Máquinas` ↔ id 7). Validação de cobertura sobre 100.000 linhas em 8 campos categóricos: 0 valores caindo em `UNK`.

**Dependências**: Tarefa 4.
**Skill de apoio**: `customer-sequence-serialization`.

---

## ✅ Tarefa 3 — Serialização das sequências por cliente (concluída)

**Objetivo**: montar, por `cpf`, a sequência ordenada de tokens (evento a evento) usando o tokenizer da Tarefa 5.

**Definition of Done**:
- Sequência de token-ids por cliente, respeitando o corte de split (sequência de treino não inclui eventos além do corte de validação).
- Distribuição do comprimento de sequência reportada.
- Spot-check manual de 5 clientes decodificados.
- Clientes com 1 evento gerando sequência `[BOS][EVT][EOS]` válida.

**Resultado**: `pipeline/tarefa3_serializacao.py` → `pipeline/artifacts/sequencias.json` (15.000 clientes, `seq_full` + `seq_train` truncado no corte de treino). Comprimento de `seq_full`: mediana 37 tokens, p90=205, p99=481, máx=1057 (consistente com p99=40/máx=88 eventos/cliente). 870 clientes sem nenhum evento de treino (`seq_train=null` — só entraram na base após o corte). Spot-check de 5 clientes decodificados conferido linha a linha contra os dados originais (categoria/marca/buckets/recência batendo com os deltas de data reais). Validado: cliente de 1 evento gera `[BOS, EVT, ..., EOS]`.

**Dependências**: Tarefa 7, Tarefa 4, Tarefa 5.
**Skill de apoio**: `customer-sequence-serialization`.

---

## ⚠️ Tarefa 6 — Arquitetura e instanciação do modelo (código pronto, validação em GPU pendente)

**Objetivo**: config do modelo causal (HuggingFace, estilo GPT-2) com NoPE e FlashAttention2.

**Definition of Done**:
- `vocab_size` (Tarefa 5), `n_positions=128`, `n_embd`/`n_layer`/`n_head` (256-384 / 6-8 / 8), positional embedding neutralizado (NoPE), `attn_implementation="flash_attention_2"`.
- Modelo instancia sem erro; parâmetros reportados (~10-30M).
- Forward pass com batch real (Tarefa 3).
- Smoke test de 1-2 passos de treino sem erro/OOM.

**Resultado**: `pipeline/tarefa6_arquitetura.py`. Ambiente local não tem GPU/torch/transformers, então o script foi dividido em duas partes: (1) roda sempre, sem torch — deriva `n_positions=512` dos dados reais (Tarefa 3: p99=481 tokens de `seq_full`, corrigindo o `n_positions=128` original do DoD, que confundia eventos com tokens) e valida a lógica de batching/padding/máscara causal com numpy sobre um batch real de 8 clientes; (2) roda só se torch/transformers estiverem disponíveis — instancia `GPT2Config`/`GPT2LMHeadModel` (`n_embd=384`, `n_layer=8`, `n_head=8`, ~14,3M parâmetros estimados), neutraliza NoPE (zera e congela `wpe.weight`) e ativa `attn_implementation="flash_attention_2"`, com forward pass + smoke test de 2 passos de treino. Parte (1) validada nesta sessão; parte (2) requer GPU (Colab, ~16GB VRAM) — pendente de execução lá.

**Dependências**: Tarefa 5, Tarefa 3.
**Skill de apoio**: `causal-transformer-nope-flashattn`.

---

## Tarefa 8 — Rótulos para tarefas downstream

**Objetivo**: rótulos de churn (≥90 dias sem compra), próxima categoria, faixa de valor da próxima compra + baselines simples.

**Definition of Done**:
- Datasets rotulados restritos a `elegivel_downstream=True` onde aplicável.
- Base rate de churn documentado (esperado ~29,7%).
- Baseline calculado por tarefa (AUC-ROC para churn; accuracy/F1 para categoria/valor).

**Dependências**: Tarefa 7, Tarefa 4, Tarefa 3.
**Skill de apoio**: `downstream-label-engineering`.

---

## Tarefa 9 — Memo de limitações e riscos

**Objetivo**: `LIMITACOES.md` com os riscos, informado pelos artefatos concretos das tarefas 3-8.

**Dependências**: Tarefas 3, 4, 5, 6, 7, 8.
**Skill de apoio**: nenhuma.

---

## Tarefa 10 — Perguntas para o time / plano de validação com dados reais

**Objetivo**: `PERGUNTAS_ABERTAS.md` com decisões que só podem ser validadas com dados reais.

**Dependências**: Tarefa 9.
**Skill de apoio**: nenhuma.

---

## Ordem de execução recomendada

`7 (✅) → 4 (✅) → 5 (✅) → 3 (✅) → 6 (⚠️) → 8 → 9 → 10`

## Protocolo de execução

Cada tarefa é implementada, validada contra seu DoD, e reportada para aprovação antes de iniciar a próxima. Ver `ARQUITETURA.md` para o desenho detalhado de como as tarefas 3-6 se encaixam no pipeline fim-a-fim.
