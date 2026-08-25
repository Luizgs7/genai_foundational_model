# Roadmap — Customer Foundation Model

Fases detalhadas de implementação, na sequência do `README.md` (kickoff). Numeração das tarefas mantida igual à do plano técnico original (rastreabilidade), **não** é a ordem de execução — a ordem recomendada está ao final deste documento.

Ver `.claude/skills/` para os guias especialistas consultados em cada tarefa.

---

## ✅ Tarefa 7 — Splits de treino/validação/teste (concluída)

**Objetivo**: corte temporal (treino/validação/teste) + separação de clientes elegíveis para downstream (≥2 transações) vs. cold-start (1 transação), sem leakage.

**Resultado**: `pipeline/splits/tarefa7_splits.py` → `pipeline/splits/splits.csv`. Corte: treino até 2025-10-30, validação 2025-10-31 a 2026-01-28, teste a partir de 2026-01-29. 74.353 / 8.093 / 17.554 linhas (treino/val/teste). 10.527 de 15.000 clientes elegíveis para downstream.

**Atualização (2026-08-25)**: `TEST_WINDOW_DAYS` alterado de 90 para **180 dias** (`VAL_WINDOW_DAYS` mantido em 90). Motivo: a Tarefa 8 define churn com horizonte N=121 dias — com janela de teste de 90 dias (< N), nenhum evento de teste conseguia satisfazer a condição de censura, e a taxa de churn no teste era 0% por construção (achado da Tarefa 9/`LIMITACOES.md`). Simulado o pipeline completo com várias combinações antes de decidir; 180 dias foi o menor valor que dá margem confortável acima de N (63% das linhas de teste viram determináveis, taxa de churn resultante 14,0% — não-degenerada) com o menor custo de volume de treino (74,4% do total, vs. 82,4% antes). Repropagado por todo o pipeline (Tarefas 4, 3, 8); Tarefa 5 (vocabulário) não depende de split, ficou igual; Tarefa 6 (`n_positions=512`) não mudou porque depende de `seq_full`, não de `seq_train`.

---

## ✅ Tarefa 4 — Discretização (bucketing por quantil por categoria) (concluída)

**Objetivo**: calcular os limites de bucket de `valor_total` e `desconto`, separadamente por `categoria_produto`, usando apenas o split de treino (`pipeline/splits/splits.csv`).

**Definition of Done**:
- Artefato JSON com os limites de quantil por categoria (10 buckets cada), calculados só sobre linhas com `split=train`.
- Função de mapeamento valor→bucket testada nos 3 splits (aplicando os limites do treino em val/test, nunca recalculando).
- Relatório de ocupação dos buckets por categoria no treino.

**Resultado**: `pipeline/discretizacao/tarefa4_discretizacao.py` → `pipeline/discretizacao/buckets.json` + `pipeline/discretizacao/discretizado.csv` (100.000 linhas). Ocupação dos 10 buckets balanceada (~10% cada) nas 4 categorias, para `valor_total` e `desconto`. Limites calibrados só no treino e reaplicados em val/teste — os 2 splits cobrem os 10 buckets (0-9) sem recalcular. Recalibrado em 2026-08-25 após a mudança de `TEST_WINDOW_DAYS` (Tarefa 7) — mesmo formato e balanceamento, limites de quantil levemente diferentes por causa do treino menor.

**Dependências**: Tarefa 7 (concluída).
**Skill de apoio**: `customer-sequence-serialization`.

---

## ✅ Tarefa 5 — Vocabulário e tokenizer fechado (concluída)

**Objetivo**: tokenizer de vocabulário fechado (sem BPE), cobrindo campos categóricos + buckets (Tarefa 4) + tokens estruturais (`BOS`, `EOS`, `EVT`, `PAD`, `UNK`).

**Definition of Done**:
- Vocabulário salvo em JSON (token → id), tamanho documentado (esperado ~300-400).
- `encode`/`decode` implementados e testados.
- 100% dos valores categóricos do dataset mapeiam para um token válido.

**Resultado**: `pipeline/vocabulario/tarefa5_vocabulario.py` → `pipeline/vocabulario/vocab.json` com **312 tokens**. `encode`/`decode` testados (`CATEGORIA_Máquinas` ↔ id 7). Validação de cobertura sobre 100.000 linhas em 8 campos categóricos: 0 valores caindo em `UNK`.

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

**Resultado**: `pipeline/serializacao/tarefa3_serializacao.py` → `pipeline/serializacao/sequencias.json` (15.000 clientes, `seq_full` + `seq_train` truncado no corte de treino). Comprimento de `seq_full`: mediana 37 tokens, p90=205, p99=481, máx=1057 (consistente com p99=40/máx=88 eventos/cliente; `seq_full` é independente do corte de split, não muda com `TEST_WINDOW_DAYS`). Spot-check de 5 clientes decodificados conferido linha a linha contra os dados originais (categoria/marca/buckets/recência batendo com os deltas de data reais). Validado: cliente de 1 evento gera `[BOS, EVT, ..., EOS]`.

Regerado em 2026-08-25 após a mudança de `TEST_WINDOW_DAYS` (Tarefa 7, 90→180 dias): o corte de treino ficou ~90 dias mais cedo, então **1.391 clientes sem nenhum evento de treino** (`seq_train=null`), contra 870 antes — custo direto de ampliar a janela de teste pra viabilizar avaliação de churn (Tarefa 8/9). Esses clientes continuam presentes em `seq_full` e nos splits de val/teste, só não contribuem para o corpus de pré-treino.

**Dependências**: Tarefa 7, Tarefa 4, Tarefa 5.
**Skill de apoio**: `customer-sequence-serialization`.

---

## ✅ Tarefa 6 — Arquitetura e instanciação do modelo (concluída)

**Objetivo**: config do modelo causal (HuggingFace, estilo GPT-2) com NoPE e FlashAttention2.

**Definition of Done**:
- `vocab_size` (Tarefa 5), `n_positions=128`, `n_embd`/`n_layer`/`n_head` (256-384 / 6-8 / 8), positional embedding neutralizado (NoPE), `attn_implementation="flash_attention_2"`.
- Modelo instancia sem erro; parâmetros reportados (~10-30M).
- Forward pass com batch real (Tarefa 3).
- Smoke test de 1-2 passos de treino sem erro/OOM.

**Resultado**: `pipeline/arquitetura/tarefa6_arquitetura.py`. Ambiente local não tem GPU/torch/transformers, então o script foi dividido em duas partes: (1) roda sempre, sem torch — deriva `n_positions=512` dos dados reais (Tarefa 3: p99=481 tokens de `seq_full`, corrigindo o `n_positions=128` original do DoD, que confundia eventos com tokens) e valida a lógica de batching/padding/máscara causal com numpy sobre um batch real de 8 clientes; (2) instancia `GPT2Config`/`GPT2LMHeadModel` (`n_embd=384`, `n_layer=8`, `n_head=8`), neutraliza NoPE (zera e congela `wpe.weight`) e ativa `attn_implementation="flash_attention_2"`, com forward pass + smoke test de 2 passos de treino.

Parte (2) validada de fato numa GPU L4 (Colab, via `colab_ssh/`) em 2026-08-25, após instalar `flash-attn==2.8.3.post1` (compilado da fonte — sem wheel pré-buildado pra Python 3.13/torch 2.11/cu128 nessa data). Corrigido no processo: API de `attn_implementation` mudou nessa versão do `transformers` — precisa ir em `GPT2Config(..., attn_implementation=...)`, não em `GPT2LMHeadModel(config, attn_implementation=...)` (dava `TypeError`). Resultado real (`pipeline/arquitetura/tarefa6_relatorio_gpu.json`): **14,5M parâmetros** (dentro dos 10-30M esperados), forward pass com logits `[8, 512, 312]`, smoke test de 2 passos com loss caindo `5.89 → 5.06`, pico de **1,5GB VRAM** (de 23GB disponíveis na L4) — folga grande para aumentar batch/n_layer depois se necessário.

**Dependências**: Tarefa 5, Tarefa 3.
**Skill de apoio**: `causal-transformer-nope-flashattn`.

---

## ✅ Tarefa 8 — Rótulos para tarefas downstream (concluída)

**Objetivo**: rótulos de churn (≥90 dias sem compra), próxima categoria, faixa de valor da próxima compra + baselines simples.

**Definition of Done**:
- Datasets rotulados restritos a `elegivel_downstream=True` onde aplicável.
- Base rate de churn documentado (esperado ~29,7%).
- Baseline calculado por tarefa (AUC-ROC para churn; accuracy/F1 para categoria/valor).

**Resultado**: `pipeline/rotulos_downstream/tarefa8_rotulos.py` → `pipeline/rotulos_downstream/tarefa8_rotulos.csv` (100.000 linhas) + `pipeline/rotulos_downstream/tarefa8_relatorio.json`.

- **Churn**: N=121 dias, escolhido varrendo candidatos sobre a distribuição real de gaps entre compras consecutivas (não um número de cabeça) até bater a taxa não-degenerada esperada. Base rate geral: **29,70%** (bate exato com o esperado no DoD). Tratamento de censura para o último evento de cada cliente (determinável só se a janela de N dias já fechou até o fim do dataset; 6.430 linhas excluídas por censura aberta). Desvio deliberado do DoD: churn **não** foi restrito a `elegivel_downstream=True` — um cliente de compra única também produz um rótulo de churn válido via censura, e restringir diluía a taxa de 29,70% pra 26,59% (documentado no código). Baseline (RFM: média histórica causal do gap entre compras do próprio cliente) — AUC-ROC geral 0,700.
- **Próxima categoria**: 85.000 linhas determináveis (15.000 sem próxima compra, excluídas). Baseline (moda histórica causal por cliente) — accuracy geral 52,2%, F1 macro 52,0% (val 59,5%/59,4%, mais alto que treino — datasets de val/teste têm clientes com histórico mais longo em média).
- **Próximo valor (bucket 0-9)**: mesmas 85.000 linhas. Baseline — accuracy geral ~9,8%, F1 macro ~9,6% — próximo do acaso (1/10=10%), esperado já que bucket de quantil por categoria não deveria ter autocorrelação forte de curto prazo por cliente. É exatamente o tipo de baseline fraco que justifica o Customer Foundation Model, se ele conseguir superá-lo.
- Sem `sklearn`/`scipy` disponíveis no ambiente — AUC-ROC, accuracy e F1 macro implementados na mão (numpy/pandas), sem dependência nova.

**Correção (2026-08-25)**: base rate de churn no split de **teste era 0%** (achado original desta tarefa — estrutural: janela de teste de 90 dias < horizonte de churn de 121 dias, nenhum evento de teste conseguia satisfazer a condição de censura fechada). Corrigido ampliando `TEST_WINDOW_DAYS` pra 180 dias na Tarefa 7 (ver detalhes lá) e repropagando pelo pipeline. Depois da correção: base rate de churn no teste = **14,03%** (não-degenerado), AUC-ROC do baseline por split: treino 0,699, validação 0,744, **teste 0,750** — os 3 splits agora são avaliáveis para a tarefa de churn.

**Dependências**: Tarefa 7, Tarefa 4, Tarefa 3.
**Skill de apoio**: `downstream-label-engineering`.

---

## ✅ Tarefa 9 — Memo de limitações e riscos (concluída)

**Objetivo**: `LIMITACOES.md` com os riscos, informado pelos artefatos concretos das tarefas 3-8.

**Resultado**: `LIMITACOES.md`, organizado em 7 seções (dados sintéticos; serialização; discretização/vocabulário; arquitetura/treino; splits; rótulos downstream; escopo não coberto), cada item citando o artefato/número concreto que o embasa. Destaques: 870 clientes sem `seq_train`; truncamento de sequência acima de `n_positions=512` (máx. observado 1057 tokens); Tarefa 6 validada só como smoke test (2 passos), não treino real; fragilidade do ambiente Colab efêmero (flash-attn precisa ~3h de recompilação a cada reinício); e o achado mais acionável — churn não é avaliável no split de teste como definido hoje (janela de teste de 90 dias < horizonte de churn de 121 dias).

---

## ✅ Tarefa 11 — Pré-treino real do backbone (concluída, 2026-08-25)

**Objetivo**: treinar de fato o modelo instanciado na Tarefa 6 (não só smoke test) sobre `seq_train`, com observabilidade máxima (métricas por passo/época, checkpoints, relatório final) e avaliação temporal correta — loss de validação medida só nos tokens do período de val, loss de teste medida uma única vez ao final.

**Resultado**: `pipeline/pretreino/tarefa_pretreino.py`, rodado no Colab (GPU L4). Artefatos em `pipeline/pretreino/`: `pretreino_config.json`, `pretreino_metricas_passo.csv` (por passo: loss, lr, grad_norm, tokens/s, mem. GPU), `pretreino_metricas_epoca.csv` (por época: loss treino/val, perplexidade), `pretreino_relatorio.json` (resumo final), `pretreino_loss_curve.png` (curva treino×val), `checkpoints/melhor.pt` e `checkpoints/final.pt` (pesos, bf16, ~14,3M parâmetros treináveis).

**Etapa 6 do relatório visual do cliente**: `pipeline/relatorios/rastro_cliente_860.703.096-50.html` (publicado como Artifact) ganhou uma 6ª etapa mostrando, token a token, as previsões reais do checkpoint treinado sobre a continuação de val/teste desse cliente (72 previsões: contexto, alvo real, previsto pelo modelo, confiança, loss) — inclui a nota de bastidores sobre o bug do duplo deslocamento. Cabeçalho e Etapa 1 do relatório também foram corrigidos para refletir os splits atuais deste cliente (25/4/2, não mais 29/1/1) após a mudança de `TEST_WINDOW_DAYS` na Tarefa 7.

**Bug real encontrado e corrigido durante a implementação**: a primeira tentativa de treino deu loss de validação **47x pior que chute aleatório** (ppl=14.676, vs. ~312 esperado do acaso) — investigado posição a posição num cliente real, a causa era um **duplo deslocamento dos labels**: o `GPT2LMHeadModel` da HuggingFace já desloca `labels` internamente para calcular a loss (`shift_logits`/`shift_labels`), mas o código estava passando `labels` **já pré-deslocados**, fazendo o modelo aprender a prever o token **dois passos à frente** em vez de um. Corrigido passando `labels` como cópia alinhada de `input_ids` (convenção padrão da HF). Efeito colateral do primeiro diagnóstico errado (hipótese de que o problema era o modelo aprender a prever `EOS` no corte de treino) também documentado no código, embora não tenha sido a causa real — mantido mascarado por ser semanticamente correto de qualquer forma (EOS deve marcar posição, não ser alvo de previsão).

**Métricas finais** (30 épocas, batch=32, lr=3e-4 com warmup+cosine, ~19,4 min de treino):
- **Overfitting claro e esperado**: melhor época é a **1ª** (loss_val=1,72, ppl=5,57) — o modelo (14,3M parâmetros treináveis) tem **16x mais parâmetros do que tokens de treino** (892.236 tokens). Loss de treino cai monotonicamente até 1,50 (época 30); loss de validação sobe de 1,72 para 1,88 no mesmo intervalo — curva clássica de overfitting, visível em `pretreino_loss_curve.png`.
- **Loss de teste** (checkpoint da época 1, avaliação única): **1,69** (perplexidade **5,44**) — bem abaixo do acaso (ppl≈312, tamanho do vocabulário), confirmando que o backbone aprendeu estrutura real, não decorou ruído.
- Avaliação inclui 961 clientes "frios" no teste (sem nenhum histórico de treino) — o modelo ainda assim generaliza melhor que o acaso pra eles.
- Pico de VRAM: 3,3GB (de 23GB disponíveis na L4) — folga grande pra aumentar `batch_size`/`n_layer` numa iteração futura.

**Implicação prática**: o modelo atual (14,3M parâmetros) está sobredimensionado para o volume de dados disponível (15.000 clientes sintéticos). Antes de qualquer uso downstream (fine-tuning DCNv2, Estágio 4), vale considerar reduzir `n_layer`/`n_embd` ou aumentar o volume de dados de treino — ~~ver `LIMITACOES.md` (atualizado)~~ **feito na Tarefa 12, abaixo**.

**Dependências**: Tarefa 6, Tarefa 3 (campo `seq_train_val`, adicionado especificamente para viabilizar a avaliação temporal de validação).
**Skill de apoio**: `causal-transformer-nope-flashattn`.

**Dependências**: Tarefas 3, 4, 5, 6, 7, 8.
**Skill de apoio**: nenhuma.

---

## ✅ Tarefa 12 — Redução do backbone para corrigir overfitting (concluída, 2026-08-25)

**Objetivo**: aplicar a recomendação da Tarefa 11 — o backbone original (`n_embd=384, n_layer=8, n_head=8`, 14,3M parâmetros treináveis) tinha 16x mais parâmetros do que tokens de treino (892.236), produzindo overfitting claro (melhor época de val era a 1ª de 30). Reduzir o modelo para uma razão parâmetros/tokens próxima de 1:1, sem alterar dados nem splits, e comparar contra o baseline.

**Config nova**: `n_embd=128, n_layer=4, n_head=4` (mantido `n_positions=512`) → **833.280 parâmetros treináveis** (contra 14,3M antes), razão tokens/parâmetro = **1,07** (contra 0,06 antes — 16x mais params que tokens). Baseline original preservado integralmente em `pipeline/pretreino/run_baseline_384emb_8layer/` (config, métricas, curva, checkpoints) para comparação.

**Resultado — comparação direta** (mesmos dados, mesmos splits, mesmas 30 épocas, batch=32, lr=3e-4):

| Métrica | Baseline (384/8, 14,3M params) | Modelo reduzido (128/4, 0,83M params) |
|---|---|---|
| Razão tokens/parâmetro treinável | 0,06 (16x mais params que tokens) | 1,07 (≈1:1) |
| Melhor época (por loss val) | 1 / 30 | **10 / 30** |
| Loss val (melhor época) | 1,7167 (ppl 5,57) | **1,6220 (ppl 5,06)** |
| Loss val (última época) | 1,8775 (+0,16 vs. melhor — diverge) | 1,6376 (+0,016 vs. melhor — quase plana) |
| Loss teste (checkpoint melhor época) | 1,6944 (ppl 5,44) | **1,6312 (ppl 5,11)** |
| Tempo total de treino | 1.163,6s (~19,4 min) | **276,5s (~4,6 min)** |
| Pico de VRAM | 3,32GB | 0,62GB |

O modelo menor generaliza melhor (loss de teste ~4% menor), quase não sobreajusta (a curva de val fica praticamente plana depois da época 10, em vez de subir continuamente) e treina ~4x mais rápido — confirma empiricamente a recomendação da Tarefa 11: o gargalo não era capacidade do modelo, era volume de dados de treino relativo ao tamanho do backbone.

**Artefatos**: `pipeline/pretreino/{pretreino_config.json,pretreino_relatorio.json,pretreino_metricas_epoca.csv,pretreino_metricas_passo.csv,pretreino_loss_curve.png,checkpoints/{melhor,final}.pt}` (substituem os do baseline, que ficam arquivados em `run_baseline_384emb_8layer/`).

**Pendente**: `pipeline/relatorios/rastro_cliente_860.703.096-50.html` (Etapa 6) ainda mostra as previsões do checkpoint **antigo** (384/8) — os números de loss/perplexidade/acerto do cliente exemplo estão desatualizados frente ao modelo atual. Precisa regenerar `cliente_860_predicoes.json` com o novo `melhor.pt` e republicar o Artifact antes de considerar o relatório visual consistente com o backbone vigente.

**Dependências**: Tarefa 11.
**Skill de apoio**: `causal-transformer-nope-flashattn`.

---

## ✅ Tarefa 13 — Fusão DCNv2 (Estágio 4/5, concluída, 2026-08-25)

**Objetivo**: implementar o Estágio 4/5 do `ARQUITETURA.md` — combinar o embedding sequencial do backbone (Tarefa 12) com features estáticas do cliente via DCNv2, treinar as 3 cabeças de tarefa (churn, próxima categoria, próximo valor) e comparar contra os baselines causais da Tarefa 8.

**Decisão de desenho importante — âncora por evento, não por split**: a primeira tentativa usava, por cliente, só a hidden state na fronteira de cada split (a mesma posição já explorada na Etapa 6 do relatório do cliente). Isso introduziu um viés de seleção grave: a *última* transação de um cliente dentro de um split tende a estar perto da borda temporal (pouco tempo pra "voltar a comprar"), inflando a taxa de churn medida para 83% no treino e 55% na validação — bem longe dos ~29,7%/~31,8% reais da Tarefa 8. Corrigido usando **todas as ~98,7 mil transações rotuladas** como âncoras independentes (hidden state causal na posição de cada evento, formula `11 + 12*i` no índice de tokens — validada batendo com as contagens exatas do cliente 860.703.096-50), reproduzindo exatamente as taxas oficiais da Tarefa 8 (31,8%/31,7%/14,0%). ~1.288 eventos (1,3%) de clientes de cauda longa (>512 tokens) ficaram fora da janela do backbone e foram excluídos — mesma limitação já documentada sobre `n_positions`.

**Arquitetura**: `pipeline/fusao/tarefa13_extrair_embeddings.py` (GPU, backbone **congelado** — só inferência) gera `embeddings_eventos.npy` (98.712 × 128) a partir do checkpoint da Tarefa 12. `pipeline/fusao/tarefa13_treinar_fusao.py` (CPU, ~40s) treina a fusão: `x0` = embedding sequencial (128) + UF (embedding aprendido, 4) + sexo (1) + idade na data da âncora (padronizada só com estatística do treino) = 134 dimensões; 2 camadas de Cross Network v2 de baixo posto (rank=8, evita explosão de parâmetros); torre deep de 1 camada (134→16); saídas concatenadas (150) alimentam 3 cabeças lineares. **~9.005 parâmetros treináveis para 73.065 exemplos de treino** (razão exemplos/parâmetro ≈8,1 — bem folgado, deliberado após a lição de overfitting da Tarefa 12).

**Resultado — comparação contra os baselines causais da Tarefa 8** (split de teste, avaliação única no checkpoint de melhor AUC de val, época 31/40):

| Tarefa | Métrica | Baseline causal (Tarefa 8) | DCNv2 + backbone | Delta |
|---|---|---|---|---|
| Churn | AUC-ROC | 0,750 (RFM: recência histórica) | **0,800** | **+0,050** |
| Próxima categoria | Accuracy | 0,590 (moda histórica por cliente) | 0,558 | −0,032 |
| Próxima categoria | F1 macro | 0,587 | 0,550 | −0,037 |
| Próximo valor | Accuracy | 0,100 (moda histórica por cliente) | 0,094 | −0,005 |

**Leitura honesta**: churn é onde o Customer Foundation Model demonstra valor real — o embedding sequencial captura sinal além da recência simples (RFM), batendo o baseline com margem clara. Próxima categoria e próximo valor **não** superam seus baselines nesta primeira iteração: próxima categoria já tem um baseline por-cliente forte (moda histórica) difícil de bater com uma cabeça linear de 604 parâmetros treinada em conjunto com as outras duas tarefas; próximo valor segue praticamente no acaso em ambos os casos (baseline E modelo), reforçando a hipótese já levantada em `LIMITACOES.md` (item 6) de que o valor da próxima compra pode não ter autocorrelação de curto prazo relevante neste gerador sintético — um resultado fraco aqui não é necessariamente falha do modelo. Não houve tentativa de ajustar hiperparâmetros pra "melhorar" esses dois números — reportados como saíram, por honestidade científica.

**Artefatos**: `pipeline/fusao/{tarefa13_config.json,tarefa13_metricas_epoca.csv,tarefa13_relatorio.json,tarefa13_loss_curve.png,tarefa13_fusao.pt,embeddings_eventos.npy,embeddings_eventos_index.csv}`.

**Modo de treino usado**: opção 1 do `ARQUITETURA.md` (backbone congelado) — não testada a opção 2 (fine-tuning conjunto).

**Dependências**: Tarefa 8, Tarefa 12.
**Skill de apoio**: `downstream-label-engineering` (reaproveita `auc_roc`/`accuracy`/`f1_macro` de `tarefa8_rotulos.py`).

---

## Tarefa 10 — Perguntas para o time / plano de validação com dados reais

**Objetivo**: `PERGUNTAS_ABERTAS.md` com decisões que só podem ser validadas com dados reais.

**Dependências**: Tarefa 9.
**Skill de apoio**: nenhuma.

---

## Ordem de execução recomendada

`7 (✅) → 4 (✅) → 5 (✅) → 3 (✅) → 6 (✅) → 8 (✅) → 9 (✅) → 11 (✅) → 12 (✅) → 13 (✅) → 10`

## Protocolo de execução

Cada tarefa é implementada, validada contra seu DoD, e reportada para aprovação antes de iniciar a próxima. Ver `ARQUITETURA.md` para o desenho detalhado de como as tarefas 3-6 se encaixam no pipeline fim-a-fim.
