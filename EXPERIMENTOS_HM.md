# Experimento H&M — Customer Foundation Model vs. ML tradicional (H0–H10)

Branch `luiz_g_product_exp`. Primeiro teste do motor genérico numa base real
grande: **H&M Personalized Fashion** (mirror HF
`einrafh/hnm-fashion-recommendations-data`) — 31,8M itens comprados por 1,36M
clientes em 734 dias (2018-09-20 a 2020-09-22). Plano aprovado em 2026-10-16
(`/home/lgsouza/.claude/plans/na-branch-luiz-g-product-exp-fa-a-zesty-hollerith.md`).
Números consolidados em `runs/hm_clientes/resultados_consolidados.json`
(`pipeline/relatorios/consolidar_hm.py`).

## Perguntas e respostas curtas

| # | Pergunta | Resposta (teste, avaliação única H10) |
|---|---|---|
| H6 | O embedding sozinho alimenta bem um GBDT? | **Não.** Fica acima do baseline causal em churn, mas muito abaixo do GBDT tradicional (churn 0,768 vs 0,824). |
| H5 | Quanto entrega o GBDT tradicional (97 features + HPO)? | É o **melhor modelo** nas 4 tarefas: churn 0,824, online 0,845, categoria 0,373, LTV 0,399. |
| H7 | Híbrido (features + embedding) ganha do tradicional? | **Não de forma consistente.** Com LightGBM perde nas 4 tarefas, por pouco; com XGBoost ganha só em churn (+0,007), porque o XGBoost tradicional é o mais fraco ali. |
| H8 | Quais escolhas de arquitetura importam? | **RoPE** domina a loss de pré-treino (0,91 vs 1,59), mas a diferença é quase toda o token de mês do calendário, que o NoPE não extrapola; o resto é ±1–2%. **A loss não se traduz em downstream**: o backbone final (5M, 4× mais barato) empata na validação e perde no teste. |
| H9 | Qual o efeito do cadastro na fusão? | **Pequeno**: +0,007 em categoria, ≈0 em online, −0,003 em churn e −0,002 em LTV no teste. O **embedding** pesa mais: +0,025 em online e +0,026 em categoria, mas −0,004 em churn e −0,008 em LTV. |

Conclusão operacional: **nesta base, o modelo fundacional não supera o ML
tradicional bem-feito em nenhuma das 4 tarefas**. Ele agrega sinal real
(sequencial) a online e categoria **na fusão**, mas o GBDT com RFM e composição
já captura esse sinal. O ganho que aparece é sempre menor que a diferença entre
o GBDT e a fusão.

## Dados e protocolo

- **Entidade** = cliente; **evento/token** = item comprado; **âncora downstream** = cesta (cliente + dia, no último item).
  Ancorar por item daria rótulos triviais: um item seguido por outro da mesma cesta tem gap de 0 dias e nunca é churn.
- **Splits temporais.**

  | Split | Período |
  |---|---|
  | Treino | até 2019-12-26 |
  | Validação | 90 dias, até 2020-03-25 |
  | Teste | 180 dias, 2020-03-26 a 2020-09-22, cobrindo a COVID |

  A regra é janela de teste ≥ 2× o horizonte de churn.
- **Âncoras.**

  | Split | Âncoras | Clientes |
  |---|---|---|
  | Treino | 2,0M (amostra de 5,9M) | — |
  | Validação | 940 mil | 469 mil |
  | Teste | 2,24M | 742 mil |

  Todos os modelos usam **as mesmas âncoras**.
- **Tarefas.**
  - `churn`: sem nova cesta em 90 dias, com censura. Taxa base: 25% no treino, 30% na validação e 16% no teste.
  - `proxima_online`: a próxima cesta tem item online.
  - `next_category`: grupo de produto dominante da próxima cesta, 10 classes.
  - `ltv_90d`: gasto em 90 dias, em faixas de quantil (8 classes efetivas).
- **Tokenização.** Vocabulário fechado de 2.428 tokens. Cada item vira ~11 tokens: índice, grupo, tipo, cor, linha, canal, artigo (top-2000 + "outros"), faixa de preço por grupo, recência e mês.
  São 227M tokens de treino; `n_positions` = 1024 cobre 95% dos clientes inteiros.
- **Seleção.** Tudo é escolhido na validação: HPO Optuna TPE, k do PCA, arquitetura.
  O teste é avaliado **uma vez** (H10), com hiperparâmetros congelados (`tarefa14_gbdt.py --params_de`) e 3 sementes.
- **Comparações.** IC 95% por **bootstrap Poisson por cliente**: as predições das 3 sementes são promediadas e cada cliente recebe um peso Poisson(1) por réplica (`avaliacao.py`).

## Resultados no teste (H10, 3 sementes)

AUC para churn/online; accuracy para categoria/LTV.

| Modelo | Churn | Online | Categoria | LTV |
|---|---|---|---|---|
| Baseline causal | 0,7435 | 0,8034 | 0,3302 | 0,3386 |
| **GBDT tradicional — LightGBM (H5)** | **0,8242** | 0,8447 | 0,3727 | **0,3987** |
| GBDT tradicional — XGBoost (H5) | 0,8153 | 0,8447 | **0,3736** | 0,3985 |
| GBDT só embedding H3, PCA-32 (H6) | 0,7676 | 0,7880 | 0,3429 | 0,3139 |
| GBDT híbrido LightGBM (H7) | 0,8205 | 0,8450 | 0,3698 | 0,3918 |
| GBDT híbrido XGBoost (H7) | 0,8215 | **0,8453** | 0,3706 | 0,3886 |
| GBDT híbrido LightGBM, backbone final | 0,8187 | 0,8445 | 0,3719 | 0,3953 |
| GBDT híbrido LightGBM sem cadastro (H9) | 0,8195 | 0,8438 | 0,3704 | 0,3915 |
| Fusão DCNv2 completa, backbone H3 (H4) | 0,8098 | 0,8355 | 0,3510 | 0,3677 |
| Fusão sem cadastro (H9 c) | 0,8122 | 0,8351 | 0,3456 | 0,3716 |
| Fusão sem embedding (H9 e) | 0,8157 | 0,8116 | 0,3327 | 0,3818 |
| Fusão completa, backbone final (H8) | 0,8021 | 0,8278 | 0,3454 | 0,3609 |

Os valores são médias das 3 sementes; os desvios entre sementes são ≤ 0,004 em
todos os casos. O bootstrap usa o ensemble das sementes, por isso os valores
dele diferem um pouco destes.

### Deltas no teste com IC 95% (bootstrap por cliente, B = 300)

Fonte: `runs/hm_clientes/h10_bootstrap_final.json`.
Com ~600 mil clientes no teste, o IC tem largura de ±0,0005. A pergunta relevante passa a ser a **magnitude** do efeito, não a significância.

| Comparação (B − A) | Churn | Online | Categoria | LTV |
|---|---|---|---|---|
| H4: fusão completa − GBDT tradicional LightGBM | -0.0111 [-0.0115, -0.0107] | -0.0068 [-0.0071, -0.0066] | -0.0143 [-0.0148, -0.0137] | -0.0245 [-0.0254, -0.0236] |
| H4: fusão completa − GBDT tradicional XGBoost | -0.0023 [-0.0028, -0.0019] | -0.0070 [-0.0072, -0.0067] | -0.0152 [-0.0157, -0.0146] | -0.0239 [-0.0248, -0.0230] |
| H8: fusão com backbone final − GBDT tradicional LightGBM | -0.0168 [-0.0172, -0.0165] | -0.0142 [-0.0144, -0.0139] | -0.0280 [-0.0286, -0.0273] | -0.0355 [-0.0365, -0.0345] |
| H6: só embedding − tradicional (LightGBM) | -0.0542 [-0.0547, -0.0536] | -0.0564 [-0.0571, -0.0558] | -0.0298 [-0.0306, -0.0291] | -0.0876 [-0.0888, -0.0865] |
| H7: híbrido − tradicional (LightGBM) | -0.0037 [-0.0039, -0.0036] | +0.0003 [+0.0003, +0.0003] | -0.0029 [-0.0032, -0.0026] | -0.0073 [-0.0078, -0.0068] |
| H7: híbrido − tradicional (XGBoost) | +0.0068 [+0.0066, +0.0070] | +0.0004 [+0.0003, +0.0004] | -0.0030 [-0.0034, -0.0027] | -0.0100 [-0.0106, -0.0095] |
| H7: híbrido backbone final − tradicional (LightGBM) | -0.0055 [-0.0056, -0.0055] | -0.0001 [-0.0002, -0.0001] | -0.0006 [-0.0009, -0.0004] | -0.0034 [-0.0038, -0.0031] |
| H9: efeito do embedding na fusão (d − e) | -0.0042 [-0.0046, -0.0037] | +0.0249 [+0.0246, +0.0253] | +0.0256 [+0.0248, +0.0262] | -0.0083 [-0.0090, -0.0076] |
| H9: efeito do cadastro na fusão (d − c) | -0.0025 [-0.0027, -0.0023] | +0.0003 [+0.0002, +0.0005] | +0.0069 [+0.0064, +0.0074] | -0.0016 [-0.0021, -0.0012] |
| H9: efeito do cadastro no GBDT híbrido | +0.0008 [+0.0007, +0.0009] | +0.0013 [+0.0012, +0.0014] | -0.0006 [-0.0008, -0.0003] | +0.0003 [+0.0001, +0.0005] |
| H8: backbone final − referência H3 (fusão) | -0.0057 [-0.0060, -0.0054] | -0.0073 [-0.0075, -0.0071] | -0.0137 [-0.0143, -0.0131] | -0.0110 [-0.0118, -0.0103] |

Todas as 52 comparações calculadas têm IC excluindo o zero (`runs/hm_clientes/h10_bootstrap_final.json`). Os deltas usam o **ensemble** das 3 sementes; a diferença para as médias por semente é maior na fusão, que ganha mais com o ensemble (categoria: −0,014 contra −0,022). O bootstrap foi fechado em parte localmente (`scripts/certeza_local.sh`, com a projeção PCA baixada do Colab via `--pca_cache`); as reexecuções reproduziram exatamente as métricas originais.

## Resultados na validação (seleção)

| Modelo | Churn | Online | Categoria | LTV |
|---|---|---|---|---|
| Baseline causal | 0,7339 | 0,8007 | 0,3558 | 0,3348 |
| GBDT tradicional LightGBM (40/25 tentativas) | 0,8000 | 0,8295 | 0,3932 | 0,4243 |
| GBDT tradicional XGBoost | 0,8004 | 0,8293 | 0,3942 | 0,4226 |
| GBDT tradicional, orçamento da H7 (20/12) | 0,7996 | 0,8294 | 0,3932 | — |
| GBDT só embedding PCA-32 (H3) | 0,7391 | 0,7683 | 0,3783 | 0,3642 |
| GBDT só embedding bruto 384 (H3) | 0,7286 | 0,7633 | cortado | cortado |
| GBDT só embedding PCA-32 (final) | 0,7348 | 0,7907 | 0,3761 | 0,3692 |
| GBDT híbrido LightGBM | 0,7955 | 0,8297 | 0,3921 | 0,4167 |
| GBDT híbrido XGBoost | 0,7952 | 0,8298 | 0,3931 | 0,4156 |
| GBDT híbrido LightGBM (final) | 0,7988 | 0,8297 | 0,3925 | 0,4199 |
| GBDT híbrido sem cadastro | 0,7945 | 0,8282 | 0,3928 | 0,4184 |
| Fusão (a) só embedding | 0,7655 | 0,7761 | 0,3719 | 0,3424 |
| Fusão (b) embedding + cadastro | 0,7681 | 0,7704 | 0,3735 | 0,3610 |
| Fusão (c) embedding + causais | 0,7824 | 0,8165 | 0,3660 | 0,3757 |
| Fusão (d) completa | 0,7814 | 0,8206 | 0,3780 | 0,3646 |
| Fusão (e) cadastro + causais | 0,7826 | 0,8067 | 0,3738 | 0,4136 |
| Fusão completa (backbone final) | 0,7843 | 0,8190 | 0,3772 | 0,3889 |

Participação do embedding na importância (gain) do GBDT híbrido:

| Tarefa | Participação |
|---|---|
| Churn | 4–7% |
| Online | ~1% |
| Categoria | 9–12% |
| LTV | 3–5% |

## H3 — Backbone de referência (GPT-2 + NoPE)

- **Modelo:** 384×6, 6 cabeças, 11,6M parâmetros treináveis, o que dá 19,6 tokens de treino por parâmetro.
- **Treino:** 3 épocas e 684M tokens vistos, em 3,2 h de L4 (59,7 mil tokens/s, pico de 17,0 GB).
- **Seleção:** a melhor época foi a 2. Na 3ª, a validação piorou (1,688) enquanto o treino caía para 0,77, sinal claro de sobreajuste.
- **Loss / perplexidade:** validação 1,635 / 5,13; teste 1,455 / 4,29.

## H8 — Ablações (1 época, 1 semente, validação)

A sonda GBDT é um LightGBM com parâmetros fixos sobre o PCA-32. A sonda de fusão é a DCNv2 completa.

| Ablação | Params | Loss val | Tempo | GBDT churn | GBDT categoria | Fusão churn | Fusão categoria | Fusão LTV | Fusão online |
|---|---|---|---|---|---|---|---|---|---|
| ref (decoder próprio ≡ H3) | 11,6M | 1,588 | 34 min | 0,682 | 0,376 | 0,778 | 0,364 | 0,385 | 0,814 |
| tam 128×4 | 1,1M | 1,611 | 10 min | 0,719 | 0,376 | 0,775 | 0,376 | 0,409 | 0,808 |
| tam 256×6 | 5,4M | 1,576 | 21 min | 0,735 | 0,375 | 0,780 | 0,355 | 0,407 | 0,821 |
| tam 512×8 | 26,5M | 1,548 | 59 min | 0,694 | 0,374 | 0,776 | 0,325 | 0,371 | 0,815 |
| attn GQA kv=2 | 10,4M | 1,598 | 31 min | 0,734 | 0,378 | **0,789** | 0,380 | 0,411 | 0,821 |
| attn MQA kv=1 | 10,1M | 1,575 | 31 min | 0,706 | 0,376 | 0,785 | 0,377 | 0,395 | **0,823** |
| FFN SwiGLU | 11,6M | 1,570 | 35 min | 0,728 | **0,381** | 0,786 | 0,377 | 0,402 | 0,817 |
| **posição RoPE** | 11,6M | **0,911** | 42 min | **0,739** | 0,379 | 0,787 | **0,383** | 0,389 | 0,821 |
| posição aprendida | 12,0M | 1,463 | 33 min | 0,685 | 0,378 | 0,784 | 0,373 | **0,413** | 0,822 |
| norm RMSNorm | 11,6M | 1,606 | 32 min | 0,728 | 0,377 | 0,782 | 0,378 | 0,396 | 0,821 |
| dropout 0 | 11,6M | 1,581 | 31 min | 0,682 | 0,377 | 0,770 | 0,360 | 0,371 | 0,820 |
| contexto 512 | 11,6M | 1,443 | 29 min | 0,683 | 0,376 | **0,790** | 0,377 | 0,410 | 0,820 |
| contexto 2048 | 11,6M | 1,659 | 35 min | 0,730 | 0,379 | 0,786 | 0,362 | 0,403 | 0,822 |
| "moderna" (RoPE+RMS+SwiGLU+GQA2+drop0) | 10,4M | 0,953 | 43 min | 0,716 | 0,378 | 0,785 | 0,378 | 0,382 | 0,816 |

- Os números de ref, tam_\*, ctx_\*, pos_\* e "moderna" foram lidos no Colab antes de uma queda do runtime. Estão transcritos em `ablacoes/resultados_transcritos.json`, e os checkpoints dessas ablações se perderam. As de atenção, FFN, RMSNorm e dropout têm o JSON original.
- Com 1 semente, o ruído da fusão é de ~±0,005. Diferenças menores que isso não discriminam.
- A **loss de contexto não é comparável** entre `ctx_*`, porque o conjunto de tokens avaliados muda com a janela.

**Repetição com 3 sementes (NoPE × RoPE, 1 época, validação; média ± desvio):**

| | Loss val | Sonda GBDT churn | Fusão churn | Fusão online | Fusão categoria | Fusão LTV |
|---|---|---|---|---|---|---|
| NoPE (`ref`, `ref_s1`, `ref_s2`) | 1,601 ± 0,015 | 0,724 ± 0,003 | 0,783 ± 0,004 | 0,823 ± 0,002 | 0,381 ± 0,004 | 0,400 ± 0,008 |
| RoPE (`pos_rope`, `_s1`, `_s2`) | 0,916 ± 0,007 | 0,713 ± 0,008 | 0,786 ± 0,002 | 0,816 ± 0,002 | 0,382 ± 0,001 | 0,385 ± 0,018 |

O RoPE não melhora nenhuma tarefa. A sonda GBDT de 1 semente oscila até ±0,04 entre execuções (a `ref` original deu 0,682 e as três novas, ~0,724), então ela não serve para ranquear ablações; o `ref.json` atual é a reexecução da semente 0.

**Diagnóstico pareado** (`relatorios/diagnostico_copia_pareado.json`; 384×6, 1 época, 3 sementes por lado, 5.000 clientes): a contribuição do token MES é de 0,74–0,77 no NoPE e de 0,07–0,08 no RoPE. A loss sem MES fica em 0,943–0,946 no NoPE e em 0,928–0,933 no RoPE. Fora o mês, sobra ~0,014 nats, cerca de 2% da diferença total.

**Backbone final** (escolhido na validação):
- **Arquitetura:** RoPE + SwiGLU + GQA (4 cabeças, kv=2) + LayerNorm + dropout 0,1, em 256×6, com 4,99M parâmetros.
- **Por que essa combinação:**
  - RoPE foi o maior efeito isolado;
  - GQA e SwiGLU foram neutros ou positivos;
  - dropout 0 piorou o downstream;
  - 256×6 empatou com 384×6 com metade dos parâmetros.
- **Resultado com 3 sementes e orçamento completo:**

  | | Valor |
  |---|---|
  | Loss val (média) | 0,915 |
  | Loss teste (média) | 0,886 |
  | Melhor época | sempre a 1ª; a 2ª já sobreajusta |
  | Tempo por semente | 44–49 min, 3,9–4,3× mais barato que a H3 |
  | Pico de memória | 7,2 GB |
  | Fusão val | churn 0,784, categoria 0,377, LTV 0,389, online 0,819 |
  | Fusão teste | **pior que a H3** nas 4 tarefas, de −0,006 a −0,014 |

**Leitura crítica: a diferença de loss é o token de mês.** Ela foi decomposta
por campo em `pipeline/common/diagnostico_copia.py` (2.000 clientes de validação,
mesmos alvos) e `pipeline/common/diagnostico_mes.py`. Os resultados estão em
`runs/hm_clientes/relatorios/diagnostico_{copia,mes}.json`.

| Modelo | Loss total | Contribuição do token MES | Loss sem MES |
|---|---|---|---|
| NoPE H3 (3 épocas) | 1,655 | 0,786 | 0,955 |
| NoPE dropout 0 (1 época) | 1,589 | 0,736 | 0,939 |
| NoPE GQA kv=2 (1 época) | 1,604 | 0,743 | 0,947 |
| RoPE final, sementes 0/1/2 | 0,915–0,920 | 0,066–0,072 | 0,932–0,938 |

- **Fora o mês, NoPE e RoPE empatam**, inclusive nos campos que repetem o valor do item anterior. A hipótese de "cópia intra-cesta", levantada antes desta análise, está **refutada**.
- **Mecanismo.** Sem posição, a atenção não distingue o último token MES do histórico dos anteriores. O NoPE aprendeu a prever os meses em que as sequências de treino terminam:

  | Split | Meses reais | O NoPE prevê | O RoPE prevê | Loss no MES (NoPE / RoPE) |
  |---|---|---|---|---|
  | Validação | jan–mar | dez/nov/out | jan–mar | 8,4 / 0,7 |
  | Teste | abr–ago | dez | abr–ago | 5,2 / 0,4 |

  O RoPE copia o mês do evento anterior pela distância relativa.
- **Mesma falha no decoder próprio com NoPE:** não é uma particularidade do GPT-2 da HF.
- **Consequências.**
  - A melhora de 43% na loss é calendário, sem informação de comportamento, e não havia por que melhorar o downstream.
  - Isso explica a loss de teste da H3 abaixo da de validação, pelo erro menor no MES.
  - O token de mês absoluto é um defeito de desenho da tokenização: domina quase metade da loss da referência. Deve sair do alvo ou virar codificação relativa, e a loss reportada deve ser decomposta por campo.
- **Também descartados como causa:**
  - **Vazamento:** o teste de causalidade passou, ou seja, alterar tokens futuros não muda os logits passados.
  - **Bug de avaliação:** uma avaliação independente com o rotulador legado reproduziu 0,94 contra 1,66.

**Implicação:** a loss de pré-treino **não serve como critério de seleção** de
backbone para estas tarefas. É preciso uma sonda downstream, e ela tem que ser
avaliada fora do período de validação.

## Custos medidos (Colab L4)

| Etapa | Tempo |
|---|---|
| H3 (11,6M, 3 épocas) | 3,2 h |
| Backbone final (5M, 2 épocas, 1ª selecionada) | ~47 min/semente |
| Extração dos embeddings (5,2M âncoras, fp16) | ~23 min |
| Fusão DCNv2 (2M âncoras, 8 épocas) | 5–10 min |
| Ablação H8 (pré-treino de 1 época + extração + 2 sondas) | ~1 h cada |
| GBDT tradicional com HPO, CPU de 12 núcleos | ~1–2 h por conjunto de 4 tarefas |
| GBDT híbrido XGBoost na GPU (`XGB_DEVICE=cuda`) | ~1 h com HPO, contra ~4 h do LightGBM na CPU |

## Limitações deste experimento

1. **Deslocamento de distribuição.** O teste (mar–set/2020) cobre a COVID: lojas fechadas e 100% de vendas online em algumas semanas. As taxas base mudam (churn de 30% na validação para 16% no teste). A ordem entre modelos muda entre validação e teste em alguns pares: o XGBoost híbrido em churn e o backbone final.
2. **O IC do bootstrap cobre só a amostragem de clientes,** não a variação entre sementes de pré-treino nem a variação temporal. Diferenças de ±0,005 entre fusões não devem orientar decisão de produto.
3. **Ablações com 1 semente e 1 época,** e 9 delas sem os artefatos originais.
4. **Cortes de custo.** O GBDT sobre o embedding bruto multiclasse e o híbrido bruto foram cortados, pois custavam ~5× o PCA-32 e já perdiam dele em churn e online. A HPO do híbrido teve orçamento menor que a da H5 (20/12 contra 40/25 tentativas); o controle `trad_lgbm_orcamento_h7` mostrou que esse orçamento não muda o tradicional.
5. **O backbone final não passou por varredura própria.** É uma combinação de efeitos isolados, sem teste das interações (a "moderna" é a mais próxima).
6. **Escopo do cadastro.** O cadastro da H&M é pobre: FN e Active estão 65% vazios, e o CEP vem com hash. O efeito incremental pequeno do cadastro não se generaliza para bases com cadastro rico.

## Reprodução

```bash
# local (dados e features; teto de memória: scripts/rodar_limitado.sh)
python3 adapters/baixar_hm.py && python3 adapters/compactar_hm.py && python3 adapters/adapter_hm_clientes.py
# tarefas 7, 4, 5, 3, 8 do motor + features_tradicionais.py (ver GENERALIZACAO.md)
bash scripts/rodar_h5.sh                                   # H5 (CPU)
# Colab L4 (dados enviados por tar | ssh)
python3 pipeline/common/tarefa_pretreino.py configs/hm_clientes.yaml --retomar      # H3
bash scripts/rodar_h4_h9_fusao.sh                          # H4/H9
bash scripts/rodar_h6_h7.sh ; bash scripts/rodar_h7_resto.sh   # H6/H7
python3 pipeline/common/experimento_ablacao.py configs/hm_clientes.yaml <ablacoes...>   # H8
python3 pipeline/common/experimento_backbone_final.py configs/hm_clientes.yaml \
  --arq posicao=rope n_embd=256 n_layer=6 n_head=4 n_kv_heads=2 ffn=swiglu --seeds 0 1 2
GPU=1 bash scripts/rodar_h10.sh; XGB=1 XGB_DEVICE=cuda bash scripts/rodar_h10.sh; CPU=1 bash scripts/rodar_h10.sh
bash scripts/bootstrap_h10.sh
python3 pipeline/relatorios/consolidar_hm.py
```

Para não perder resultados quando o Colab cair, rode `scripts/sincronizar_colab.sh <host>` em segundo plano na máquina local.

## Fase 2 — hipóteses priorizadas (em andamento, 2026-10-22)

Todas respondem às duas hipóteses macro do projeto:
- **M1:** um modelo fundacional próprio traz benefício real para a empresa?
- **M2:** a forma de treino é flexível para a base de qualquer empresa?

O prefixo F2 evita confusão com as etapas H0–H10. Os critérios de sucesso foram escritos antes dos resultados (artigo, seção 11).

| Hipótese | Pergunta | Macro | Código |
|---|---|---|---|
| F2-H7 | Em clientes com pouco histórico (1–3 cestas), o embedding vence o GBDT? | M1 | `pipeline/common/analise_segmentos.py` |
| F2-H1 | Com 1–25% dos clientes rotulados, o embedding vence o GBDT? | M1 | `--frac_treino/--semente_amostra` em `tarefa13_treinar_fusao.py` e `tarefa14_gbdt.py`; `schema.amostra_clientes`; `scripts/f2_h1_local.sh` |
| F2-H6 | Custo (tempo, mudanças) e qualidade de uma pergunta nova (`proxima_cesta_alta`) | M1, M2 | `configs/hm_clientes_f2h6*.yaml`; `scripts/f2_h6_local.sh` |
| F2-H2 | Ajuste fino conjunto do backbone com a fusão DCNv2 (receita nuFormer) | M1 | `pipeline/common/tarefa15_ajuste_conjunto.py` |

Adiadas (ver ROADMAP): F2-H3, H4, H5, H8, H9, H10 (LLM open source) e H11 (TabPFN). A fila de GPU está em `scripts/fila_colab_fase2_base.sh` e `scripts/fila_colab_fase2.sh`.
