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

### Resultados da fase 2 (teste, avaliação única, 3 sementes/amostras)

| Hipótese | Resultado | Critério | Macro |
|---|---|---|---|
| F2-H7 clientes novos | Na 1ª compra, o híbrido ganha +0,004 em canal e +0,002 em categoria; a fusão perde em todas as faixas; o churn na 1ª compra é quase imprevisível (AUC 0,576) | formalmente atingido, sem relevância prática | M1: não sustenta |
| F2-H1 poucos rótulos | GBDT com 1% dos clientes rotulados fica a ≤0,017 do máximo; o valor do embedding na fusão é **negativo** com 1% e só aparece a partir de 5% | não atingido | M1: não sustenta |
| F2-H6 pergunta nova (`proxima_cesta_alta`) | Só configuração nos três caminhos. AUC no teste: tradicional 0,757, híbrido 0,756, fusão 0,739. Treino: 52 / 43 / 10 min | M2 atingido; M1 não | **M2 confirmada** |
| F2-H2 ajuste fino conjunto | Contra a fusão congelada: +0,010 em categoria e +0,017 em LTV. Contra o GBDT: −0,013 em churn, −0,003 em canal, −0,005 em categoria, −0,007 em LTV | não atingido, por pouco | M1: mais promissor |

Arquivos: `f2_h7_segmentos.json`, `f2_h1_curvas.json`, `f2_h6_bootstrap.json`, `f2_h2_bootstrap_3sementes.json` e `f2_h2_rodada1/`.

**Balanço:** M2 confirmada; M1 sem confirmação nesta base. O ajuste fino conjunto passa a ser o modo padrão do modelo fundacional. Próximos testes com chance real para M1:
- ajuste fino conjunto com poucos rótulos e com dados recentes;
- tarefas sequenciais sem features prontas (F2-H3);
- a base da empresa (F2-H9).

## Fase 2, segunda rodada — F2-H3, F2-H10, F2-H11 (2026-10-23)

Desenhos e critérios registrados no artigo (seção 12) antes de rodar.

### F2-H3 — recomendação do próximo artigo (MAP@12)

Semana-alvo de validação 05–11/03/2020 (antes da COVID); teste 16–22/09/2020
(última semana, a da competição). Ranqueador treinado nas 2 semanas anteriores
a cada alvo (40 mil clientes por semana). Candidatos iguais para todos os braços
(recompra 8 semanas, 100 populares, variantes de cor, comprados junto, 50
populares da faixa etária): ~125 por cliente, cobertura 11,7% (val) e 16,2% (teste).
O backbone (final_s0..s2, congelado) pontua cada candidato pelo log da
probabilidade do bloco de tokens do artigo depois do histórico; todos os
candidatos de um cliente numa passada, com máscara de atenção
(`f2h3_pontuar_fm.py`; idêntico à pontuação isolada até a 4ª casa).

| Braço | MAP@12 val | MAP@12 teste | vs R2 (teste) | IC 95% |
| --- | --- | --- | --- | --- |
| R0 populares | 0,0062 | 0,0087 | −73% | [−0,0248; −0,0227] |
| R1 recompra + populares | 0,0198 | 0,0229 | −29% | [−0,0101; −0,0089] |
| R2 LightGBM LambdaRank tradicional | 0,0283 | 0,0324 | — | — |
| R3 backbone sozinho | 0,0182 | 0,0192 | −41% | [−0,0139; −0,0126] |
| R4 híbrido | 0,0287 | 0,0329 | +1,6% | [+0,0002; +0,0008] |
| R5 LightGBM só com o backbone | 0,0209 | 0,0216 | −33% | [−0,0114; −0,0101] |

Critério (ganho ≥ 2% relativo com IC excluindo zero; forma forte R3/R5 ≥ R2):
**não confirmada** — ganho real mas de 1,6%. 90% dos artigos comprados no teste
caem no token ARTIGO_OUTROS (fora do top-2000 do treino), e os pesos do backbone
são de até 12/2019. Arquivos: `runs/hm_clientes/f2h3/resultado_{val,test}.json`.
Reprodução: `f2h3_candidatos.py` (local, 6 semanas) → `f2h3_pontuar_fm.py`
(Colab, 6 semanas × 3 sementes, ~1h40) → `f2h3_ranquear.py --fase val` → `--fase test`.

### F2-H10 — LLM open source como codificador de clientes

Amostra de 10% dos clientes (111.143 clientes; 199.146 âncoras de treino).
Todos os braços usam a mesma fusão DCNv2 (variante d), 3 sementes, e são
avaliados uma vez no teste, com o rótulo de churn já corrigido (59.541
clientes; 108.839 âncoras em churn e LTV e 149.064 em online e categoria).
AUC em churn e online; accuracy em categoria e LTV.

| Braço | Churn | Online | Categoria | LTV | Média | Média − E0 (IC 95%) |
| --- | --- | --- | --- | --- | --- | --- |
| T — GBDT tradicional | 0,8241 | 0,8432 | 0,3645 | 0,3945 | 0,6066 | +0,0118 [+0,0110; +0,0127] |
| E0 — nosso backbone (H3), congelado | 0,8125 | 0,8372 | 0,3551 | 0,3741 | 0,5947 | — |
| E1 — ajuste fino conjunto | 0,8118 | 0,8400 | 0,3641 | 0,3785 | 0,5986 | +0,0039 [+0,0031; +0,0046] |
| E2 — CoLES nos nossos tokens, só com a amostra | 0,8065 | 0,8359 | 0,3549 | 0,3680 | 0,5913 | −0,0034 [−0,0042; −0,0025] |
| E3a0 — Qwen3-Embedding-0.6B sem ajuste | 0,8029 | 0,8331 | 0,3411 | 0,3828 | 0,5900 | −0,0048 [−0,0057; −0,0038] |
| E3a — Qwen3-Embedding-0.6B com LoRA contrastivo | 0,8111 | 0,8322 | 0,3476 | 0,3842 | 0,5938 | −0,0009 [−0,0017; −0,0001] |
| E3b — Qwen3.5-0.8B-Base com LoRA de próximo token (20/11) | 0,8078 | 0,8290 | 0,3500 | 0,3877 | 0,5936 | −0,0011 [−0,0019; −0,0003] |

Critérios escritos antes de rodar:

- **Principal (algum braço com LLM supera o E0 por 0,005 na média): não
  atingido.** O E3a empata com o E0 (−0,0009). O desenho mandava rodar o E3b
  se o E3a ficasse a menos de 0,005 do E0. O E3b (rodado em 20/11) também empata
  com o E0 (−0,0011) e com o E3a (−0,0001 [−0,0008; +0,0005]).
- **Secundário (algum braço supera o GBDT em uma tarefa, ou empata em todas):
  não atingido.** O mais próximo é o ajuste fino conjunto, que empata com o
  GBDT em categoria (−0,0004 [−0,0019; +0,0011]) e fica a 0,003 em online, mas
  a 0,012 em churn e a 0,016 em LTV.
- **Custo.** Gerar os embeddings com o LLM levou 76 min para 111 mil clientes,
  ou cerca de 11,5 h de L4 por milhão de clientes. O nosso backbone leva cerca
  de 20 min para 1,18 milhão. O treino do LoRA levou 2,4 h.

Leitura:

1. **O LLM "de prateleira", sem ver nenhum dado nosso no treino, fica a 0,005
   do nosso backbone** e ganha dele em LTV (+0,009). O texto dos produtos
   carrega informação que os nossos tokens não carregam.
2. **O ajuste com LoRA soma 0,004 e leva o LLM ao empate com o nosso
   backbone,** a um custo de geração de embeddings cerca de 35 vezes maior.
3. **O CoLES nos nossos tokens perde 0,003 para o E0,** mas treinou só com a
   amostra, e o E0 com a base inteira. O teste não separa o efeito do objetivo
   do efeito da quantidade de dados.
4. **O ajuste fino conjunto segue o melhor caminho com modelo fundacional:**
   +0,004 sobre o E0 e a 0,008 do GBDT na média.
5. **O E3b, um LLM de base ajustado a prever o próximo token do histórico em
   texto, não muda o quadro.** 1.000 passos de LoRA (lote 32, até 384 tokens,
   perda de 3,08 para 0,70), embedding pela média dos estados. Ganha do E0 em
   LTV (+0,014 [+0,011; +0,016], o melhor LTV entre os LLMs) e perde em churn
   (−0,005), online (−0,008) e categoria (−0,005). Na média, empata com o E3a e
   fica a 0,013 do GBDT, sem passar dele em nenhuma tarefa. Custo: 1,8 h de
   treino e 2,3 h de extração para 111 mil clientes (cerca de 21 h de L4 por
   milhão), sem o kernel otimizado da atenção linear do Qwen3.5, que não estava
   instalado. Saem duas lições de engenharia: os logits do vocabulário inteiro
   (248 mil tokens) pedem micro-lotes no treino e `logits_to_keep=1` na
   extração. Arquivos: `runs/hm_clientes/f2h10/resultado_test_e3b.json`;
   reprodução em `scripts/fila_colab_e3b.sh`.

Arquivos: `runs/hm_clientes/f2h10/resultado_test.json`. Reprodução:
`scripts/fila_colab_f2h10.sh` e `f2h10_avaliar.py`.

## Fase 3 — uma solução, vários problemas (desde 2026-11-13)

Plano, margens de empate e critérios em `PLANO_FASE3.md`, escritos antes dos resultados.

### F3-H2 — clusterização com o motor atual (backbone H3)

Uma âncora por cliente; K-means (K = 8, 3 sementes) ajustado em 600 mil clientes
de treino e aplicado a 742 mil clientes de teste. O índice é a média, sobre os
4 desfechos, da parcela da incerteza do desfecho explicada pelo segmento. A
forma de padronização de cada espaço foi escolhida na validação.

| Espaço | Índice (IC 95%) | ARI | Churn | Categoria | LTV | Online |
| --- | --- | --- | --- | --- | --- | --- |
| T1 — RFM clássico (4 variáveis) | 0,0717 [0,0710; 0,0724] | 0,973 | 0,159 | 0,002 | 0,103 | 0,023 |
| T2 — 97 agregados em 32 componentes | 0,0634 [0,0628; 0,0641] | 0,492 | 0,130 | 0,010 | 0,077 | 0,037 |
| E — embedding em 32 componentes | 0,0420 [0,0416; 0,0424] | 0,713 | 0,136 | 0,005 | 0,001 | 0,026 |
| H — T2 e E juntos | 0,0506 [0,0501; 0,0512] | 0,919 | 0,145 | 0,004 | 0,032 | 0,022 |

Critério (índice de E ou H em pelo menos 95% do melhor T, com ARI no máximo
0,05 abaixo): **não empata.** O embedding fica em 59% do índice do RFM e o
híbrido em 71%. Na validação a distância é maior: 17% e 19%.

**Por quê: o embedding organiza os clientes pelo calendário.** Diagnóstico
exploratório (`f3h2_diagnostico_calendario.py`): os 8 segmentos do embedding
explicam 57% da incerteza do **mês da âncora** na validação e 71% no teste; os
do RFM explicam 0,5%. O embedding quase não explica o LTV (0,001), porque não
soma nem conta. O número de churn do embedding no teste (0,136, contra 0,015 na
validação) vem em boa parte do achado abaixo, e não de separar clientes.

Arquivos: `runs/hm_clientes/f3h2/resultado_{val,test}.json` e
`diagnostico_calendario_{val,test}.json`. Reprodução:
`f3h2_clusterizacao.py --fase val` e depois `--fase test`.

### Achado: censura unilateral do rótulo de churn no fim do teste

A receita de churn (`tarefa8_rotulos.py`) marca como determinável toda âncora
que tem uma cesta seguinte, mesmo com a janela de 90 dias ainda aberta no fim
do dataset. Nos últimos 90 dias do teste, portanto, uma âncora só entra se o
cliente voltou, sempre com rótulo 0. É uma seleção pelo próprio desfecho.

| Âncoras de teste com rótulo de churn | Quantidade | Taxa de churn |
| --- | --- | --- |
| Janela de 90 dias fechada | 1.087.108 | 25,4% |
| Janela aberta (últimos 90 dias) | 633.352 | 0,0% |

A taxa de 16% publicada para o teste vinha dessa mistura, e não da COVID.
Treino e validação não são afetados: todas as janelas deles estão fechadas.

AUC de churn no teste, recalculada com as predições guardadas
(`verificar_censura_churn.py`; conjunto das 3 sementes):

| Modelo | Como publicada | Só com janela fechada |
| --- | --- | --- |
| GBDT tradicional (LightGBM) | 0,8244 | 0,8291 |
| GBDT híbrido (LightGBM) | 0,8206 | 0,8245 |
| GBDT só embedding (PCA-32) | 0,7702 | 0,7914 |
| Fusão completa (backbone H3) | 0,8133 | 0,8125 |

A ordem entre os modelos não muda, mas as distâncias mudam.

**Correção aplicada em 2026-11-14.** A receita agora só considera determinável
a âncora com a janela fechada, como já fazia a de LTV (`tarefa8_rotulos.py`).
Os rótulos, as features e a amostra da F2-H11 foram corrigidos por
`corrigir_censura_churn.py` (633.352 rótulos anulados; treino e validação
idênticos). A AUC do baseline causal no teste passa de 0,744 para 0,728.

Deltas de churn no teste recalculados (bootstrap por cliente, B = 300;
`h10_bootstrap_churn_corrigido.json`):

| Comparação | Publicado | Corrigido (IC 95%) |
| --- | --- | --- |
| Fusão completa (H3) − GBDT tradicional | −0,0111 | −0,0166 [−0,0171; −0,0161] |
| GBDT híbrido − tradicional | −0,0037 | −0,0046 [−0,0047; −0,0044] |
| GBDT híbrido com backbone final − tradicional | −0,0055 | −0,0012 [−0,0013; −0,0012] |
| GBDT só embedding − tradicional | −0,0542 | −0,0377 [−0,0383; −0,0371] |
| Efeito do cadastro no GBDT híbrido | +0,0008 | +0,0009 [+0,0008; +0,0010] |

F2-H7, churn por faixa de histórico (`f2_h7_segmentos.json`; o original ficou
em `f2_h7_segmentos_rotulo_original.json`):

| Faixa | AUC do tradicional (publicada → corrigida) | Fusão − tradicional | Híbrido − tradicional |
| --- | --- | --- | --- |
| 1 cesta | 0,576 → 0,604 | −0,030 → −0,005 | +0,000 → −0,005 |
| 2 a 3 cestas | 0,669 → 0,746 | −0,060 → −0,084 | −0,015 → −0,012 |
| 4 a 10 cestas | 0,577 → 0,649 | −0,000 → −0,018 | −0,010 → −0,009 |
| 11 ou mais | 0,729 → 0,753 | −0,024 → −0,014 | −0,002 → −0,009 |

F2-H1, churn do GBDT com parte dos clientes rotulados (média das 3 amostras;
`f2_h1_curvas_churn_corrigido.json`):

| Fração rotulada | Tradicional (publicada → corrigida) | Híbrido (publicada → corrigida) |
| --- | --- | --- |
| 1% | 0,8197 → 0,8178 | 0,8179 → 0,8179 |
| 5% | 0,8212 → 0,8245 | 0,8210 → 0,8229 |
| 10% | 0,8235 → 0,8264 | 0,8202 → 0,8236 |
| 25% | 0,8218 → 0,8279 | 0,8198 → 0,8252 |
| 100% | 0,8242 → 0,8291 | 0,8205 → 0,8245 |

Com o rótulo corrigido a curva fica monótona, como se espera. Com 1% dos
clientes rotulados o tradicional fica a 0,011 do máximo em churn (publicado:
0,005).

**O que muda nas conclusões.** Nenhum veredito das fases 1 e 2 se inverte: o
GBDT tradicional continua o melhor em churn e o embedding continua sem
agregar. A distância da fusão em churn é maior que a publicada (0,017, e não
0,011).

**Sem recálculo possível,** porque as predições se perderam com o Colab: o
ajuste fino conjunto (F2-H2), as curvas de fusão da F2-H1, as variantes de
fusão da H9, o backbone final na fusão e os modelos XGBoost. Os números de
churn desses itens seguem com o rótulo antigo e estão marcados assim.

### F3-H2 refeita com o rótulo de churn corrigido

Mesmo protocolo, com o rótulo corrigido e mais um espaço: o embedding do
backbone final (RoPE), que estava em cache local. Teste, K = 8:

| Espaço | Índice | Razão sobre o RFM | ARI |
| --- | --- | --- | --- |
| T1 — RFM clássico | 0,0695 | — | 0,973 |
| T2 — 97 agregados em 32 componentes | 0,0599 | 86% | 0,492 |
| E — embedding do backbone H3 | 0,0088 | 13% | 0,713 |
| E — embedding do backbone final | 0,0058 | 8% | 0,628 |
| H — agregados e embedding H3 | 0,0292 | 42% | 0,919 |
| H — agregados e embedding final | 0,0187 | 27% | 0,575 |

O veredito não muda (**não empata**) e a distância real é maior que a do
primeiro teste: sem o defeito do rótulo, o embedding explica 13% do que o RFM
explica, perto dos 17% da validação. Os resultados com o rótulo original
ficaram em `runs/hm_clientes/f3h2/resultado_{val,test}_rotulo_original.json`.

### F3-H1 — propensão a desconto com o motor atual

Pergunta nova, criada só com configuração: a próxima cesta tem algum item
comprado pelo menos 10% abaixo do preço de referência do artigo? A definição
foi calibrada só no treino (`runs/hm_clientes/f3h1/relatorio.json`): 50% das
cestas no treino, 57% na validação e 52% no teste.

| Braço | AUC no teste | Diferença para T (IC 95% por cliente) |
| --- | --- | --- |
| T — GBDT tradicional, com o histórico de desconto | 0,6842 | — |
| H — GBDT híbrido (features e embedding H3) | 0,6858 | +0,0015 [+0,0014; +0,0017] |
| E — fusão com o backbone H3 (tarefa única, 3 sementes) | 0,6667 | −0,0175 [−0,0181; −0,0170] |

**Veredito: critério atendido, pelo braço H.** O critério escrito antes de
rodar pedia E **ou** H a até 0,01 de T. O híbrido empata (e fica um pouco
acima). A fusão sozinha fica 0,0175 abaixo, fora da margem: o embedding H3
não carrega o histórico de desconto que o GBDT recebe pronto (a participação
de cestas com desconto, por janela). É o mesmo padrão das 4 tarefas antigas
(E1/E0 abaixo de T; H empata). M2 também atendido: nenhuma linha de código
específica do problema, além da coluna derivada no adapter.

Custo: 17 min de CPU para cada GBDT (40 tentativas de HPO) e cerca de 2 min de
L4 para cada semente da fusão, depois de extraídos os embeddings (23 min,
uma vez para todas as perguntas). Arquivos em `runs/hm_clientes_f3h1/`
(`f3h1_bootstrap_completo.json`); reprodução: `scripts/f3_h1_local.sh` e o
bloco 2 de `scripts/fila_colab_f3b.sh`.

### F2-H11 — modelos tabulares pré-treinados (TabPFN-3.5, TabICL): resultado no teste

Pergunta: um modelo tabular pré-treinado, **sem ajuste de hiperparâmetros**,
empata com o LightGBM? Critério do plano: se empatar, vira candidato a cabeça
leve da bateria. Mesmas features do braço T (97 tradicionais; o híbrido soma o
embedding H3 em PCA-32), mesmas âncoras de teste da F2-H10, 1 semente por
braço, bootstrap por cliente (300 reamostras). O "contexto" é o número de
âncoras de treino que o modelo vê; o LightGBM foi treinado com as mesmas
amostras, para separar o efeito do modelo do efeito do volume.

| Braço | Churn (AUC) | Próxima online (AUC) | Próxima categoria (acc.) | LTV 90d (acc.) | Média |
|---|---|---|---|---|---|
| T — LightGBM, 300 mil âncoras, HPO de 40 tentativas, 3 sementes | 0,8241 | 0,8432 | 0,3645 | 0,3945 | 0,6066 |
| LightGBM, 50 mil âncoras | 0,8187 | 0,8426 | 0,3597 | 0,3909 | 0,6029 |
| **TabPFN-3.5, 50 mil âncoras** | 0,8247 | 0,8440 | 0,3678 | 0,3959 | **0,6081** |
| TabPFN-3.5 híbrido, 50 mil | 0,8252 | 0,8442 | 0,3620 | 0,3989 | 0,6076 |
| TabICL, 50 mil | 0,8218 | 0,8444 | 0,3606 | 0,3898 | 0,6042 |
| LightGBM, 10 mil | 0,8131 | 0,8375 | 0,3512 | 0,3850 | 0,5967 |
| TabPFN-3.5, 10 mil | 0,8204 | 0,8437 | 0,3627 | 0,3946 | 0,6053 |
| TabICL, 10 mil | 0,8154 | 0,8427 | 0,3594 | 0,3875 | 0,6013 |

Diferenças (IC 95% por cliente):

| Comparação | Churn | Próxima online | Próxima categoria | LTV 90d | Média |
|---|---|---|---|---|---|
| TabPFN 50 mil − LightGBM 50 mil | +0,0060 | +0,0014 | +0,0081 | +0,0050 | +0,0051 [+0,0045; +0,0057] |
| TabPFN 50 mil − T completo | +0,0006 | +0,0008 | +0,0033 | +0,0014 | +0,0015 [+0,0009; +0,0022] |
| TabICL 50 mil − LightGBM 50 mil | +0,0032 | +0,0018 | +0,0009 | −0,0011 | +0,0012 [+0,0007; +0,0019] |
| TabPFN 10 mil − LightGBM 10 mil | +0,0073 | +0,0062 | +0,0115 | +0,0096 | +0,0087 [+0,0080; +0,0094] |

**Veredito: critério atendido.** Os dois modelos empatam com o LightGBM em
todas as tarefas, sem HPO. O TabPFN-3.5 vai além: com 50 mil âncoras supera o
LightGBM treinado com as mesmas 50 mil e empata com o T completo (seis vezes
mais dados, HPO e 3 sementes), ligeiramente acima na média. A vantagem cresce
quando há pouco dado (+0,009 com 10 mil), o que interessa a perguntas novas
com poucos rótulos. O embedding H3 não acrescenta nada ao TabPFN (híbrido
0,6076 contra 0,6081).

**Custos e limites práticos.**
- Tempo por configuração: 23 a 109 min de L4 no TabPFN e 11 a 42 min no
  TabICL, contra menos de 1 min do LightGBM.
- O TabPFN-3.5 precisa de uma chave da Prior Labs e da licença aceita; para
  uso comercial, a licença tem de ser revista antes da entrega à empresa.
- O TabICL morreu por falta de RAM (52 GB) com 199 mil âncoras, duas vezes, e
  com 50 mil no híbrido (15/11). Fica registrado como limite: acima de ~50 mil
  âncoras de contexto, ele não cabe numa máquina comum.
- O TabPFN com 199 mil âncoras não terminou numa sessão; roda no fim da fila,
  se sobrar tempo, fora desta avaliação.

**Consequência para a fase 3.** O TabPFN-3.5 entra como cabeça leve candidata
na bateria (braços E e H), ao lado da fusão e do LightGBM. É coerente com a
promessa do produto: pergunta nova vira uma cabeça sem ajuste sobre features
e vetores já calculados. Arquivo: `runs/hm_clientes/f2h11/resultado_test.json`.


### Motor v2 na H&M — preparo e um defeito achado no primeiro pré-treino (15/11)

**Serialização.** A H&M entra no v2 com um atalho de 9 linhas no config: uma
posição por cesta (cliente e dia), o artigo como item do catálogo inteiro e os
5 atributos do artigo. São 9,08 milhões de posições e 31,8 milhões de linhas,
serializadas em 50 s na máquina local. O maior histórico tem 427 cestas, então
todo cliente cabe numa janela de 512: **100% das âncoras sem truncamento**, que
é o critério de cobertura da F3-H3 (no v1, 11 tokens por item, a janela de 1024
cortava o histórico de parte dos clientes). Os testes obrigatórios
(`v2_testes.py`) passam na H&M.

**Custo.** 6 min de L4 por época (444 passos de 0,81 s, 15,7 GB de memória),
contra 64 min por época do backbone H3 do v1. É menos de um décimo, e o
critério de custo da F3-H3 pede no máximo um quinto.

**Defeito: o modelo aprendia a não recomendar lançamentos.** Na primeira
rodada, a perda de itens na validação ficou em 0,996 e depois 1,014, na escala
em que 1,0 é escolher ao acaso entre os 104 mil artigos; no treino ela estava
em 0,82. A causa é a troca de coleção: 27% das linhas compradas na validação e
**69% das do teste** são de artigos que não existiam no treino. Como o softmax
cobre o catálogo inteiro, esses artigos só apareciam no treino como resposta
errada, e o viés de cada um afundava.

A correção, no lado da saída (`ModeloV2.saida_catalogo`):

- no treino, os itens que só aparecem depois do corte ficam fora do softmax:
  ainda não existem no período de treino;
- na hora de pontuar, um item novo é representado só pelos atributos, com um
  viés compartilhado de "item novo";
- esse viés e o caminho pelos atributos são aprendidos desligando ao acaso,
  no treino, o vetor próprio de 10% dos itens vistos (o desenho já previa isso
  na entrada; faltava na saída).

O pré-treino foi refeito do zero com a correção. A rodada com o defeito está
guardada em `runs/hm_clientes/v2_pretreino_sem_frio/`.

**Por que importa para a empresa.** Perfumaria e beleza lançam produtos a cada
ciclo. Sem esse tratamento, a recomendação direta do modelo (F3-H4) nunca
indicaria um lançamento, e o embedding do cliente aprenderia a ignorá-los.

### F3-H3 — o evento composto (motor v2) contra o v1 na H&M: resultado no teste

Mesmo decoder (256×6, 11,3 milhões de parâmetros), mesmas âncoras e a mesma
fusão da H4 (embedding + cadastro + baselines causais), 3 sementes, rótulo de
churn corrigido. O pré-treino v2 inclui a correção do item novo (seção
anterior) e o critério de época sem as janelas que não cabem na validação.
Melhor época: 7 de 8.

**Critério, escrito antes de rodar, e resultado.**

| Critério | Exigência | Resultado | |
|---|---|---|---|
| Custo | ≤ 1/5 do tempo de GPU por época do v1 | 6,9 min contra 64 min (0,11×) | atendido |
| Cobertura | 100% das âncoras numa janela, sem truncamento | 100% (maior histórico: 427 cestas; janela 512) | atendido |
| Qualidade | fusão v2 a até 0,005 da fusão v1 nas 4 tarefas | ganha em 2, perde em 2 | **não atendido** |

| Tarefa | GBDT tradicional (T) | Fusão v1 (H4) | Fusão v2 | v2 − v1 (IC 95%) | v2 − T |
|---|---|---|---|---|---|
| Churn 90 dias (AUC) | 0,8291 | 0,8125 | **0,8194** | **+0,0069** [+0,0066; +0,0074] | −0,0097 |
| Próxima categoria (acc.) | 0,3727 | 0,3585 | 0,3339 | **−0,0246** [−0,0252; −0,0239] | −0,0388 |
| LTV 90 dias (acc.) | 0,3991 | 0,3747 | **0,3853** | **+0,0106** [+0,0099; +0,0114] | −0,0139 |
| Próxima online (AUC) | 0,8448 | 0,8380 | 0,8280 | **−0,0100** [−0,0102; −0,0098] | −0,0168 |

**Leitura.**

- **O v2 ganha onde os alvos dele apontam.** O pré-treino do v2 prevê o
  intervalo até o próximo pedido, o valor e as contagens e gastos futuros, e
  é exatamente em churn e LTV que ele supera o v1. Em churn, a fusão v2 fica
  a 0,0097 do GBDT tradicional: **pela primeira vez uma fusão empata com T em
  churn** pela margem de 0,01 (a v1 ficava a 0,0166).
- **Perde em composição e canal.** A próxima categoria depende da composição
  da cesta, que o v2 só aprende pelo artigo exato, e o artigo troca de
  coleção. O canal ficou fora dos alvos de propósito (teste de transferência).
- **Na validação as quatro passavam.** Validação (dez/2019 a mar/2020): churn
  +0,009, categoria −0,004, LTV +0,044, online −0,004. A diferença aparece no
  teste.
- **Não é envelhecimento; é a mudança de regime.** Mês a mês no teste, a perda
  em categoria é maior no início (−0,031 em março, −0,038 em abril, −0,033 em
  maio de 2020, com as lojas fechadas) e cai até −0,006 em setembro; em canal,
  de −0,015 em março para −0,006 a partir de junho. Se fosse envelhecimento da
  representação, a perda cresceria com a distância do corte. Churn e LTV ficam
  melhores no v2 em todos os meses.

**Consequência.** O v2 resolve a escala (10× mais barato por época e sem
truncamento) e melhora churn e LTV. Mas não preserva a qualidade nas 4
tarefas, e o critério não foi atendido como escrito. A variante F3-H3b
(composição do próximo pedido por atributo, registrada em `PLANO_FASE3.md`
depois deste resultado) ataca a perda em categoria. Arquivos:
`runs/hm_clientes/f3h3/bootstrap_test.json`, `runs/hm_clientes/v2_pretreino/v2_relatorio.json`.

### F3-H3b — v2 com alvos de composição por atributo: resultado no teste

Variante registrada **depois** de ver o teste da F3-H3 (evidência mais fraca;
ver `PLANO_FASE3.md`). Uma mudança só: o pré-treino também prevê, para cada
atributo do catálogo (índice, grupo, tipo, cor, linha), a distribuição dos
valores nas linhas do próximo pedido. Canal continua fora dos alvos. Mesmo
desenho: 8 épocas (63 min de GPU, dividindo a placa com outra fila), melhor
época 7, mesmas âncoras e fusão, 3 sementes. Perda de itens na validação:
0,9229 (v2: 0,928).

**Critério, escrito antes de rodar:** a até 0,005 da fusão v1 nas 4 tarefas
e, além disso, a até 0,005 do v2 em churn e LTV.

| Tarefa | T | v1 | v2 | v2b | v2b − v1 (IC 95%) | v2b − v2 | v2b − T |
|---|---|---|---|---|---|---|---|
| Churn (AUC) | 0,8291 | 0,8125 | 0,8194 | 0,8148 | +0,0024 [+0,0020; +0,0028] | −0,0046 | −0,0143 |
| Próxima categoria (acc.) | 0,3727 | 0,3585 | 0,3339 | **0,3559** | −0,0025 [−0,0030; −0,0019] | **+0,0220** | −0,0168 |
| LTV 90 dias (acc.) | 0,3991 | 0,3747 | 0,3853 | 0,3795 | +0,0048 [+0,0041; +0,0055] | −0,0058 | −0,0196 |
| Próxima online (AUC) | 0,8448 | 0,8380 | 0,8280 | 0,8288 | **−0,0092** [−0,0094; −0,0089] | +0,0008 | −0,0160 |

**Resultado: não atendido.** Duas condições falham: próxima online fica a
0,0092 do v1 (o limite era 0,005), e o LTV perde 0,0058 do ganho do v2 (o
limite era 0,005).

**Leitura.**

- **Os alvos de composição fecham a perda em categoria.** De −0,025 para
  −0,0025 contra o v1. A hipótese da F3-H3 se confirma: o v2 perdia categoria
  porque só aprendia a composição pelo artigo exato, e o artigo troca de
  coleção; o atributo não troca.
- **Canal não melhora, como esperado.** Canal ficou fora dos alvos de
  propósito, e a perda em próxima online fica igual à do v2 (−0,009 contra
  −0,010). É a parte que o pré-treino não ensina e que a fusão sozinha não
  recupera.
- **O ganho em churn e LTV cai pela metade.** A mesma representação de 256
  dimensões passa a dividir capacidade com a composição. Churn: +0,007 → +0,002
  sobre o v1; LTV: +0,011 → +0,005. E o empate com T em churn, que o v2 tinha
  (−0,0097), se perde (−0,0143).
- **Pela margem geral do projeto (0,01), o v2b empata com o v1 nas 4 tarefas**
  com um oitavo do custo por época e sem truncar histórico. O critério desta
  hipótese, mais rígido (0,005), não é atendido.

**Consequência.** Nenhuma das duas variantes domina: o v2 é melhor para
churn e LTV; o v2b, para categoria; nenhuma resolve canal. Para a empresa,
a escolha depende de qual problema pesa mais, ou de manter as duas cabeças
de pré-treino. Arquivos: `runs/hm_clientes/f3h3/bootstrap_v2b.json`,
`runs/hm_clientes/v2b_pretreino/v2_relatorio.json`.

### F3-H8 — formas extremas e estrutura plantada, na base sintética grande (16/11)

Base sintética de beleza B2B, tamanho grande (`BASE_SINTETICA_BELEZA.md`):
100 mil clientes, 2,25 milhões de pedidos, 45 milhões de linhas, catálogo de
30 mil SKUs, pedidos de até 897 linhas, clientes com até 5.569 posições
(pedidos + sessões de navegação). Motor v2 com janela de 8192 posições e
todas as fontes (pedido, navegação, crédito, incentivo), 3 épocas.

| Critério | Exigência | Resultado | |
|---|---|---|---|
| Forma: cobertura | 100% dos eventos no contexto | 100% dos clientes inteiros numa janela (maior: 5.569 de 8.192); 45.190.593 linhas serializadas = 45.190.593 na fonte | atendido |
| Forma: custo por pedido | não cresce além do linear com o tamanho do pedido | tempo = 0,078 s + 0,00051 s × linhas por pedido (R² 0,998; lote fixo de 2.048 posições, 1 a 900 linhas); custo por linha cai de 42 para 0,30 ms por mil linhas; memória linear (~3 KB por linha) | atendido |
| Custo: projeção | o tempo medido permite projetar | 24.700 posições/s de treino numa L4 (3 épocas em 6,7 min) | atendido |
| Estrutura: segmentos | ARI de E ≥ ARI de T (margem de 5%) | E 0,095 contra T 0,451 (RFM 0,010) | **não atendido** |
| Estrutura: compra conjunta | recomendação direta acima do R1 | v2 0,0006 contra R1 0,0175 (98.569 casos) | **não atendido** |

Recomendação (MAP@12 do primeiro pedido em 21 dias a partir de 06/01/2025,
10.853 clientes): POP 0,0055, R1 (recompra) 0,284, v2 0,0077.

**Projeção para a empresa.** O custo cresce com o número de posições
(pedidos + sessões), não com o tamanho dos pedidos. Na proporção da base
grande (47 posições por cliente), 2 milhões de clientes dariam cerca de 95
milhões de posições, 66 milhões delas no treino: cerca de 45 minutos por
época numa L4, ou 2h15 para 3 épocas. É uma projeção linear; o número real
depende de quantas posições por cliente a base da empresa tiver.

**Leitura.**

- **A escala está resolvida.** Pedidos de centenas de linhas e clientes com
  milhares de pedidos cabem sem cortar nada, e o custo é linear. O primeiro
  contato com essas formas não será na empresa.
- **O embedding não separa os segmentos plantados.** Os segmentos são
  definidos pelo mix de categorias e pela faixa de preço. As features
  tradicionais trazem a participação de cada categoria explicitamente; no
  embedding, o que mais varia é o ritmo e a recência das compras (o que o
  pré-treino mais pede para prever), e o KMeans agrupa por isso.
- **A recomendação direta não personaliza.** Para 82 mil clientes, o v2
  recomenda só 136 itens distintos, e só 11% das recomendações são itens que o
  cliente já comprou; o R1, só com a recompra, chega a 0,284. A perda de itens
  na validação fica em 0,91 do sorteio uniforme. O pedido entra no modelo como
  a média das linhas, o que dilui a identidade de cada item, e a saída não tem
  como copiar do histórico. É a mesma falha da F2-H3 na H&M (−41% contra o R1),
  agora com estrutura conhecida: não é falta de sinal na base.

**Consequência.** A F3-H8 confirma o v2 como motor de escala, mas mostra
duas lacunas de representação que o caminho tradicional não tem: a
preferência por categoria e a recompra. A F3-H4 (recomendação direta) não deve
rodar com o motor como está: precisa de um termo que leve o histórico do
próprio cliente para a saída. Arquivos: `runs/beleza_b2b_grande/f3h8/custo.json`,
`runs/beleza_b2b_grande/f3h7/segmentos.json`,
`runs/beleza_b2b_grande/f3h7/recomendacao.json`,
`runs/beleza_b2b_grande/pre_v2/v2_relatorio.json`.

### F3-H7, camada 1 — todos os tipos de dado do mesmo cliente, na base sintética pequena (17/11)

Base sintética de beleza B2B, tamanho pequeno: 15 mil clientes, 336 mil
pedidos, 7,2 milhões de linhas, mais navegação (375 mil sessões), crédito e
incentivo. Motor v2 (janela 512, 12 épocas) com todas as fontes e, depois,
retirando uma fonte por vez; fusão com 3 sementes. A régua T (LightGBM, 40
tentativas de HPO, 3 sementes) recebe os agregados genéricos de todas as fontes
(`fontes_extras: v2`). H = T + embedding v2 em PCA-32.

**M2 (só configuração): atendido.** As quatro fontes entraram com um bloco de
YAML cada, sem mudança no motor; os formatos evento com itens (pedido), evento
simples (navegação) e estado (crédito, incentivo) foram aceitos.

**M1, com todas as fontes (teste; margem 0,01 e 5% em MAP/ARI):**

| Problema | T | E (fusão v2) | H (híbrido) | E − T | H − T |
|---|---|---|---|---|---|
| Churn 90 dias (AUC) | 0,8750 | 0,8648 | 0,8740 | −0,0102 | **−0,0010** |
| Próxima com desconto (AUC) | 0,8492 | 0,8442 | 0,8491 | −0,0051 | **−0,0002** |
| LTV 90 dias (acc.) | 0,5523 | 0,5410 | 0,5509 | −0,0113 | **−0,0014** |
| Próxima online (AUC) | 0,6633 | 0,6559 | 0,6635 | −0,0073 | **+0,0002** |
| Sem pedido em 7 dias (AUC) | 0,7654 | 0,7577 | 0,7651 | −0,0077 | **−0,0003** |
| Segmentos (ARI contra o gabarito) | 0,423 | 0,105 | 0,398 | −75% | −5,9% |
| Recomendação (MAP@12; T = R1) | 0,300 | 0,036 | — | −88% | — |

H empata com T nas 5 tarefas supervisionadas. Em segmentos, H fica a 5,9%
de T (a margem era 5%), e E fica longe. Em recomendação, o modelo fica muito
abaixo da recompra. **Não atendido como escrito** (segmentos e recomendação).

**M1, retirada de cada fonte (E sem a fonte − E com todas; teste):**

| Fonte retirada | Problema que depende dela (gabarito) | Queda | Maior queda em outro problema |
|---|---|---|---|
| Navegação | recomendação | MAP@12 +0,0005 (nenhuma) | online −0,0071, churn −0,0059, 7 dias −0,0050 |
| Crédito | churn | −0,0025 | LTV −0,0089 |
| Incentivo | sem pedido em 7 dias | −0,0022 | LTV −0,0102 |
| Desconto | propensão a desconto | −0,0012 | LTV −0,0043 |

Todas as quedas são significativas (IC 95% sem o zero), mas nenhuma passa a
margem de 0,01 no problema esperado. **Não atendido.**

**Leitura.**

- **O motor recebe todos os tipos de dado só com configuração** e usa cada
  um: toda retirada piora alguma coisa, e a navegação é a fonte que mais pesa
  (piora churn, canal e pedido em 7 dias).
- **O efeito plantado de crédito e incentivo é pequeno também para T.** Entre
  as 12 features mais importantes de T, nenhuma é de crédito no churn, e
  nenhuma é de incentivo no pedido em 7 dias. O ritmo de compra domina. O
  critério supunha que a fonte "dona" de cada problema pesasse mais do que
  pesa no gerador. Falta o controle: as mesmas ablações em T.
- **Segmentos e recomendação repetem a F3-H8.** O embedding não carrega o mix
  de categorias, e o modelo não personaliza a recomendação (por isso a
  navegação não mexe no MAP: o modelo nem usa o histórico do cliente para
  recomendar).
- **H é o caminho que funciona hoje:** empata com T em todas as tarefas
  supervisionadas, com o embedding respondendo por 8% a 10% da importância.

Arquivos: `runs/beleza_b2b_pequeno/f3h7/` (`bootstrap_test.json`,
`segmentos.json`, `recomendacao.json`), `runs/beleza_b2b_pequeno/gbdt/f3h7_hibrido_lgbm/relatorio.json`.

**Controle das ablações em T (19/11).** A mesma retirada de fonte na régua T
(sem as features da fonte; hiperparâmetros congelados do T completo; os
baselines causais de cada tarefa ficam, como na fusão de E). Queda no teste,
IC 95% por bootstrap:

| Fonte retirada | Problema que depende dela | Queda em T | Queda em E |
|---|---|---|---|
| Navegação | próxima online* | −0,0082 | −0,0071 |
| Crédito | churn | −0,0019 | −0,0025 |
| Incentivo | sem pedido em 7 dias | −0,0002 (n.s.) | −0,0022 |
| Desconto | propensão a desconto | −0,0016 | −0,0012 |

\*T não recomenda; para a navegação, a tarefa supervisionada mais ligada a ela.

- **Nenhuma retirada passa de 0,01 nem em T.** O gerador planta efeitos
  pequenos dessas fontes: o ritmo de compra domina todas as tarefas. O
  critério da F3-H7 ("a retirada derruba o problema por mais que a margem")
  pedia um efeito que a base não tem; nem o caminho tradicional o mostra.
- **E reage às fontes tanto quanto T, ou mais.** Nos problemas "donos" de
  cada fonte, as quedas de E e T são da mesma ordem (incentivo: só E reage).
  Fora deles, E reage mais: sem crédito ou sem incentivo, o LTV de E cai
  0,009 a 0,010, e o de T não se move. A representação usa as fontes de forma
  mais espalhada.
- **Veredito da F3-H7 (camada 1), revisto com o controle.** O critério das
  ablações, como escrito, não é atendido, mas também não seria por T: o
  defeito é de calibração do gerador, não do motor. Para a camada real
  (Synerise), o critério de navegação continua valendo como escrito.

Arquivo: `runs/beleza_b2b_pequeno/f3h7/controle_T.json`.

### Errata da F2-H3: o R1 deixava sem lista os clientes sem recompra (19/11)

Ao refazer o R1 para a F3-H4, achamos um defeito em `f2h3_ranquear.py`: a
função que completa as listas com os populares só completava os clientes que
já tinham alguma recompra candidata. Os outros (41% dos clientes na validação,
39% no teste) ficavam com a lista vazia e AP zero. Corrigido, o R1 sobe de
0,0198 para **0,0229** na validação e de 0,0229 para **0,0259** no teste. Os
outros braços da F2-H3 não mudam: todos os clientes tinham candidatos
populares. As conclusões da F2-H3 ficam: o R3 (backbone sozinho, 0,0192) segue
abaixo do R1, agora por mais, e o R2 (0,0324) segue acima.

### F3-H4 — recomendação direta do motor v2, com o backbone atualizado até a véspera (20/11)

Protocolo da F2-H3: semana-alvo de validação 05–11/03/2020 e de teste
16–22/09/2020; MAP@12 sobre os clientes que compraram na semana (73.571 e
68.984). Para cada semana, o v2 foi pré-treinado do zero com os eventos até a
véspera (8 épocas, ~1h30 numa L4), com os artigos vendidos até o corte tratados
como conhecidos (85 mil na validação, 104 mil no teste). As recomendações saem
do catálogo inteiro. Validação com 1 semente; teste com 3.

Braços (registrados antes de rodar):
- **M**: só o modelo: os 12 artigos de maior probabilidade no catálogo.
- **MH**: o modelo mais o histórico do próprio cliente:
  logp + beta × log(1 + nº de compras do artigo) + gama / (1 + dias desde a última / 7).
  Na validação, a grade escolheu beta = 3 e gama = 8 (gama no limite da grade).

| Braço | MAP@12 val | MAP@12 teste | vs R1 corrigido (IC 95%) | % do R2 |
|---|---|---|---|---|
| R1 recompra + populares (corrigido) | 0,0228 | 0,0259 | — | 80% |
| **M** só o modelo | 0,0059 | 0,0075 | −0,0184 [−0,0192; −0,0175] | 23% |
| **MH** modelo + histórico | 0,0246 | **0,0271** | **+0,0012** [+0,0006; +0,0017] | 84% |
| R2 LambdaRank tradicional (F2-H3) | 0,0283 | 0,0324 | — | 100% |
| R4 híbrido (F2-H3) | 0,0287 | 0,0329 | — | 102% |

Sementes no teste: M 0,0071–0,0079; MH 0,0268–0,0274.

**Critério:** necessário (acima do R1) **atendido só pelo MH**; empate (95% do
R2) **não atendido** (84%); forma forte, não. Como o plano previa, a
recomendação segue híbrida.

**Leitura.**

- **O modelo sozinho não recomenda.** Com o catálogo inteiro e o backbone
  atualizado até a véspera, M fica em 0,0075, abaixo até dos populares da
  semana anterior (0,0087). É o mesmo comportamento da base sintética
  (F3-H8): o v2 aprende a popularidade média do período de treino, e na moda
  a lista de populares troca de uma semana para outra. E o pedido entra no
  modelo como a média das linhas, o que dilui a identidade de cada artigo.
- **Somado ao histórico, ele acrescenta.** MH passa a recompra pura em 4,6%
  (+0,0012, IC sem o zero). A pontuação do modelo ajuda a ordenar os artigos
  já comprados e a escolher o que completa a lista melhor que os populares.
- **Mas fica longe do ranqueador tradicional** (84% do R2), que usa
  candidatos de várias fontes e 30 features por par cliente-artigo.
- **Para a empresa:** a recomendação continua sendo o problema em que a
  solução única não basta. O caminho é o da F2-H3: o modelo como uma feature
  a mais do ranqueador (R4 ficou 1,6% acima do R2). Um mecanismo de cópia do
  histórico na saída do modelo seria o próximo passo de arquitetura, mas não
  está no plano.

Arquivos: `runs/hm_clientes/f3h4/resultado.json`,
`runs/hm_clientes/f3h4_rec_{val,test}_s*/relatorio_*.json`.

### F3-H5 — embedding congelado do v2 com cabeça leve, dentro e fora dos alvos do pré-treino (20/11)

Pergunta: o vetor do v2, sem ajuste fino, empata com o LightGBM de cada
pergunta quando recebe só uma cabeça leve? O churn é da família dos alvos do
pré-treino (intervalo, valor e contagens futuras). Canal e desconto nunca
foram alvos: medem transferência. Backbone `v2_pretreino` (o da F3-H3),
cabeça DCNv2 da H4 treinada uma pergunta por vez, 3 sementes, teste avaliado
uma vez, bootstrap por cliente.

| Pergunta | Grupo | T (LightGBM) | E: cabeça de tarefa única | E − T (IC 95%) | Referência: fusão conjunta do v2 (F3-H3) ou do v1 (F3-H1) |
| --- | --- | --- | --- | --- | --- |
| Churn 90 dias | nos alvos | 0,8291 | 0,8105 | −0,0186 [−0,0189; −0,0182] | v2 conjunta: 0,8194, −0,0097 [−0,0100; −0,0094] |
| Próxima online | fora dos alvos | 0,8448 | 0,8305 | −0,0143 [−0,0145; −0,0140] | v2 conjunta: 0,8280, −0,0168 [−0,0171; −0,0166] |
| Próxima com desconto | fora dos alvos | 0,6842 | 0,6716 | −0,0126 [−0,0131; −0,0120] | v1 (H3): 0,6667, −0,0175 [−0,0181; −0,0170] |

**Veredito: não atendido.** O critério pedia o limite inferior do IC acima de
−0,01 nas três perguntas; nenhuma passa.

Leitura:

1. **Fora dos alvos, o v2 transfere melhor que o v1, mas não o bastante.** Em
   desconto, a distância para T cai de 0,0175 (v1) para 0,0126. Em canal, a
   cabeça de tarefa única fica a 0,0143. As duas perguntas ficam a cerca de
   0,013 a 0,014 do LightGBM, a 30–40% além da margem.
2. **Em churn, a cabeça de tarefa única é pior que a conjunta** (−0,0186
   contra −0,0097, também na validação: 0,782 contra 0,790). A fusão conjunta
   recebe os baselines causais das 4 tarefas como entrada, e a de tarefa
   única só o do churn. É provável que a diferença venha dessas entradas e não
   da cabeça, mas o teste não separa os dois efeitos. Mesmo a conjunta fica
   na fronteira: o limite inferior (−0,01003) passa a margem por 0,00003.
3. **Consequência para o produto.** "Pergunta nova = cabeça leve sobre um
   vetor já calculado" ainda deixa 0,013 a 0,019 de AUC para trás na H&M. O
   que já empata é o híbrido (vetor + agregados automáticos; F3-H1: +0,0015
   em desconto). O caminho da solução única continua sendo H, não E.

Custo: extração dos embeddings do v2 em 69 s para 5,2 milhões de âncoras; cada
cabeça leva 100 a 140 s de L4. Arquivos: `runs/hm_clientes/f3h5/bootstrap_test.json`;
reprodução: `scripts/fila_colab_f3h5.sh`.

### F3-H6, etapa A — a mesma representação (motor v2) nos 4 problemas, no corte atual (20–21/11)

Uma única representação, o backbone `v2_pretreino` congelado, nos 4 problemas
da empresa, com a régua T de cada um. Desenho e mudança para duas etapas
registrados em `PLANO_FASE3.md` antes de rodar. Teste avaliado uma vez.

| Problema | Margem | T — régua específica | E — vetor + cabeça leve | H — T + vetor do v2 | Veredito |
| --- | --- | --- | --- | --- | --- |
| Recomendação (MAP@12) | 5% relativo | R2 0,0324 | MH (F3-H4) 0,0271: −16% | 0,0323: −0,3% [−0,0003; +0,0001] | **H empata** |
| Churn 90 dias (AUC) | 0,01 | 0,8291 | 0,8105 (cabeça só de churn) / 0,8194 (conjunta): −0,019 / −0,010 | 0,8226: −0,0065 [−0,0068; −0,0063] | **H empata** |
| Desconto (AUC) | 0,01 | 0,6842 | 0,6716: −0,0126 | 0,6852: +0,0010 [+0,0008; +0,0011] | **H empata** |
| Clusterização (índice; ARI) | 5% e ARI −0,05 | RFM 0,0694; ARI 0,966 | 0,0796 (+15%, IC exclui o zero); ARI 0,879 | 0,0796; ARI 0,777 | **não empata: estabilidade** |

**Veredito da etapa A: não atendida, pela estabilidade da clusterização.** O
H empata em recomendação, churn e desconto. Na clusterização, o índice do v2
passa o RFM (a forma forte do índice), mas o ARI fica 0,087 abaixo, e o
critério admite 0,05. Pela regra registrada, a etapa B (segundo corte) não
roda: o veredito de M1 na H&M é **não atendida no corte atual**.

Leitura:

1. **A clusterização virou.** Com o v1, o embedding explicava 13% do que o RFM
   explica (0,0088 contra 0,0695); com o v2, 115% (0,0796). Ele empata com o
   RFM em churn (0,154 contra 0,150) e LTV (0,100 contra 0,103) e o supera no
   canal (0,061 contra 0,023). O ganho não vem do calendário: os segmentos do
   v2 explicam 4,5% da incerteza do mês da âncora (no v1 eram 71%), e o mês
   explica só 0,5% do churn com o rótulo corrigido
   (`f3h6_cluster/diagnostico_calendario_test.json`).
2. **A falha de estabilidade vem de uma semente.** ARI por semente:
   0,926, 0,740 e 0,970. Na sensibilidade (fora do critério), com K = 4 o
   ARI do E é 0,985 (RFM 0,996) e com K = 16 é 0,839 (RFM 0,817); o índice
   passa o RFM nos dois. Fica registrado como observação; o critério é K = 8.
3. **Nos três empates, o mérito é da régua, não do vetor.** O H contém todas as
   features de T. Na recomendação, as features do v2 respondem por 2–4% do
   ganho do ranqueador e, sozinhas, não ordenam os candidatos (R5 0,0046,
   abaixo dos populares). O backbone congelado em 12/2019 não conhece a
   maior parte dos artigos da semana de teste. Em churn e desconto, o vetor
   responde por 24% e 13% da importância, mas o H fica no máximo 0,0065 abaixo
   ou 0,001 acima de T.
4. **Esforço.** O caminho que empata (H) custa o mesmo que T mais o vetor:
   não reduz o esforço por problema. O caminho barato (E: uma cabeça de 2 min
   de L4 por pergunta, sem código específico) só empata na clusterização,
   em índice.

Registro de esforço (corte atual; tempos medidos nesta rodada ou nas hipóteses citadas):

| Problema | T: código específico e tempo | E: código específico e tempo | H |
| --- | --- | --- | --- |
| Custo fixo | 97 features genéricas por YAML (`features_tradicionais.py`) | pré-treino v2: 55 min de L4, uma vez; extração 69 s | os dois custos fixos |
| Recomendação | geração de candidatos e features de par (`f2h3_candidatos.py`, 217 linhas) + ranqueador (223); ~5 min por semente | MH: 253 linhas (`f3h4_*`); um backbone por semana-alvo | T + pontuação v2 (93 linhas) |
| Churn | rótulo por YAML; HPO de 40 tentativas: ~35 min de CPU | 0 linhas; cabeça 100 s de L4 | HPO ~35 min de CPU |
| Desconto | coluna no adapter (113 linhas) + 4 linhas de YAML; HPO ~31 min | a mesma coluna; cabeça 100 s | HPO ~31 min |
| Clusterização | RFM: 4 colunas; K-means | PCA do vetor; K-means | os dois |

Arquivos: `runs/hm_clientes/f3h6/bootstrap_test.json`,
`runs/hm_clientes/f2h3/resultado_{val,test}_v2.json`,
`runs/hm_clientes/f3h6_cluster/resultado_{val,test}.json`; reprodução:
`scripts/fila_colab_f3h6a.sh`. As tarefas complementares (próxima online,
LTV e próxima categoria com H do v2) seguem na fila e entram aqui ao fim.

### F3-H7, camada 2 — Synerise: navegação real em volume (21–22/11)

Base do RecSys Challenge 2025 (arquivo bruto, 23/06 a 08/12/2022; CC BY-NC
4.0). Os 909 mil clientes com compra e todos os eventos deles: 1,42 milhão de
cestas, 5,2 milhões de eventos de carrinho, 5,4 milhões de buscas e 50 milhões
de visitas de página. No motor v2, 11,9 milhões de posições (7,4 milhões são
sessões de visita). Desenho registrado em `PLANO_FASE3.md` antes de rodar.
Teste: os últimos 56 dias; validação: os 28 anteriores.

| Problema | T (LightGBM) | E (vetor v2 + fusão) | H (T + vetor) | E − T | H − T |
| --- | --- | --- | --- | --- | --- |
| Churn 30 dias (AUC) | 0,7645 | 0,7479 | 0,7643 | −0,0166 [−0,0179; −0,0154] | −0,0002 [−0,0009; +0,0006] |
| Próxima categoria (accuracy) | 0,7811 | 0,7742 | 0,7814 | −0,0069 [−0,0079; −0,0058] | +0,0003 [−0,0001; +0,0007] |
| LTV 30 dias (accuracy) | 0,8632 | 0,8484 | 0,8632 | −0,0148 [−0,0156; −0,0139] | +0,0000 [−0,0003; +0,0005] |
| Clusterização (índice; ARI) | RFM 0,0573; 0,963 | 0,0208 (36%); 0,968 | 0,0528 (92%); 0,552 | não empata | não empata |

O que a navegação (carrinho, busca e visita) acrescenta, com o braço só de compras como referência:

| Problema | E | T (controle) | H |
| --- | --- | --- | --- |
| Churn (AUC) | **+0,0219** [+0,0206; +0,0231] | +0,0274 [+0,0260; +0,0289] | +0,0242 [+0,0226; +0,0259] |
| Próxima categoria | +0,0070 [+0,0062; +0,0076] | +0,0003 | +0,0009 |
| LTV | +0,0041 [+0,0034; +0,0047] | +0,0004 | +0,0002 |
| Clusterização (índice) | −0,029 (0,0208 contra 0,0499) | — | +0,0055 |

**Veredito.**

- **M2: atendido.** As quatro fontes entraram por adapter (103 linhas, só
  leitura e filtros) e YAML (101 linhas), sem mudança no motor. Duas correções
  genéricas saíram do caminho: a linha de compra sem item do catálogo não vira
  alvo de item na validação (o catálogo do modelo é cortado nos 100 mil SKUs
  mais frequentes), e o GBDT cria a pasta do cache do PCA.
- **M1, primeira parte: atendida.** A navegação melhora E em churn por mais que
  a margem (+0,022, IC inteiro acima de 0,01). É a primeira vez, nas três bases,
  que retirar uma fonte derruba o vetor por mais que a margem.
- **M1, segunda parte: não atendida.** H empata com T nos três problemas
  supervisionados (diferenças de no máximo 0,0003), e E empata só na próxima
  categoria. Na clusterização, nenhum espaço aprendido empata: H chega a 92% do
  RFM com estabilidade baixa (ARI 0,55), e E com todas as fontes cai para 36%.

Leitura:

1. **A navegação vale tanto para T quanto para E.** Em churn, T ganha +0,027 com
   os agregados genéricos das três fontes, e E ganha +0,022. O ganho é real, mas
   não é exclusivo da representação aprendida: os agregados por YAML já capturam
   a navegação.
2. **Onde a navegação ajuda só o E, o efeito é pequeno.** Próxima categoria
   (+0,007) e LTV (+0,004) melhoram em E e não mudam em T, mas ficam abaixo da
   margem.
3. **A navegação atrapalha a clusterização pelo vetor.** O vetor com as 7,4
   milhões de sessões de visita organiza os clientes pelo jeito de navegar, e os
   segmentos explicam menos os desfechos de compra (36% do RFM contra 87% sem a
   navegação). O vetor só de compras fica perto do RFM e é o mais estável
   (ARI 0,986).
4. **Limitação do desenho.** Com 79% de churn em 30 dias, as faixas de gasto do
   LTV colapsaram em duas (gastou ou não), e o LTV virou quase o complemento do
   churn. A próxima categoria tem 72% de "outras", o que comprime as diferenças.
   O pré-treino com todas as fontes sobreajustou a partir da época 3 (a melhor
   foi a 2); o só de compras melhorou até a 7.

Custo (L4 e CPU do Colab): pré-treino 27 min com todas as fontes e 3 min só com
compras; fusão ~16 min por semente; HPO do LightGBM 2,7 h (T) e 3,8 h (H) nas
três tarefas. Arquivos: `runs/synerise/syn_bootstrap_test.json`,
`runs/synerise/syn_cluster/resultado_{val,test}.json`; reprodução:
`adapters/adapter_synerise.py` (local) e `scripts/fila_colab_synerise.sh`.

