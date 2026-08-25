# Limitações e riscos — Customer Foundation Model

Levantamento informado pelos artefatos concretos produzidos nas Tarefas 3-8
(`pipeline/`, organizado por tema — splits/discretizacao/vocabulario/serializacao/
arquitetura/rotulos_downstream/pretreino), não por especulação. Cada item cita o número/arquivo
que o embasa. Perguntas que dependem de dados reais (fora do escopo do que
dá pra responder só com os dados sintéticos atuais) ficam para o
`PERGUNTAS_ABERTAS.md` (Tarefa 10).

---

## 1. Dados são sintéticos

Todo o pipeline (Tarefas 3-8) foi construído e validado sobre
`base_sintetica_embeddings_100k_v2.csv` — 100.000 transações, 15.000
clientes, gerados artificialmente. Nenhuma conclusão de distribuição (gaps
de compra, taxas de churn, ocupação de buckets) está garantida a se sustentar
sobre dados reais. Em particular:

- A taxa de churn de 29,70% (Tarefa 8) e a distribuição de gaps entre
  compras (mediana 41 dias, p99 676 dias) são propriedades do **gerador
  sintético**, não do negócio real — os thresholds escolhidos (N=121 dias
  para churn) precisam ser recalibrados assim que houver dados reais.
- Não há sazonalidade de negócio real (picos de safra agrícola, campanhas)
  contra a qual validar se a janela de churn escolhida confunde inatividade
  sazonal normal com abandono de verdade (risco já sinalizado na skill
  `downstream-label-engineering`).

## 2. Sequências e serialização (Tarefa 3)

- **1.391 de 15.000 clientes** não têm nenhum evento em `seq_train`
  (entraram na base só depois do corte de treino, 2025-10-30) — esses
  clientes não contribuem em nada para o pré-treino auto-supervisionado; só
  aparecem "frios" em validação/teste. Esse número subiu de 870 (config.
  original) para 1.391 depois da correção da Tarefa 7/8 (janela de teste
  ampliada de 90→180 dias empurrou o corte de treino ~90 dias mais cedo) —
  é um custo aceito deliberadamente em troca de viabilizar a avaliação de
  churn no split de teste (ver item 6). Se a proporção real de clientes
  novos for parecida, o modelo pré-treinado terá pouca ou nenhuma exposição
  a um segmento não-trivial da base em produção.
- Comprimento de sequência é bastante assimétrico: mediana 37 tokens, p90=205,
  p99=481, máximo=1057 (`pipeline/serializacao/sequencias.json`). A cauda longa
  (acima de p99) é rara mas existe — ver limitação de `n_positions` abaixo.

## 3. Discretização e vocabulário (Tarefas 4, 5)

- Buckets de quantil (`pipeline/discretizacao/buckets.json`) são calibrados **só
  no split de treino** — se a distribuição de `valor_total`/`desconto` mudar
  de forma relevante no futuro (inflação, novos produtos fora da faixa de
  preço observada), a ocupação dos buckets em produção pode ficar
  desbalanceada (ex. tudo caindo no bucket 9). Não há mecanismo de
  recalibração automática — precisa ser um processo manual/periódico.
- Vocabulário fechado, sem BPE (312 tokens, `pipeline/vocabulario/vocab.json`).
  Cobertura de 100% nos dados atuais (0 valores em `UNK`), mas isso é
  garantido só porque o vocabulário foi construído a partir do próprio
  dataset atual — uma categoria/marca/forma de pagamento nova que apareça no
  futuro vira `UNK` automaticamente, sem tratamento especial. Não há
  processo definido para expandir o vocabulário sem reiniciar o pré-treino
  do zero (mudar `vocab_size` invalida os embeddings já treinados).

## 4. Arquitetura e treino (Tarefa 6)

- **`n_positions=512` não cobre a sequência inteira de todo cliente**: cobre
  o p99 de comprimento em tokens (481), mas o máximo observado é 1057 — o
  1% de clientes de cauda mais longa perdem os eventos mais antigos por
  truncamento. O `ARQUITETURA.md` propõe currículo de janela + extrapolação
  via NoPE como mitigação, mas isso é um **experimento planejado, ainda não
  executado** — não há evidência ainda de que o NoPE de fato generaliza bem
  além da janela de treino neste domínio.
- ~~A validação da Tarefa 6 foi só um smoke test~~ **RESOLVIDO na Tarefa 11
  (2026-08-25)**: pré-treino real executado (30 épocas, 892.236 tokens de
  treino). Confirmado empiricamente o que este item previa: **overfitting
  claro**, com a melhor época de validação sendo a 1ª (loss_val=1,72,
  ppl=5,57) e loss de treino continuando a cair até 1,50 na época 30
  enquanto loss de validação sobe pra 1,88 — o modelo (14,3M parâmetros
  treináveis) tem 16x mais parâmetros do que tokens de treino disponíveis.
  Loss de teste no melhor checkpoint: 1,69 (ppl=5,44, bem abaixo do acaso
  ~312) — o backbone aprendeu estrutura real, mas está sobredimensionado
  pro volume de dados. Ver `pipeline/pretreino/pretreino_relatorio.json` e
  `pretreino_loss_curve.png`. **Achado adicional, mais grave**: a primeira
  tentativa de treino tinha um bug real (duplo deslocamento de `labels` —
  o `GPT2LMHeadModel` da HF já desloca `labels` internamente, e o código
  também deslocava antes de passar, fazendo o modelo aprender a prever o
  token **dois passos à frente**) que produzia loss de validação **47x
  pior que o acaso** — só detectado inspecionando previsão-token-a-token
  num cliente real, não apareceria numa checagem só de "a loss está
  caindo" (a loss de TREINO parecia razoável mesmo com o bug). Lição: uma
  curva de loss de treino "bonita" não é evidência suficiente de
  correção — vale sempre inspecionar previsões reais antes de confiar
  numa métrica agregada.
- Recomendação decorrente: antes do fine-tuning (Estágio 4, DCNv2), avaliar
  reduzir `n_layer`/`n_embd` (modelo menor) ou aumentar o volume de dados
  de treino — na configuração atual, mais épocas só pioram overfitting.
- A API real de `attn_implementation` do `transformers` instalado (5.15.0)
  diverge do padrão assumido inicialmente no código (precisa ir na
  `GPT2Config`, não no construtor do modelo) — indício de que a
  documentação/exemplos de referência usados podem estar desatualizados
  frente à versão de fato disponível; vale reconferir outras suposições de
  API antes do treino real.
- **Fragilidade de infraestrutura de treino**: o ambiente GPU (Colab, via
  túnel SSH efêmero — `colab_ssh/README.md`) não tem estado persistente. A
  instalação do `flash-attn` (sem wheel pré-compilado disponível para
  Python 3.13/torch 2.11/cu128 nesta data) levou **~3 horas compilando da
  fonte**, e precisa ser refeita a cada reinício do runtime do Colab. Isso é
  inviável como fluxo de treino de produção/CI — só serve para
  desenvolvimento exploratório. Um treino real de várias horas/dias precisa
  de um ambiente com GPU persistente (não Colab efêmero) ou de um artefato
  de wheel pré-compilado cacheado em algum storage próprio do time.

## 5. Splits temporais (Tarefa 7)

- Corte de teste usa só os últimos 90 dias do dataset (29/04 em diante,
  9.551 linhas) — volume relativamente pequeno frente ao treino (82.446
  linhas). Ver limitação de churn abaixo: essa janela curta interage mal
  com o horizonte de churn escolhido na Tarefa 8.

## 6. Rótulos downstream (Tarefa 8)

- ~~Churn não é avaliável no split de teste~~ **RESOLVIDO em 2026-08-25.**
  O achado original era: com janela de teste de 90 dias (Tarefa 7) e
  horizonte de churn de 121 dias (Tarefa 8), nenhum evento de teste
  conseguia satisfazer a condição de censura fechada — taxa de churn 0% no
  teste por construção, não o modelo acertando tudo. Corrigido ampliando
  `TEST_WINDOW_DAYS` para 180 dias (simulado antes de aplicar; ver
  `ROADMAP.md`, Tarefa 7, para a análise completa das alternativas
  consideradas) e repropagando pelas Tarefas 4, 3 e 8. Resultado pós-fix:
  taxa de churn no teste = 14,03%, AUC-ROC do baseline no teste = 0,750.
  Trade-off aceito: treino caiu de 82,4% para 74,4% do volume total, e o
  número de clientes sem nenhum evento de treino subiu de 870 para 1.391
  (item 2). Continua valendo reavaliar com dados reais: o horizonte N=121
  foi calibrado só sobre o gerador sintético.
- 6.430 linhas (de 100.000) foram excluídas do rótulo de churn por censura
  aberta (evento perto demais do fim do dataset para saber se o cliente
  voltou ou não dentro de N dias) — é um número esperado dado o desenho,
  mas reduz o volume de treino supervisionado disponível pra essa tarefa
  especificamente.
- O desvio do DoD original (churn **não** restrito a `elegivel_downstream=
  True`, diferente de "próxima categoria"/"próximo valor") é uma decisão de
  design documentada no código (`pipeline/rotulos_downstream/tarefa8_rotulos.py`), não um
  bug — mas é uma inconsistência de critério entre as 3 tarefas que vale
  alinhar explicitamente com o time antes de seguir para o fine-tuning.
- Baseline de "próximo valor" (moda histórica do bucket por cliente) fica
  em ~9,8% de accuracy — essencialmente aleatório (1/10 buckets = 10%).
  Isso pode significar duas coisas bem diferentes, que só um experimento
  real vai distinguir: (a) o valor da próxima compra não tem autocorrelação
  de curto prazo por cliente neste negócio (a tarefa é genuinamente difícil,
  e um resultado fraco do Customer Foundation Model aqui não seria culpa do
  modelo), ou (b) o bucket de quantil por categoria está descorrelacionado
  demais do comportamento individual pra essa tarefa fazer sentido como
  está definida.
- Baseline de "próxima categoria" já é relativamente forte (52,2% de
  accuracy, 52,0% F1 macro) com só 4 categorias possíveis — o modelo neural
  precisa superar isso com margem real (não só bater com folga estatística
  pequena) para justificar a complexidade adicional do backbone
  sequencial nesta tarefa específica.

## 7. Escopo não coberto ainda

- Estágios 4 e 5 do `ARQUITETURA.md` (fusão DCNv2, embedding universal) —
  ainda não implementados; o backbone (Tarefa 11) já está treinado, mas
  sobredimensionado pro volume de dados atual (ver item 4) — vale
  reconsiderar o tamanho do modelo antes de construir a fusão em cima dele.
- ~~Nenhum treino real foi executado~~ **RESOLVIDO na Tarefa 11.** Backbone
  pré-treinado por 30 épocas com observabilidade completa (métricas por
  passo/época, checkpoints, curva de loss) — ver item 4 e
  `pipeline/pretreino/pretreino_relatorio.json`.
