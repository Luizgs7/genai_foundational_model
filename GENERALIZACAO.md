# Generalização do pipeline — de "Customer Foundation Model" a produto

Branch `luiz_g_product_exp`. Documenta a generalização do pipeline
construído tarefa a tarefa na branch `luiz_g` (Tarefas 3-13, hardcoded pra
`base_sintetica_embeddings_100k_v2.csv`) para um **motor genérico**,
config-driven, capaz de rodar sobre dados transacionais/cadastrais de
qualquer empresa. Testado com dois configs completamente diferentes,
rodados ponta a ponta (splits → discretização → vocabulário → serialização
→ pré-treino real na GPU → fusão DCNv2 → relatório visual):

- **`synthetic_agro`**: migração do pipeline original (validação de
  portabilidade — deve reproduzir os números já publicados na branch `luiz_g`).
- **`olist_sellers`**: dataset real, [Olist Brazilian E-commerce](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
  (Kaggle, 9 tabelas relacionais), entidade = **vendedor** (não cliente).

Plano completo (contexto, decisões, trade-offs discutidos com o usuário
antes de implementar) em `/home/lgsouza/.claude/plans/temporal-painting-popcorn.md`.

## Por que generalizar

O pipeline original tinha nomes de coluna, número de campos categóricos e a
própria noção de "cliente" hardcoded em cada script. Virar produto significa:
uma empresa sobe seus dados, escreve um YAML mapeando suas colunas pro
schema canônico, e recebe de volta um Customer/Seller Foundation Model
treinado e testado nas tarefas downstream que fizerem sentido pro negócio
dela — sem tocar em código.

## Arquitetura

### Schema canônico

Todo o motor (`pipeline/common/`) opera sobre uma tabela de eventos com:
`entidade_id`, `evento_id`, `data_evento`, N campos categóricos (estratégia
`fechado` ou `top_n_outros` — este último calibrado só no treino, pra
campos de alta cardinalidade), M campos numéricos (bucket de quantil,
opcionalmente agrupado por um campo categórico), campos estáticos da
entidade (categóricos/numéricos, podem ser vazios), e uma lista de tarefas
downstream ativas. Cada empresa declara isso num YAML (`configs/*.yaml`);
um **adapter** (`adapters/adapter_*.py`) faz a tradução das tabelas brutas
da empresa pro CSV canônico — é a única parte genuinamente específica por
empresa.

### Catálogo de tarefas downstream

`pipeline/common/tarefa8_rotulos.py` deixou de calcular 3 colunas fixas e
virou um registro de receitas. Duas receitas genéricas cobrem quase tudo:

- **`campo_futuro`** (`shift(-1)` por entidade sobre um campo, com limiar
  opcional) — cobre `next_category`, `next_value`, e as 2 receitas novas
  da Olist (`risco_review_negativo`, `risco_atraso_entrega`).
- **`ltv`** (soma causal numa janela de H dias, com a censura do churn) —
  prova que o catálogo aceita uma 4ª receita sem mexer no motor.
- **`churn`** mantém lógica própria (candidatos de N variados + censura).

Uma pergunta de negócio nova normalmente vira uma entrada de config, não
código novo — essa é a demonstração concreta de "versatilidade" pedida
como diferencial do produto.

### Correção estrutural: posição de âncora explícita

A Tarefa 13 original calculava a posição do último token de cada evento
por fórmula fixa (`11 + 12*i`), assumindo um número fixo de campos por
bloco. Com campos configuráveis por empresa isso deixa de valer —
`pipeline/common/tarefa3_serializacao.py` agora grava explicitamente, por
evento, o índice do seu último token (`posicoes_evento.csv`), eliminando a
reconstrução aritmética.

### Layout

```
pipeline/common/       motor genérico (schema-agnóstico)
configs/                 synthetic_agro.yaml, olist_sellers.yaml
adapters/                adapter_synthetic_agro.py, adapter_olist_sellers.py
runs/<nome>/             artefatos por run (splits/discretizacao/vocabulario/
                           serializacao/arquitetura/rotulos_downstream/
                           pretreino/fusao/relatorios)
```

## Trade-offs técnicos (decididos com o usuário antes de implementar)

1. **Config declarativo explícito** (não auto-inferência de schema) — mais
   previsível/depurável, custo é onboarding manual por empresa.
2. **Top-N + OTHER** pra campos de alta cardinalidade (produto: 200 de
   ~32,7 mil na Olist) — vocabulário pequeno, perde granularidade de cauda.
3. **Posição de âncora explícita** (não fórmula) — ver acima.
4. **Migrar o pipeline sintético pro motor genérico** (não deixá-lo
   paralelo) — única forma de provar de verdade a portabilidade.
5. **Granularidade do evento Olist = item de pedido** (não pedido inteiro)
   — paridade com "uma linha = um produto" da base sintética.
6. **Arquitetura/churn recalibrados do zero** pra Olist (nunca reaproveita
   hiperparâmetros da base sintética) — mesma disciplina das Tarefas 6/12.
7. **Catálogo com 5 receitas implementadas** (`churn`, `next_category`,
   `next_value`, `ltv`, `campo_futuro` genérico) — "Next Best Offer" coberto
   por `next_category`; "recomendação de promoções" fica só documentada
   como lacuna (nenhum dos 2 datasets tem dado de promoção/cupom).
8. **Unidade de sequência da Olist = vendedor, não cliente** — ~97% dos
   `customer_unique_id` têm 1 só pedido no dataset inteiro, inviabilizando
   a premissa de sequência. Vendedores vendem com muito mais frequência.
9. **`n_positions=4096` pra Olist** (não 512) — testado primeiro com 1024
   (cobria 93,8% dos vendedores por sequência inteira), mas por **evento**
   isso descartava 36,1% das transações rotuladas (poucos vendedores de
   cauda longa concentram um nº desproporcional de eventos). Subiu pra
   4096 (12,9% de descarte) — custo de GPU seguiu desprezível.
10. **`attn_implementation=sdpa`** em vez de `flash_attention_2` pro run
    Olist — o runtime Colab resetou por completo no meio da sessão e
    reinstalar flash-attn do zero custaria ~2-3h de compilação de novo
    (mesmo problema já documentado na branch `luiz_g`). sdpa é a atenção
    nativa do PyTorch, sem compilação, numericamente equivalente pro
    volume de dados/tamanho de modelo deste projeto.

## Resultados lado a lado

### Pré-treino do backbone (NTP)

| | `synthetic_agro` | `olist_sellers` |
|---|---|---|
| Entidade | cliente | vendedor |
| Vocabulário | 312 tokens | 323 tokens |
| Campos categóricos/evento | 7 (categoria, marca, fabricante, produto, pagamento, canal, quantidade) | 3 (categoria, pagamento, produto) |
| Campos numéricos/evento | 2 (valor, desconto — agrupados por categoria) | 2 (preço, frete — sem agrupamento) |
| `n_positions` | 512 | 4096 |
| Parâmetros treináveis | 833K | 367K |
| Tokens de treino | 892K | 611K |
| Razão tokens/parâmetro | 1,07 | 1,67 |
| Perplexidade teste | 5,11 | 4,02 |
| Tempo de treino (GPU) | 5,8 min | 8,3 min |
| Attn. usada | flash_attention_2 | sdpa (ver trade-off 10) |

Ambos batem o acaso com folga grande (vocabulário de 312-323 tokens →
ppl≈312-323 no acaso) e não mostram sinal de overfitting severo (curva de
val estabiliza, não diverge) — mesma disciplina de dimensionamento
validada na Tarefa 12.

### Fusão DCNv2 — tarefas downstream

| Tarefa | Dataset | Sinal | Baseline (AUC/acc.) | Modelo | Delta |
|---|---|---|---|---|---|
| Churn | synthetic_agro | proxy | 0,7502 | 0,7964 | +0,0462 |
| Próxima categoria | synthetic_agro | proxy | 0,5898 | 0,5327 | −0,0571 |
| Próximo valor | synthetic_agro | proxy | 0,0995 | 0,1022 | +0,0027 |
| LTV | synthetic_agro | proxy | 0,3415 | 0,3171 | −0,0244 |
| Churn | olist_sellers | proxy | 0,8418 | 0,8770 | +0,0352 |
| **Risco de review negativo** | olist_sellers | **real** | 0,5826 | 0,5707 | −0,0119 |
| **Risco de atraso na entrega** | olist_sellers | **real** | 0,5425 | 0,5925 | +0,0500 |

**Leitura**: churn e atraso de entrega batem o baseline com folga — as 2
tarefas onde o histórico causal da própria entidade (frequência de vendas,
taxa histórica de atraso) já é informativo e agora chega ao modelo como
feature explícita (ver "Iteração de qualidade" abaixo). Em review negativo
o baseline (taxa histórica causal de review ruim do vendedor) é forte e o
modelo fica ligeiramente abaixo dele — a satisfação do PRÓXIMO comprador
específico parece carregar ruído que nem a sequência nem o histórico do
vendedor capturam (depende do produto/comprador daquele pedido, não só do
padrão do vendedor). Próxima categoria e LTV no synthetic_agro não bateram
seus baselines nesta rodada; reportado sem ajuste cosmético, mesma postura
da Tarefa 13 original.

### Iteração de qualidade — features causais explícitas na fusão

O ganho de AUC do DCNv2 sobre o baseline de churn na Olist era pequeno na
primeira rodada (+0,014). Investigação (sem GPU) achou 3 causas concretas:

1. **A cabeça DCNv2 recebia só o embedding congelado + `seller_state`** —
   nenhuma estatística causal chegava ao modelo como número, nem a mesma
   que os próprios baselines usam (ex. gap médio histórico entre vendas).
   `tarefa13_treinar_fusao.py` agora expõe automaticamente, como feature
   numérica de entrada, toda coluna `{tarefa}_baseline` numérica das
   tarefas ativas do config + uma feature genérica nova
   (`n_eventos_anteriores`, quantos eventos a entidade já teve antes
   deste, calculada em `tarefa8_rotulos.py`). Nenhum nome de campo
   específico de empresa é hardcoded — o mecanismo lê
   `tarefas_ativas(config)` e filtra por tipo numérico, então funciona pra
   qualquer config novo sem mudança de código.
2. **Bug no baseline de `risco_review_negativo`** — em `receita_campo_futuro`,
   quando o campo-fonte de uma receita com `limiar` não é binário (caso do
   `review_score`, 1-5), o baseline comparava contra a moda histórica do
   valor bruto em vez da taxa histórica causal do rótulo já limiarizado.
   Corrigido: o baseline agora usa sempre a mesma transformação aplicada
   ao rótulo, mas sem o `shift(-1)` do rótulo (`media_expandida_causal` já
   desloca 1 posição por conta própria — aplicar em cima do rótulo
   reintroduziria o próprio evento atual na média). Baseline de
   `risco_review_negativo` sobe de 0,486 (quase aleatório) pra 0,583
   (teste); `risco_atraso_entrega`, que já não tinha esse bug, ficou
   idêntico (0,5425→0,5426, diferença de arredondamento).
3. **BCE sem `pos_weight`** — as 3 tarefas Olist são desbalanceadas
   (churn 4,9%, review-ruim 15,8%, atraso 7,9%); adicionado `pos_weight`
   por tarefa (razão neg/pos do treino, calculada uma vez a partir do
   split de treino).

**Ablation de interferência multi-tarefa**: treinar só a cabeça de churn
(sem review/atraso) deu delta=+0,0336 — praticamente igual ao treino
conjunto (+0,0352). As 3 tarefas dividindo o mesmo tronco pequeno não
está prejudicando churn; hipótese descartada.

**Efeito misto no synthetic_agro**: o mesmo pacote de mudanças (features
causais + pos_weight) piorou levemente churn (+0,056→+0,046) e
next_category (−0,032→−0,057), mas melhorou bastante ltv (−0,061→−0,024).
Com 4 tarefas ativas dividindo o mesmo tronco pequeno (`n_cross=2`,
`rank=8`, `deep_hidden=16`), mais features de entrada competem por
capacidade entre tarefas de formas que esta rodada não isolou por
completo — candidato natural pra uma próxima iteração (peso por tarefa na
loss, ou crescer a cabeça agora que ela tem mais sinal pra usar).

## Limitações desta generalização

- **12,9% dos eventos da Olist** (14.413 de 112.101) ficaram sem embedding
  por truncamento de `n_positions` — concentrados nos vendedores de cauda
  mais longa (um vendedor sozinho tem 2.025 vendas).
- **`produto` quase sempre cai em `OUTROS`** pra vendedores de volume
  médio/baixo (ver relatório visual do vendedor-exemplo — nenhum dos
  produtos dele está no top-200) — o token de produto carrega pouco sinal
  de identidade específica nesses casos.
- **`sdpa` em vez de `flash_attention_2`** nesta rodada (trade-off 10) —
  resultado esperado ser equivalente, mas não foi comparado lado a lado
  com flash-attn pro mesmo config Olist (só temos essa comparação pro
  `synthetic_agro`, na branch `luiz_g`).
- **"Recomendação de promoções personalizadas"** citada pelo usuário como
  problema de negócio relevante não tem dado de suporte em nenhum dos 2
  datasets — fica documentada como lacuna, não implementada.
- **Só uma entidade de exemplo por dataset** teve relatório visual gerado
  manualmente — outros vendedores/clientes não foram inspecionados
  individualmente.

## Artefatos e relatórios visuais

- Relatório do cliente 860.703.096-50 (base sintética, branch `luiz_g`):
  https://claude.ai/code/artifact/7d147268-fc28-4c69-b8ac-084a41ebd356
- Relatório do vendedor Olist (esta branch):
  https://claude.ai/code/artifact/38eb86d0-17fb-4d2c-9f3d-2fa9b806eeeb
- Dados brutos da Olist em `dados_olist/` (git-ignorado — dados de
  terceiros, baixados via `kagglehub`).
