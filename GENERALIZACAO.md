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
| Churn | synthetic_agro | proxy | 0,750 | 0,806 | +0,056 |
| Próxima categoria | synthetic_agro | proxy | 0,590 | 0,558 | −0,032 |
| Próximo valor | synthetic_agro | proxy | 0,099 | 0,103 | +0,003 |
| LTV (nova receita) | synthetic_agro | proxy | 0,341 | 0,280 | −0,061 |
| Churn | olist_sellers | proxy | 0,842 | 0,856 | +0,014 |
| **Risco de review negativo** | olist_sellers | **real** | 0,486 | 0,514 | +0,028 |
| **Risco de atraso na entrega** | olist_sellers | **real** | 0,543 | 0,599 | +0,056 |

**Leitura**: nas 2 tarefas com rótulo real (não inventado), o baseline
simples (taxa histórica do vendedor) é fraco — quase aleatório pra review
(0,486) — e é exatamente onde o embedding sequencial mostra o ganho
proporcional mais claro. Próxima categoria e LTV não bateram seus
baselines nesta rodada; reportado sem ajuste cosmético, mesma postura da
Tarefa 13 original.

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
