# Base sintética de beleza B2B

Uma empresa fictícia de perfumaria e beleza que vende para revendedores e
lojas. É a única base do projeto em que **todos os tipos de dado do mesmo
cliente** aparecem juntos: pedidos, navegação, crédito, programa de incentivo
e cadastro. Serve às hipóteses F3-H7 e F3-H8 de `PLANO_FASE3.md`.

Não é dado real. Ela prova **capacidade**: que o motor aproveita cada tipo de
dado quando há sinal, e que aguenta formas extremas. A evidência de valor de
negócio vem das bases reais.

## Como gerar

```bash
python3 gerar_dados_sinteticos_beleza_b2b.py --tamanho pequeno   # mini | pequeno | grande
python3 validar_base_beleza.py dados_beleza_b2b/pequeno          # confere a estrutura plantada
```

A saída vai para `dados_beleza_b2b/<tamanho>/`, que não entra no git. A
geração é determinística para a mesma semente.

| Tamanho | Clientes | SKUs | Pedidos | Linhas de pedido | Eventos de navegação | Tempo |
|---|---|---|---|---|---|---|
| mini | 2.000 | 1.500 | 45 mil | 0,9 milhão | 0,45 milhão | 4 s |
| pequeno | 15.000 | 6.000 | 332 mil | 7,1 milhões | 3,5 milhões | 17 s |
| grande | 100.000 | 30.000 | a gerar | a gerar | a gerar | — |

São 7 anos (2019 a 2025), em 122 ciclos de campanha de 21 dias.

As tabelas de formas e de conferência abaixo foram medidas antes de um ajuste
feito em 14/11: os eventos de navegação passaram a se agrupar em sessões (1 ou
2 por pedido, nas 72 horas anteriores). A conferência foi refeita e todos os
efeitos continuam presentes; os números exatos ainda não foram atualizados
aqui (`gabarito/validacao.json` traz os atuais).

## Tabelas

| Arquivo | Conteúdo | Natureza |
|---|---|---|
| `pedidos.parquet` | Cabeçalho: cliente, data, ciclo, canal, forma de pagamento, valor | Evento com itens |
| `itens_pedido.parquet` | Linhas: SKU, quantidade, preço unitário, desconto, se estava em promoção | Itens do evento |
| `navegacao.parquet` | Busca, visualização de produto e de campanha, carrinho e abandono, no aplicativo e no site | Evento simples, de alta frequência |
| `credito.parquet` | Por ciclo: limite, utilizado, disponível, dias de atraso e bloqueio | Estado que muda no tempo |
| `incentivo.parquet` | Início do ciclo: nível e meta. Fim do ciclo: pontos e se atingiu a meta | Estado e eventos |
| `clientes.parquet` | Tipo de cliente, UF, data de cadastro, canal de aquisição, idade | Atributo fixo |
| `produtos.parquet` | Nome, categoria, tipo, família, linha, marca, faixa de preço, preço de lista, lançamento e descontinuação | Catálogo |
| `campanhas.parquet` | SKUs em promoção em cada ciclo e o desconto | Calendário comercial |

O gabarito fica em `gabarito/` e **nunca** entra no motor nem na régua
tradicional. Serve só para avaliar:

| Arquivo | Conteúdo |
|---|---|
| `clientes_latentes.parquet` | Segmento, frequência e tamanho de pedido de base, sensibilidade a desconto, engajamento digital, risco de crédito, risco de saída, ciclo de saída |
| `regras_compra_conjunta.parquet` | Pares SKU A, SKU B e a probabilidade de B acompanhar A |
| `pedidos_origem.parquet` | Se o pedido é um pedido de complemento para bater a meta |
| `itens_origem.parquet` | Origem de cada linha: preferência, promoção, interesse, reposição ou compra conjunta |
| `validacao.json` | Saída de `validar_base_beleza.py` |

## Formas (tamanho pequeno)

| Tipo de cliente | Clientes | Pedidos por cliente (mediana; máximo) | Linhas por pedido (mediana; máximo) | Valor mediano do pedido |
|---|---|---|---|---|
| Revendedor | 12.949 | 5; 296 | 4; 84 | R$ 402 |
| Revendedor líder | 1.463 | 32; 604 | 15; 126 | R$ 2.931 |
| Loja | 588 | 40; 2.517 | 49; 830 | R$ 32.205 |

- 21% dos clientes têm um único pedido.
- 13 clientes têm mil pedidos ou mais.
- 12.566 pedidos têm cem linhas ou mais.
- O pedido vai de cerca de R$ 100 (percentil 5) a R$ 494 mil (máximo).

## Estrutura plantada e conferência

Cada efeito foi conferido nos dados gerados (tamanho pequeno), comparando o
grupo em que deveria aparecer com um grupo de referência.

| Efeito plantado | Onde deveria aparecer | Medido | Referência |
|---|---|---|---|
| Segmento latente define o mix | Categoria dominante de cada segmento | 35% a 58% das linhas | 8 categorias |
| Sensibilidade a desconto | Parcela de linhas em promoção, clientes mais sensíveis | 27% | 15% nos menos sensíveis (correlação de 0,61) |
| Compra conjunta | Chance de B estar no pedido quando A está | 11,9% | 0,75% sem A |
| Navegação antecede a compra | Produto visto vira linha de pedido em 30 dias | 56% | 0,6% para um SKU ao acaso |
| Crédito e saída | Taxa de saída no ciclo em que o cliente está bloqueado | 11,6% | 2,8% sem bloqueio |
| Crédito e tamanho do pedido | Tamanho relativo do pedido com menos de 40% do limite livre | 0,87 | 1,00 com mais de 90% livre |
| Pedido de complemento | Pedido nos últimos 4 dias do ciclo, com 75% a 100% da meta | 67% | 14% com menos de 50% da meta |

Origem das linhas de pedido: preferência do segmento 47%, reposição 20%,
promoção 20%, interesse vindo da navegação 7% e compra conjunta 6%.

## Limitações conhecidas

- A conversão de visualização em compra (56%) é mais alta que a de um varejo
  real, porque as lojas compram centenas de SKUs por ciclo.
- A individualidade do cliente vem do segmento, dos itens de reposição e das
  variáveis latentes. Não há gosto individual além disso.
- Os efeitos são independentes entre si por construção. Em dado real eles se
  misturam.
