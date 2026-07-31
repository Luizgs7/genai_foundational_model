---
name: temporal-customer-splits
description: Como fazer splits de treino/validação/teste sem leakage para bases de transações de clientes usadas em um Customer Foundation Model (pré-treino sequencial causal + fine-tuning downstream).
---

# Splits temporais e por cliente sem leakage

## Princípio central

Um Customer Foundation Model treina com Next Token Prediction sobre a sequência de eventos de cada cliente. Há dois eixos de split que precisam ser aplicados **juntos**, nunca isoladamente:

1. **Split temporal** (obrigatório, primário): corte de data. Nenhum evento de treino pode ter `data_compra` posterior ao início do período de validação. Embaralhar transações de clientes diferentes ignorando a data é uma forma sutil de leakage — mesmo que o cliente-alvo esteja fora do conjunto de teste, o modelo "vê o futuro" através de outros clientes que compartilham padrões de mercado/sazonalidade daquele período.
2. **Split por cliente** (secundário, para downstream): clientes usados para validar tarefas de fine-tuning devem ter a fronteira temporal de avaliação estritamente após a fronteira usada no pré-treino — do contrário o modelo já "memorizou" o cliente específico antes de ser avaliado nele.

## Passo a passo

1. Definir um corte único de data para todo o dataset (não por cliente) — ex.: últimos N meses = teste, N meses anteriores = validação, resto = treino. Escolher N de forma que cada split tenha volume suficiente (checar contagem de linhas E de clientes, não só de linhas).
2. Verificar automaticamente que `max(data_compra em treino) < min(data_compra em validação) < min(data_compra em teste)` — sem overlap. Isso deve ser um assert no código, não uma checagem visual.
3. Marcar cada cliente com uma flag de elegibilidade para downstream (ex.: `elegivel_downstream = nº de transações >= 2`). Clientes de evento único formam uma população separada de "cold start" — não descartar, mas não tratar como equivalente a clientes com histórico.
4. Qualquer estatística calculada "no treino" (quantis de bucketing, médias de normalização, vocabulário) deve ser calculada **só** com linhas do split de treino e depois aplicada (nunca recalculada) em validação/teste.
5. Se o volume após o corte temporal for pequeno demais para um teste estatisticamente estável, considerar validação cruzada temporal (múltiplas janelas deslizantes) em vez de um único corte.

## Erros comuns a evitar

- Fazer split por cliente (ex. `train_test_split` aleatório sobre CPFs) sem também respeitar a ordem temporal — gera leakage porque o mesmo período de tempo aparece em treino e teste.
- Recalcular buckets/estatísticas de normalização separadamente em cada split — sempre "fit" no treino, "transform" em todos.
- Excluir clientes com poucas transações — eles são um caso de uso real (cliente novo), não ruído a descartar.
