---
name: downstream-label-engineering
description: Como derivar rótulos de tarefas downstream (churn, próxima categoria, próximo valor) a partir de um log de transações de clientes, com baselines simples e métricas apropriadas.
---

# Engenharia de rótulos downstream a partir de log de transações

## Princípio central

Toda tarefa downstream derivada de um log de transações deve ter o rótulo calculado **olhando só para o futuro daquele evento específico**, nunca para o dataset inteiro do cliente de uma vez — do contrário o rótulo vaza informação que só existiria em produção depois do fato.

## Churn / inatividade

- Rótulo: "o cliente NÃO compra novamente dentro de uma janela de N dias após o evento atual". N deve ser escolhido olhando a distribuição real do **gap entre compras consecutivas** do próprio dataset (não um número arbitrário como 30/60/90 "de cabeça") — calcular percentual de gaps abaixo de cada corte candidato e escolher o que produz uma taxa de positivos nem trivialmente baixa (ex. <5%) nem trivialmente alta (ex. >90%).
- Só é aplicável a clientes com pelo menos 2 transações (precisa de um "evento atual" com um "próximo" observável, ou censura conhecida no fim da janela de dados).
- Métrica: AUC-ROC (não accuracy) — churn tende a ser desbalanceado mesmo quando calibrado com cuidado.
- Baseline simples: regra de recência (RFM) — ex. "prever churn se dias desde a última compra > threshold" — sem nenhum aprendizado. O modelo só se justifica se superar esse baseline com margem.
- **Verificar antes de fixar o threshold definitivo**: se o negócio tem sazonalidade forte (ex. picos de compra concentrados em certos meses do ano), uma janela fixa de N dias pode confundir "inatividade sazonal normal" com churn de verdade — vale checar com a área de negócio se já existe uma definição operacional de churn em uso.

## Próxima categoria / próximo valor

- Rótulo: valor do campo-alvo na próxima transação do mesmo cliente (linha seguinte, ordenada por data). Trivial de derivar via `groupby(cliente).shift(-1)` ou equivalente.
- Baseline **por cliente** (ex. moda histórica das compras anteriores daquele cliente) é o teste certo para saber se um embedding aprendeu preferência individual — um baseline só de frequência global é fraco demais e não isola esse efeito.
- Para valor numérico, preferir prever o bucket discretizado (ver skill `customer-sequence-serialization`) em vez de o valor bruto, salvo se regressão for explicitamente necessária — classificação em buckets já calibrados evita problemas de escala/outliers.

## Checklist antes de considerar uma tarefa "pronta"

1. O rótulo é calculado com um corte temporal consistente com o split de treino/val/teste (ver skill `temporal-customer-splits`) — nenhum rótulo de treino depende de um evento que só existe no futuro além do corte de validação.
2. A taxa de positivos (base rate) foi calculada e documentada — uma tarefa com base rate abaixo de ~2-3% ou acima de ~97-98% provavelmente é degenerada e não vale o esforço de fine-tuning até ser recalibrada.
3. Existe um baseline simples e sua métrica já calculada, antes de qualquer treino do modelo — o valor do Customer Foundation Model só se justifica se ele superar esse baseline.
