---
name: customer-sequence-serialization
description: Como discretizar campos numéricos, desenhar um vocabulário fechado e serializar transações tabulares em sequências de tokens por cliente para um Transformer causal (Customer Foundation Model).
---

# Serialização de eventos de cliente para Transformer causal

## Discretização (bucketing)

- Usar **quantis**, nunca intervalos fixos — quantis garantem ocupação balanceada de cada bucket/token, o que ajuda o treino do embedding (tokens raros treinam mal). Intervalos fixos exigem conhecer a escala a priori, o que é frágil se a escala mudar entre categorias ou ao longo do tempo (inflação, novos produtos).
- Se o campo numérico tiver escala muito diferente entre subgrupos (ex.: categorias de produto com ordens de grandeza diferentes de preço), calcular os quantis **separadamente por subgrupo**, não globalmente — bucketing global concentra a maioria dos casos em 1-2 buckets e desperdiça resolução exatamente onde está o volume.
- Calcular os limites de quantil **apenas no split de treino** (ver skill `temporal-customer-splits`) e reaplicar os mesmos limites em validação/teste — nunca recalcular.
- Campos já discretos de baixa cardinalidade (ex. quantidade de itens 1-10) não precisam de bucketing — usar os próprios valores como tokens categóricos diretos.
- Datas: não tokenizar a data absoluta (explode vocabulário, não generaliza entre anos). Serializar como **delta desde o evento anterior**, discretizado em buckets desenhados sobre a distribuição real de gaps observada (não arbitrários) — tipicamente granularidade fina no intervalo onde se concentra a maior parte da massa (dias/semanas) e mais grossa na cauda (meses/anos).

## Vocabulário e tokenizer

- Antes de decidir entre BPE/subword e vocabulário fechado, **inspecionar os valores reais** dos campos de texto (não assumir a partir do nome da coluna). Campos de "descrição" frequentemente são templated/categóricos disfarçados (poucas variações fixas), não texto livre — nesse caso, tratar como token categórico é mais simples, mais interpretável e evita desperdiçar capacidade do modelo reconstruindo um template.
- Com cardinalidades baixas (dezenas a poucas centenas de valores por campo), um vocabulário 100% fechado e enumerado manualmente (dict token→id) é preferível a BPE: mais simples, sem sub-tokenização de categorias atômicas.
- Reservar tokens estruturais: `BOS`/`EOS` (início/fim de sequência do cliente), `EVT` (separador de evento, se a sequência tiver múltiplos campos por evento), `PAD`, `UNK`.
- Só usar BPE/subword para campos que, após inspeção manual de amostras, sejam de fato texto livre de alta variação — e mesmo assim, aplicar BPE só naquele campo específico, mantendo os demais campos como tokens categóricos fechados (decisão por campo, não uma escolha única para a tabela toda).

## Construção da sequência

1. Ordem dos campos dentro de cada evento deve ser **fixa e consistente** em 100% dos eventos — com NoPE (sem positional encoding), o modelo aprende ordem por conteúdo, então a ordem escolhida vira parte da "sintaxe" implícita.
2. Campos estáticos do cliente (não mudam entre eventos, ex. cidade de cadastro) não devem ser repetidos em cada evento da sequência — são candidatos a features estáticas (joint fusion no fine-tuning), não a tokens da sequência causal.
3. Campos redundantes/determinísticos entre si (ex. um total que é função exata de preço × quantidade) — serializar só um, nunca ambos, para não inflar o vocabulário sem adicionar sinal.
4. Clientes com um único evento devem gerar uma sequência válida e mínima (`BOS` + evento + `EOS`) sem tratamento especial no código — a ausência de contexto é o comportamento correto para esse caso, não um erro.
5. Sempre fazer um spot-check manual: decodificar de volta 3-5 sequências para forma legível e conferir visualmente contra as linhas originais da fonte.
