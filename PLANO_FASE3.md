# Plano da fase 3 — uma solução, vários problemas

Reestruturação dos experimentos, decidida em 2026-11-13. Parte de duas coisas:
o balanço das fases 1 e 2 na H&M (`EXPERIMENTOS_HM.md`) e as respostas do
responsável pelo projeto sobre a base da empresa e sobre o que conta como
sucesso.

Este arquivo registra a pergunta, o desenho e o critério de cada hipótese
**antes** de qualquer resultado da fase 3. O artigo recebe o mesmo conteúdo
(seção 13) quando o conector do Claude Docs for reconectado. As hipóteses da
fase 3 usam o prefixo **F3-**.

## 1. O que mudou

### O que as fases 1 e 2 mostraram

1. **As 4 tarefas de propensão da H&M parecem saturadas por agregados simples.**
   O GBDT com 1% dos clientes rotulados fica a 0,005 do máximo em churn. O
   híbrido não ganha nem nos clientes com 11 ou mais cestas. Metade dos
   clientes tem até 3 cestas em dois anos.
2. **Onde havia espaço (recomendação), a tokenização descartou a identidade do
   produto.** Os 2.000 artigos com token próprio cobrem 27% das vendas; na
   semana de teste da F2-H3, 90% caem em "outros".
3. **O pré-treino aprende outra coisa.** Cerca de 80% da loss do backbone final
   é prever atributos do próximo item (cor 22%, preço 14%, índice e tipo 12%
   cada). As tarefas perguntam sobre a próxima cesta e os próximos 90 dias.
4. **A comparação "fusão contra GBDT" mistura duas coisas.** A cabeça de fusão
   recebe o embedding e 10 variáveis; o GBDT recebe 97 features.

### O que a empresa precisa (respostas de 2026-11-13)

| Tema | Resposta |
|---|---|
| Base | Mais de 7 anos de transações e mais de 2 milhões de clientes. Pedidos de R$ 200 com 2 ou 3 SKUs até R$ 100 mil com centenas de SKUs. Clientes com 1 pedido e clientes com milhares. |
| Outras fontes | Cadastro, crédito disponível, navegação no aplicativo e no site, participação em programas de incentivo, entre outras. |
| Problemas de negócio | Recomendação de produtos (upsell e cross-sell); probabilidade de churn em 90 dias; clusterização de clientes; propensão a uso de desconto. Hoje resolvidos, na maioria, com modelos tradicionais. |
| Critério de sucesso | **Empatar já basta:** fazer o mesmo que os modelos tradicionais com menos esforço, com uma única solução para vários problemas. Ganhar em acurácia é bônus. |
| Alcance | Uso interno primeiro. Se funcionar, virar produto digital para outras empresas; por isso os testes em várias bases. |

### Consequências para o desenho

- **O critério deixa de ser "vencer o GBDT" e passa a ser "empatar com menos
  esforço".** Isso pede margem de empate definida antes e um registro de
  esforço medido, e não só AUC.
- **A bateria de avaliação passa a espelhar os 4 problemas da empresa.** Dois
  deles nunca foram medidos: clusterização e propensão a desconto. Um falhou
  por desenho: recomendação.
- **O motor atual não comporta a base da empresa.** Cada item vira cerca de 11
  tokens e a janela é de 1.024 tokens, ou cerca de 93 itens. Um único pedido
  com centenas de SKUs estoura a janela; um cliente com milhares de pedidos
  perde quase todo o histórico. Catálogo grande e várias fontes de evento
  também ficam de fora.

## 2. Hipóteses macro, com critério explícito

- **M1 — benefício.** Uma única representação de clientes, treinada uma vez
  por base, resolve os problemas de negócio com qualidade equivalente à dos
  modelos tradicionais específicos e com menos esforço por problema.
  "Equivalente" é ficar dentro da margem de empate da seção 3. Ganho de
  acurácia é bônus.
- **M2 — flexibilidade.** O motor treina essa representação na base de
  qualquer empresa só com configuração. Passa a incluir o que a base da
  empresa exige: várias fontes de evento, catálogo grande, históricos de 1 a
  milhares de pedidos e pedidos de 2 a centenas de itens.

A redação anterior de M1 ("traz benefício real") continua **não confirmada na
H&M** pelos critérios escritos nas fases 1 e 2. A fase 3 testa a redação
acima; não reinterpreta resultados antigos como confirmação.

## 3. A bateria dos quatro problemas

Cada base passa pela mesma bateria. Na H&M:

| Problema da empresa | Espelho na H&M | Régua tradicional específica (T) | Métrica | Margem de empate |
|---|---|---|---|---|
| Recomendação (upsell, cross-sell) | Artigos comprados na semana seguinte (protocolo da F2-H3) | Candidatos + features de par + LightGBM LambdaRank (R2) | MAP@12 | 5% relativo |
| Churn em 90 dias | `churn` (já existe) | LightGBM com as 97 features e HPO | AUC | 0,01 |
| Propensão a desconto | A próxima cesta tem item abaixo do preço de referência do artigo (`proxima_com_desconto`, nova) | LightGBM com as features e HPO, incluindo o histórico de desconto | AUC | 0,01 |
| Clusterização | Segmentos formados no momento da compra | K-means sobre RFM e sobre os agregados | Parcela da incerteza dos desfechos futuros explicada pelo segmento | 5% relativo |

As tarefas antigas (`proxima_online`, `next_category`, `ltv_90d`) continuam
sendo medidas como complemento.

### Braços

- **T — régua.** O modelo tradicional específico de cada problema, como a
  empresa faz hoje.
- **E — representação aprendida.** O vetor do cliente (e o do item, na
  recomendação) com uma cabeça leve por problema, sem engenharia de features.
- **H — representação híbrida.** O vetor aprendido junto com os agregados
  automáticos (as 97 features genéricas, geradas pelo YAML), com cabeça leve.
- **J — ajuste fino conjunto.** Bônus de acurácia; custa um treino por
  problema.

**Sucesso de M1:** E ou H empata com T nos 4 problemas, com esforço marginal
menor. Se só H empatar, o produto é híbrido por desenho. Em propensão, H
contém as features de T, então o empate de H é quase garantido ali; o que
informa é o resultado de E, o dos problemas sem features prontas
(recomendação e clusterização) e o registro de esforço.

### Margens de empate

- 0,01 de AUC (ou de accuracy) nas tarefas de propensão. É o dobro do ruído
  que já medimos entre sementes e entre períodos (cerca de 0,005).
- 5% relativo em MAP@12 e no índice de clusterização.
- O empate é declarado quando o limite inferior do IC 95% da diferença
  (solução única menos régua) fica acima de menos a margem.

As margens valem para toda a fase 3 e só mudam antes de um resultado, com
registro aqui.

### Protocolo

- Mesmas âncoras e mesmos splits para todos os braços; teste avaliado uma vez.
- IC 95% por bootstrap Poisson por cliente (`avaliacao.py`).
- O teste central (F3-H6) roda em **dois cortes temporais**: o atual (teste de
  26/03 a 22/09/2020, com a COVID) e um anterior, deslocado 6 meses.
- O pré-treino pode usar todos os eventos anteriores à data de corte de cada
  avaliação, porque não depende de rótulo. É assim que rodaria em produção.

### Registro de esforço

Para cada problema e cada braço:

| Item | Como medir |
|---|---|
| Código específico do problema | Linhas de código fora do motor genérico |
| Configuração | Linhas de YAML |
| Tempo até o primeiro modelo | Horas de CPU e de GPU, e tempo de parede |
| Peças a manter | Número de pipelines e de modelos distintos |
| Custo fixo | Pré-treino e geração de features, contados à parte |

## 4. Etapas e hipóteses

### Etapa 0 — encerrar a rodada atual (F2-H10 e F2-H11)

Em andamento no Colab. Os critérios já estão no artigo (seções 12.2 e 12.3).

- **F2-H10 (LLM).** Se o E3a não superar o E0 por 0,005 na média das 4
  tarefas, a linha de LLM por cliente para. Gerar embeddings com o Qwen custa
  cerca de 11,5 h de GPU por milhão de clientes, contra cerca de 20 min do
  nosso backbone. Um codificador de texto ainda pode servir para descrever
  **itens** (F3-H4), o que roda uma vez por catálogo.
- **F2-H11 (TabPFN, TabICL).** Se empatarem com o LightGBM sem ajuste de
  hiperparâmetros, viram candidatos a cabeça leve da bateria.

### Etapa 1 — placar do motor atual nos 4 problemas

Sem pré-treino novo. Dá a linha de base da fase 3 e estreia os dois problemas
que nunca medimos.

**Leitura retrospectiva (não é confirmação).** Aplicando a margem de 0,01 aos
resultados já publicados, no teste:

| Problema | Régua T | E congelado (fusão completa) | J (ajuste conjunto) | H (GBDT híbrido) |
|---|---|---|---|---|
| Churn 90 dias | 0,824 | −0,011 (fora) | −0,013 (fora) | −0,004 (empata) |
| Próxima online | 0,845 | −0,007 (empata) | −0,003 (empata) | +0,0003 (empata) |
| Próxima categoria | 0,373 | −0,014 (fora) | −0,005 (empata) | −0,003 (empata) |
| LTV 90 dias | 0,399 | −0,025 (fora) | −0,007 (empata) | −0,007 (empata) |
| Recomendação (MAP@12) | 0,0324 | −41% (fora) | — | +1,6%, mas usa todo o pipeline específico |

"Fusão completa" é o embedding com o cadastro e 3 estatísticas causais, não o
embedding sozinho.

#### F3-H1 — propensão a desconto com o motor atual

- **Pergunta.** A pergunta "o cliente vai usar desconto na próxima compra?"
  entra só com configuração, e a solução única empata com o modelo específico?
- **Por que importa (M1 e M2).** É um dos 4 problemas da empresa e nunca foi
  medido.
- **Desenho.**
  - Preço de referência de cada artigo: o maior valor da mediana diária de
    preço observado até a véspera. É causal: só usa vendas passadas.
  - Um item tem desconto quando o preço fica abaixo de (1 − d) vezes a
    referência. O limiar d (5%, 10% ou 20%) e a agregação por cesta (algum
    item, ou a maioria) são escolhidos **só no treino**, para a taxa base
    ficar entre 20% e 80%, antes de treinar qualquer modelo.
  - Rótulo `proxima_com_desconto`: a próxima cesta tem desconto.
  - T: LightGBM com as features genéricas mais o histórico de desconto
    (`campos_binarios`), com HPO. E: fusão com o backbone H3. H: GBDT
    híbrido. 3 sementes.
- **Critério, escrito antes de rodar.**
  - M2: nenhuma linha de código específica do problema além da coluna
    derivada no adapter.
  - M1: E ou H dentro de 0,01 de AUC de T.
- **Custo.** CPU local (features e HPO) e cerca de 15 min de GPU.

#### F3-H2 — clusterização

- **Pergunta.** Os segmentos formados a partir da representação única explicam
  o comportamento futuro tão bem quanto os segmentos tradicionais?
- **Por que importa (M1).** É um dos 4 problemas da empresa e é o único sem
  rótulo: depende só da geometria da representação.
- **Desenho.**
  - Uma âncora por cliente, sorteada com semente fixa. O K-means é ajustado
    nas âncoras de treino e aplicado às de teste.
  - Espaços comparados, todos padronizados:
    - T1: RFM clássico (intervalo desde a cesta anterior, frequência e gasto
      em 365 dias, tempo de casa);
    - T2: os 97 agregados reduzidos a 32 componentes;
    - E: o embedding reduzido a 32 componentes (backbone H3; o final e o CoLES
      entram se os embeddings estiverem disponíveis);
    - H: T2 e E juntos.
  - K = 8 (principal); K = 4 e K = 16 como sensibilidade. 3 sementes.
  - Desfechos futuros da âncora: churn, faixa de LTV em 90 dias, próxima
    categoria, próxima online e, quando existir, próxima com desconto.
  - Índice principal: média, sobre os desfechos, da parcela da incerteza
    explicada pelo segmento, I(segmento; desfecho) / H(desfecho).
  - Estabilidade: ARI entre dois ajustes em metades disjuntas dos clientes de
    treino, aplicados ao mesmo teste.
  - O código é desenvolvido na validação; o teste roda uma vez.
- **Critério, escrito antes de rodar.**
  - Empate: o índice de E ou de H é pelo menos 95% do melhor entre T1 e T2, e
    o ARI não fica mais de 0,05 abaixo.
  - Forma forte: E ou H acima do melhor T, com IC excluindo o zero.
- **Custo.** Só CPU local.

### Etapa 2 — motor v2: evento composto

Um redesenho do que entra e do que sai do backbone. Corrige os diagnósticos 2
e 3 da seção 1 e é pré-requisito para a base da empresa. O desenho detalhado
(formato dos dados, cabeças e perdas) é escrito antes da primeira execução.

Princípios:

- **Uma posição por cesta (ou por linha, conforme o YAML),** e não 11 tokens
  por item. O vetor da posição soma os embeddings dos campos de cada linha e
  agrega as linhas da cesta. Um pedido de centenas de itens ocupa uma posição.
- **Identidade de item no catálogo inteiro.** O vetor do item soma um
  embedding próprio com os embeddings dos seus atributos (e, opcionalmente, do
  texto do produto). Item novo ou raro é representado pelos atributos.
- **Tempo contínuo na entrada.** O intervalo desde o evento anterior e o
  calendário entram como entrada, e não como alvo. Isso elimina o defeito do
  token de mês.
- **Saídas alinhadas às perguntas.** A posição prevê a próxima cesta (itens,
  valor, intervalo até ela) e alvos de janela futura gerados automaticamente
  (número de cestas e gasto em 30, 90 e 180 dias).
- **Tipos de evento.** Cada fonte (compra, navegação, crédito, incentivo) é um
  tipo, com campos próprios, na mesma linha do tempo.

#### F3-H3 — o evento composto preserva a qualidade com custo muito menor

- **Pergunta.** Trocar 11 tokens por item por uma posição por cesta mantém a
  qualidade e resolve a escala?
- **Por que importa (M2).** Sem isso o motor não comporta pedidos com centenas
  de itens nem clientes com milhares de pedidos.
- **Desenho.** Mesmo decoder (256×6), mesmas âncoras e a mesma fusão da H4, com
  3 sementes. O motor v2 também roda nas bases sintética e Olist, só com
  configuração.
- **Critério.**
  - Qualidade: a fusão com o embedding do v2 fica a no máximo 0,005 da fusão
    do v1 nas 4 tarefas antigas.
  - Custo: o pré-treino usa no máximo um quinto do tempo de GPU por época.
  - Cobertura: 100% das âncoras numa única janela, sem truncamento.

#### F3-H4 — recomendação direta do modelo, com identidade de item

- **Pergunta.** Com o catálogo inteiro na saída e o backbone atualizado até a
  véspera, o modelo recomenda sozinho, sem o pipeline de candidatos e features?
- **Por que importa (M1).** Recomendação é o problema em que a solução única
  falhou (−41%) e o que mais custa esforço específico no caminho tradicional.
- **Desenho.** Protocolo da F2-H3 (validação em 05/03/2020, teste em
  16/09/2020). O backbone é treinado com os eventos até a véspera de cada
  semana-alvo, como seria em produção. As 12 recomendações saem do catálogo
  inteiro pela pontuação do modelo. 3 sementes.
- **Critério.**
  - Necessário: MAP@12 acima do R1 (recompra + populares; 0,0229 no teste).
  - Empate: pelo menos 95% do R2 (0,0324 no teste).
  - Forma forte: acima do R2 com IC excluindo o zero.
- **Se não empatar.** A recomendação segue híbrida: a pontuação do modelo
  entra como feature do ranqueador específico.

#### F3-H5 — pré-treino alinhado ao futuro serve a propensão sem ajuste fino

- **Pergunta.** Com alvos de janela futura no pré-treino, o embedding
  congelado com cabeça leve empata com o modelo específico, inclusive em
  perguntas que não estavam entre os alvos?
- **Por que importa (M1).** É a promessa central do produto: pergunta nova
  vira uma cabeça leve sobre um vetor já calculado.
- **Desenho.** Os alvos do pré-treino são genéricos: conteúdo, valor e
  intervalo da próxima cesta, e contagens e gasto em 30, 90 e 180 dias.
  `proxima_online` e `proxima_com_desconto` ficam **fora dos alvos** e servem
  de teste de transferência. 3 sementes.
- **Critério.** E congelado dentro de 0,01 de T em churn (pergunta da família
  dos alvos) e nas duas perguntas de fora. Os dois grupos são reportados
  separados.

#### F3-H6 — teste central: uma solução, quatro problemas

- **Pergunta.** A mesma representação, treinada uma vez, empata com os 4
  modelos específicos com menos esforço?
- **Desenho.** Motor v2, bateria completa, braços T, E, H e J, dois cortes
  temporais, 3 sementes e o registro de esforço.
- **Critério.**
  - Qualidade: E ou H empata com T nos 4 problemas, nos dois cortes.
  - Esforço: nenhum código específico por problema no caminho da solução
    única; tempo marginal por problema novo menor que o de T; menos pipelines
    a manter.
- **O que decide.** O veredito de M1 na H&M pelo critério da seção 2.

### Etapa 3 — várias fontes de evento

#### F3-H7 — navegação e compra na mesma linha do tempo

- **Pergunta.** O motor v2 ingere várias fontes só com configuração, e a
  navegação melhora a representação sem engenharia de features?
- **Por que importa (M2 e M1).** A base da empresa tem navegação, crédito e
  incentivos; a H&M só tem compras.
- **Desenho.** Segunda base pública com compra e navegação. Primeira
  candidata: os dados do RecSys Challenge 2025 (compra, carrinho, visita de
  página e busca; tarefas de churn e de propensão por categoria e por SKU), se
  continuarem disponíveis. Alternativa: o MBD (transações, geolocalização e
  diálogos de atendimento).
- **Critério.**
  - M2: a bateria roda com adapter e YAML, sem mudança no motor.
  - M1: incluir a navegação melhora E em pelo menos um problema por mais que a
    margem, e E ou H empata com T (que também recebe agregados da navegação).

### Etapa 4 — base da empresa

#### F3-H8 — a bateria com os problemas e os rótulos reais (antiga F2-H9)

- **Pergunta.** Na base da empresa, a solução única empata com os modelos que
  a empresa já usa, com menos esforço?
- **Desenho.** Os 4 problemas com os rótulos reais; T são os modelos em
  produção ou a melhor reprodução deles.
- **Critério.** O mesmo da F3-H6.
- **Pré-requisitos.** Seção 7.

## 5. O que sai da fila

- **Ablações de arquitetura do backbone** (tamanho, posição, normalização). A
  H8 mostrou que não mudam o downstream.
- **Novas comparações de AUC nas 4 tarefas antigas com o motor v1,** depois da
  etapa 0.
- **F2-H4 e F2-H5** deixam de existir como hipóteses separadas: viram a
  F3-H5 e a F3-H4.
- **F2-H8 (transferência entre empresas)** continua adiada até a etapa 4.
- **Fusão conjunta com as 97 features** (sugerida na avaliação de 13/11) vira
  o braço J da bateria, e não uma hipótese própria.

## 6. Ordem, custo e pontos de decisão

| Etapa | O que roda | Custo estimado | Decisão ao fim |
|---|---|---|---|
| 0 | F2-H10 e F2-H11 | 9 a 10 h de L4 | Continuar ou parar a linha de LLM |
| 1 | F3-H1 e F3-H2 | CPU local; 15 min de L4 | Onde o motor atual já empata |
| 2 | Motor v2; F3-H3, F3-H4, F3-H5, F3-H6 | Implementação; menos de 1 h de L4 por pré-treino | Veredito de M1 na H&M |
| 3 | F3-H7 | A estimar com a base escolhida | Motor pronto para várias fontes |
| 4 | F3-H8 | Depende do ambiente da empresa | Uso interno |

Os custos da etapa 2 em diante são estimativas, sem medição.

Pontos de decisão:

1. **Depois da etapa 1.** Se a clusterização com o embedding atual já empatar,
   o v2 não precisa mirar nela.
2. **Depois da F3-H4.** Se a recomendação direta não superar nem o R1, a
   promessa da solução única exclui recomendação, que segue com o ranqueador
   específico.
3. **Depois da F3-H6.** Veredito de M1 na H&M. Os pré-requisitos da etapa 4
   correm em paralelo desde já, porque dependem da empresa.

## 7. Pré-requisitos da base da empresa

1. **Ambiente aprovado para os dados.** A base da empresa não deve ir para o
   Colab pelo túnel sem aprovação de segurança e de LGPD. É preciso definir
   onde o treino roda (nuvem da empresa, com GPU).
2. **Dicionário de dados** das tabelas de pedidos, itens, cadastro, crédito,
   navegação e incentivos, com volumes. Basta o esquema; não precisa de dados.
3. **Definição dos 4 problemas com os donos de cada um:** rótulo, população,
   data de corte e o modelo que está em produção hoje.
4. **Motor v2 validado** (F3-H3), porque o v1 não comporta a base.

## 8. Decisões em aberto

1. **Margens de empate.** Valem as da seção 3, salvo indicação em contrário
   antes do primeiro resultado.
2. **Ambiente para a base da empresa** (pré-requisito 1).
3. **Persistência no Colab.** Montar o Google Drive no notebook evita refazer
   horas de GPU quando o runtime cai (`scripts/colab_persistencia.sh`).
4. **Licença do TabPFN-3.5,** ainda não aceita na conta da Prior Labs; sem
   ela a F2-H11 fecha só com o TabICL.
