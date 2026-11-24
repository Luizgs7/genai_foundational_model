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
- **A base da empresa não entra nos experimentos.** Decisão de 2026-11-13: a
  estratégia e a arquitetura são validadas e estressadas em bases públicas e
  sintéticas. Para a empresa vai só o código, no fim. O perfil da base dela
  (seção 1) serve de especificação do que o motor precisa suportar.

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

As margens foram confirmadas pelo responsável pelo projeto em 2026-11-13.
Valem para toda a fase 3 e só mudam antes de um resultado, com registro aqui.

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

### Bases de dados

Só bases públicas e sintéticas. Cada uma cobre uma parte do perfil da base da
empresa:

| Base | O que tem | O que estressa | O que não cobre |
|---|---|---|---|
| H&M (31,8 milhões de itens, 1,36 milhão de clientes) | Compras e cadastro | Escala, catálogo de 105 mil artigos e os 4 problemas | Outras fontes de evento; pedidos grandes |
| Olist (112 mil itens; entidade = vendedor) | 9 tabelas: itens, pagamentos, avaliações e entregas | Vários tipos de evento na mesma linha do tempo; históricos longos (um vendedor tem 2.025 vendas); rótulos reais | Volume |
| Sintética de beleza B2B (gerador próprio, a construir) | Uma empresa de perfumaria e beleza com todos os tipos de dado do mesmo cliente: pedidos, navegação, crédito, incentivos e cadastro | Vários tipos de dado juntos, formas extremas (centenas de itens por pedido, milhares de pedidos por cliente) e estrutura plantada, com gabarito | Não é dado real: não serve de evidência de valor de negócio |
| Synerise, do RecSys Challenge 2025 (aprovada em 2026-11-13; a baixar na etapa 3) | Compra, carrinho, visita de página e busca de um varejista real | Navegação em dado real e em volume; placar público de representações universais | Só eventos de site; licença CC BY-NC 4.0, só para pesquisa |

Nenhuma base pública reúne, para o mesmo cliente de uma mesma empresa, todos
os tipos de dado da seção 1. Por isso a validação dessa hipótese combina três
bases: a sintética (todos os tipos, com gabarito), a Synerise (navegação real)
e a Olist (tabelas heterogêneas reais, em volume pequeno).

#### Base sintética de beleza B2B: especificação

Uma empresa fictícia de perfumaria e beleza que vende para revendedores e
lojas. O gerador é novo (`gerar_dados_sinteticos_beleza_b2b.py`), com semente
fixa e dois tamanhos: um pequeno, para desenvolvimento, e um grande, para
estresse.

- **Clientes.** Revendedores pequenos, revendedores líderes e lojas. Cada um
  tem variáveis latentes: segmento, porte, cadência de compra, sensibilidade a
  desconto, engajamento digital e risco de crédito.
- **Catálogo.** Dezenas de milhares de SKUs em categorias de beleza
  (perfumaria, maquiagem, cabelos, corpo e banho, cuidados com a pele,
  masculino, infantil, kits e presentes), com marca, linha, variante e preço
  de lista. Há lançamentos e descontinuações a cada ciclo de campanha.
- **Calendário comercial.** Ciclos de campanha de cerca de 3 semanas, com SKUs
  em promoção, e picos sazonais (Dia das Mães, Dia dos Namorados, Dia dos
  Pais, Black Friday e Natal).
- **Tipos de dado, todos do mesmo cliente:**

  | Tipo | Conteúdo | Natureza |
  |---|---|---|
  | Pedidos | Cabeçalho (canal, valor, pagamento) e linhas (SKU, quantidade, preço, desconto); de 2 a centenas de linhas | Evento com itens |
  | Navegação | Busca, visualização de produto, visualização de campanha, carrinho e abandono, no aplicativo e no site | Evento simples, de alta frequência |
  | Crédito | Limite, valor disponível, atraso de pagamento e bloqueio | Estado que muda no tempo |
  | Incentivo | Adesão ao programa, nível, pontos no ciclo e meta atingida | Estado e eventos |
  | Cadastro | Tipo de cliente, região, data de cadastro e canal de aquisição | Atributo fixo |

- **Estrutura plantada, com gabarito:**
  - segmentos latentes, que definem o mix de categorias e a faixa de preço;
  - reposição periódica e regras de compra conjunta (perfume e loção da mesma
    linha; xampu e condicionador);
  - navegação que antecede a compra: ver uma categoria ou um SKU aumenta a
    chance de comprá-lo no pedido seguinte;
  - crédito: pouco limite disponível reduz o pedido, e atraso de pagamento
    aumenta o risco de churn;
  - incentivo: quem está perto da meta no fim do ciclo faz um pedido de
    complemento. É um padrão que depende da ordem e do momento dos eventos;
  - sensibilidade a desconto, que define quem compra em promoção.
- **Regra.** As variáveis latentes nunca entram no motor nem na régua
  tradicional. Servem só para avaliar.

A base sintética prova capacidade: que o motor aproveita cada tipo de dado
quando há sinal, e que aguenta as formas extremas. A evidência de valor de
negócio vem das bases reais (H&M e Synerise).

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
  - **Refinamento de 2026-11-14, feito na validação e antes do teste.** Nos
    espaços reduzidos por PCA (T2, E e H), padronizar dimensão a dimensão dá
    peso igual a componentes de pouca variância; na validação isso derrubou o
    T2 (índice de 0,016 contra 0,049 sem essa padronização). Para a régua não
    ficar artificialmente fraca, cada espaço tem duas formas (padronizado por
    dimensão, ou só posto na mesma escala por bloco) e a forma de cada espaço
    é escolhida na validação pelo índice principal. O teste usa essas
    escolhas, fixas.
- **Critério, escrito antes de rodar.**
  - Empate: o índice de E ou de H é pelo menos 95% do melhor entre T1 e T2, e
    o ARI não fica mais de 0,05 abaixo.
  - Forma forte: E ou H acima do melhor T, com IC excluindo o zero.
- **Custo.** Só CPU local.

### Etapa 2 — motor v2: evento composto

Um redesenho do que entra e do que sai do backbone. Corrige os diagnósticos 2
e 3 da seção 1 e é pré-requisito para as formas da base da empresa. O desenho detalhado
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

#### F3-H3b — composição do próximo pedido por atributo (variante, registrada em 15/11)

- **Origem (post hoc, declarado).** A F3-H3 rodou e **não atendeu** o critério
  de qualidade no teste: o v2 ganhou em churn (+0,007) e LTV (+0,011), mas
  perdeu em próxima categoria (−0,025) e próxima online (−0,010). Na validação
  as quatro ficavam dentro de 0,005. A diferença se concentra nos meses do
  fechamento das lojas (março a junho de 2020) e diminui depois. Esta variante
  foi desenhada **depois de ver o teste**. Por isso o resultado dela é
  reportado como evidência mais fraca que a de uma hipótese registrada antes.
- **Mudança, uma só.** Alvos genéricos de composição: para cada atributo do
  catálogo, a distribuição dos valores dele nas linhas do próximo pedido
  (`--alvos_atributos`). Na H&M, o artigo troca de coleção (69% dos artigos do
  teste não existiam no treino), mas índice, grupo, tipo, cor e linha
  permanecem. O canal continua **fora** dos alvos (teste de transferência).
- **Desenho.** Igual à F3-H3: 8 épocas, melhor época pela validação, mesmas
  âncoras e fusão, 3 sementes; o teste é avaliado uma vez para esta variante.
- **Critério.** O da F3-H3 (a até 0,005 da fusão do v1 nas 4 tarefas) e,
  além dele, sem perder o ganho em churn e LTV (a até 0,005 do v2).
- **Resultado (16/11): não atendido.** Categoria volta para −0,0025 do v1, mas próxima online fica a −0,0092 (canal fora dos alvos) e o LTV perde 0,0058 do ganho do v2. Ver `EXPERIMENTOS_HM.md`.

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
- **Ajustes registrados em 19/11, antes de rodar.**
  - **R1 corrigido.** O R1 da F2-H3 deixava sem lista os clientes sem recompra
    recente (41% na validação, 39% no teste), em vez de completá-los com os
    populares. Corrigido: 0,0229 na validação e **0,0259 no teste** (era
    0,0229). O critério "acima do R1" passa a usar o valor corrigido.
  - **Dois braços.** M: só o modelo, catálogo inteiro. MH: a pontuação do
    modelo mais um termo do histórico do próprio cliente,
    beta × log(1 + nº de compras do artigo) + gama / (1 + dias desde a última / 7).
    beta e gama saem de uma grade na semana de validação (semente 0) e ficam
    congelados no teste. A F3-H7 e a F3-H8 mostraram que o v2 sozinho não
    personaliza (o pedido entra como a média das linhas); o MH testa se a
    pontuação do modelo acrescenta algo à recompra.
  - **Itens vistos até o corte.** O backbone de cada semana trata como
    conhecidos os artigos vendidos até a véspera, e não só os do treino original
    (até 15/09/2020 são 103.791 artigos, contra 78.989 até 27/12/2019).
  - Validação com 1 semente (só para a grade); teste com 3.
- **Resultado (20/11): necessário atendido só com o histórico; empate não.** M 0,0075 (23% do R2); MH 0,0271, acima do R1 corrigido (+0,0012, IC sem o zero) e em 84% do R2. A recomendação segue híbrida. Ver `EXPERIMENTOS_HM.md`.

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
- **Como roda (registrado em 20/11, antes de rodar).** O backbone é o v2 da
  F3-H3 (`v2_pretreino`, 8 épocas), cujos alvos já são os genéricos acima:
  canal e desconto nunca foram alvo. A cabeça leve é a fusão DCNv2 da H4,
  treinada **uma pergunta por vez** (a fusão da F3-H3 era conjunta e escolhia
  a época pelo churn, o que desfavorece as demais). A régua T é o LightGBM com
  HPO de cada pergunta: `h10_trad_lgbm` em churn e canal, `f3h1_trad_lgbm`
  (com o histórico de desconto) em desconto. Empate = limite inferior do IC
  95% de E − T acima de −0,01. Já se sabe, pela fusão conjunta da F3-H3, que
  churn fica no limite (−0,0097 [−0,0100; −0,0094]) e canal fora (−0,0168);
  a fusão por pergunta é a forma justa do teste, não uma segunda chance
  escolhida depois de ver o resultado.
- **Resultado (20/11): não atendido.** E − T no teste: churn −0,0186 (a
  fusão conjunta da F3-H3 fica em −0,0097, no limite), canal −0,0143,
  desconto −0,0126 (o v1 ficava em −0,0175). O v2 transfere melhor que o v1
  para perguntas fora dos alvos, mas o vetor congelado com cabeça leve segue
  fora da margem; só o híbrido empata. Ver `EXPERIMENTOS_HM.md`.

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
- **Mudança de desenho, aprovada em 20/11 antes de rodar: duas etapas.** O
  segundo corte refaz rótulos, features, HPO, pré-treino e a semana de
  recomendação. Ele só muda o veredito se o primeiro corte empatar.
  - **Etapa A — corte atual, motor v2.** A mesma representação nos 4
    problemas: o backbone `v2_pretreino` (F3-H3), congelado.
    - Recomendação (protocolo da F2-H3, margem 5% relativo sobre o R2):
      E = modelo + histórico (MH da F3-H4, já medido); H = o LambdaRank do R2
      mais três features do v2 (log-probabilidade do artigo, a mesma relativa
      ao melhor candidato do cliente e o posto), com folhas e árvores
      escolhidas na validação e 3 sementes do LightGBM, como o R4.
    - Churn e desconto (margem 0,01 de AUC): E = cabeças leves da F3-H5; H =
      LightGBM com as 97 features e o PCA-32 do vetor v2, HPO de 40
      tentativas, 3 sementes, como os híbridos da H10 e da F3-H1. As outras
      3 tarefas antigas entram como complemento.
    - Clusterização (protocolo da F3-H2, mesmos desfechos e escolhas de
      forma na validação): E = PCA-32 do vetor v2; H = T2 e E juntos.
    - Esforço: a tabela da seção 3 para cada problema e braço.
    - J (ajuste fino conjunto) fica fora: é bônus de acurácia e não muda o
      veredito de empate.
  - **Etapa B — segundo corte, 6 meses antes.** Só roda se a Etapa A
    empatar nos 4 problemas (E ou H). Se não empatar, o veredito de M1 na
    H&M sai da Etapa A: não atendida no corte atual. O segundo corte não
    poderia virar esse resultado, porque o critério exige empate nos dois.
- **Resultado da etapa A (21/11): não atendida, pela estabilidade da
  clusterização.** H empata em recomendação (−0,3%), churn (−0,0065) e
  desconto (+0,0010). Na clusterização, o índice do v2 supera o RFM
  (0,0796 contra 0,0694), mas o ARI fica 0,087 abaixo (limite: 0,05). A etapa
  B não roda; M1 na H&M: não atendida no corte atual. O H empata sem reduzir
  o esforço por problema; o E reduz o esforço, mas só empata no índice da
  clusterização. Ver `EXPERIMENTOS_HM.md`.

### Etapa 3 — estresse e generalização

#### F3-H7 — vários tipos de dado do mesmo cliente, numa empresa só

- **Pergunta.** O motor v2 recebe, só com configuração, todos os tipos de dado
  que uma empresa tem sobre o mesmo cliente, e a representação aproveita cada
  um deles sem engenharia de features?
- **Por que importa (M2 e M1).** É o contexto real do produto: uma empresa,
  um cliente, e vendas, cadastro, crédito, navegação e incentivos ao mesmo
  tempo. A H&M só tem compras e cadastro.
- **Desenho.** Três camadas de evidência:
  1. **Base sintética de beleza B2B: todos os tipos, com gabarito.** A bateria
     dos 4 problemas roda com todas as fontes e, depois, retirando uma fonte
     por vez. A régua T recebe os agregados genéricos de todas as fontes.
  2. **Synerise: navegação real, em volume.** Compra, carrinho, visita de
     página e busca do mesmo usuário. A bateria roda com e sem a navegação. A
     comparação com o placar público do desafio é informativa, e não critério.
  3. **Olist: tabelas reais e heterogêneas.** Vendas, pagamentos, avaliações e
     entregas do vendedor entram como tipos de evento. Teste só funcional,
     porque o volume é pequeno.
- **Critério, escrito antes de rodar.**
  - M2: nas três bases, cada fonte entra com adapter e um bloco de YAML, sem
    mudança no motor. Os quatro formatos são aceitos: evento com itens, evento
    simples, estado que muda no tempo e atributo fixo.
  - M1, na base sintética: com todas as fontes, E ou H empata com T nos 4
    problemas. E a retirada de cada fonte derruba, por mais que a margem, o
    problema que depende dela pelo gabarito (navegação: recomendação; crédito:
    churn; incentivo: pedido nos 7 dias seguintes, tarefa complementar;
    desconto: propensão a desconto).
  - M1, em dado real (Synerise): incluir a navegação melhora E em pelo menos um
    problema por mais que a margem, e E ou H empata com T.
- **Resultado da camada 1 (17/11).** M2 atendido. M1 não: H empata com T nas 5 tarefas supervisionadas, mas segmentos (H −5,9%) e recomendação (0,036 contra 0,300) ficam fora; nenhuma retirada de fonte passa a margem no problema esperado (todas significativas, todas abaixo de 0,01). Ver `EXPERIMENTOS_HM.md`.
- **Camada 2, Synerise: desenho registrado em 21/11, antes de rodar.**
  - **Base.** O arquivo bruto do desafio (`synerise_dataset.tar.gz`, 23/06 a
    08/12/2022). Entram os 909 mil clientes com pelo menos uma compra e todos
    os eventos deles: 2,3 milhões de compras (1,42 milhão de cestas), 3,6
    milhões de adições e 1,6 milhão de remoções do carrinho, 5,4 milhões de
    buscas e 50 milhões de visitas de página.
  - **Fontes no motor v2.** `compra` (itens, alvo; cesta = cliente e
    instante), `carrinho` (adição e remoção, evento com item), `busca` e
    `visita` (eventos simples em sessões de 30 min). O conteúdo da busca (um
    vetor quantizado) e o identificador da página (milhões de valores) não
    entram: só a ocorrência e a sessão.
  - **Catálogo.** Os 100 mil SKUs mais frequentes em compra e carrinho no
    período de treino viram item do modelo. Os demais entram só com os
    atributos da linha: categoria (as 50 mais compradas no treino mais
    "outras") e faixa de preço (as 100 faixas do arquivo em 10 grupos). Sem o
    corte, a saída teria centenas de milhares de itens e não caberia na L4.
  - **Splits.** Teste nos últimos 56 dias, validação nos 28 anteriores.
  - **Problemas.** Desconto e canal não existem na base, e a recomendação
    pede um gerador de candidatos específico, que fica fora. A bateria tem:
    churn em 30 dias (AUC), próxima categoria (as 50 mais "outras",
    accuracy), faixa de gasto em 30 dias (LTV, com a faixa de preço como
    valor, accuracy) e a clusterização sobre esses desfechos (protocolo da
    F3-H2). Margens da seção 3.
  - **Braços.** T: LightGBM com os agregados genéricos de todas as fontes
    (`fontes_extras: v2`) e HPO de 40 tentativas. E: vetor do v2 congelado
    com a fusão conjunta. H: T mais o PCA-32 do vetor. Todos com e sem a
    navegação: sem carrinho, busca e visita, só a compra. 3 sementes; teste
    avaliado uma vez.
  - **Critério (o da camada 2, acima).** A navegação melhora E em pelo menos
    um problema por mais que a margem (E completo − E só compra, IC 95%
    acima da margem), e E ou H empata com T com todas as fontes. O controle
    da camada 1 também roda: o quanto T ganha com a navegação.
- **Resultado da camada 2 (22/11).** M2 atendido. M1: a navegação melhora E em
  churn por mais que a margem (+0,022), a primeira parte do critério; mas o
  empate com T só sai do H nos três problemas supervisionados, e a
  clusterização não empata (H 92% do RFM com ARI 0,55; E com navegação, 36%). A
  navegação ajuda T tanto quanto E (+0,027 em churn). Ver `EXPERIMENTOS_HM.md`.
- **Camada 3, Olist: como roda (registrado em 22/11, antes de rodar).**
  Entidade = vendedor (3 mil; 112 mil linhas de venda, 2016-09 a 2018-09). Os
  quatro formatos: venda (itens, alvo), pagamento e avaliação (eventos),
  entrega (estado: dias de atraso contra a data prometida) e a UF do vendedor
  (atributo fixo). Tarefas do config antigo (`olist_sellers.yaml`): churn em
  30 dias, risco de avaliação negativa e risco de atraso no próximo pedido. T,
  E e H como na Synerise, com e sem as fontes além da venda, 3 sementes. O
  critério é só o de M2 (cada fonte por adapter e YAML, sem mudar o motor);
  os números são relatados, sem veredito de M1, porque o volume é pequeno.
- **Resultado da camada 3 (22/11): M2 atendido** nos quatro formatos, sem mudar
  o motor. Em churn, H empata com T (−0,0007); as tarefas de nota e de atraso
  ficam perto do acaso em todos os braços. Ver `EXPERIMENTOS_HM.md`.
- **Reserva.** Se as camadas 1 e 2 derem respostas opostas, o MBD (transações,
  geolocalização e diálogos de atendimento dos mesmos clientes de um banco)
  entra como segunda base real com vários tipos de dado. É uma base nova e
  depende de aprovação.

#### F3-H8 — formas extremas e estrutura conhecida, na base sintética de beleza

- **Pergunta.** O motor v2 aguenta as formas da base da empresa e recupera uma
  estrutura que nós plantamos?
- **Por que importa (M2).** Nenhuma base pública tem pedidos com centenas de
  itens nem clientes com milhares de pedidos. Sem esse teste, o primeiro
  contato com essas formas seria na empresa.
- **Desenho.** O tamanho grande da base sintética: pedidos de 2 a centenas de
  itens, clientes de 1 a milhares de pedidos e catálogo de dezenas de milhares
  de SKUs, com lançamentos a cada ciclo.
- **Critério.**
  - Forma: 100% dos eventos entram no contexto, sem truncamento, e o custo de
    pré-treino por pedido não cresce com o tamanho do pedido além do linear.
  - Estrutura: os segmentos do braço E recuperam os segmentos plantados pelo
    menos tão bem quanto os de T (ARI contra o gabarito, margem de 5%), e a
    recomendação direta recupera as regras de compra conjunta acima do R1.
  - Custo: o tempo medido permite projetar o treino para o volume da empresa.
- **Quando.** A parte de forma roda logo depois da F3-H3.
- **Resultado (16/11): forma e custo atendidos; estrutura não.** Cobertura 100%, custo linear no tamanho do pedido; segmentos E 0,095 contra T 0,451 (ARI); compra conjunta v2 0,0006 contra R1 0,0175; MAP@12 v2 0,008 contra R1 0,284. Ver `EXPERIMENTOS_HM.md`.

### Etapa 4 — entrega do código

Não é um experimento. A empresa recebe o código e roda no ambiente dela, com
os dados dela. O pacote só sai quando os critérios da seção 7 forem atendidos.

### Etapa 5 — próximos passos do artigo revisado (registrados em 23/11/2026)

Os próximos passos da seção 11 do artigo revisado, na ordem dele. O passo 1 é a
etapa 4 acima. Cada hipótese abaixo é registrada antes de rodar.

#### F3-H9 — ajuste fino conjunto do backbone v2 com a cabeça de cada tarefa

- **Pergunta.** Na fase 2, o ajuste fino conjunto foi o melhor uso do modelo
  fundacional (F2-H2, no v1), mas nunca rodou no v2. Ajustado junto com as
  tarefas, o vetor do v2 empata com T nas perguntas supervisionadas em que o
  vetor congelado não empata (F3-H5)?
- **Desenho.**
  - Backbone `v2_pretreino` (o da F3-H3, F3-H5 e F3-H6) + a cabeça DCNv2 da H4,
    treinados juntos, com as cinco tarefas da H&M ao mesmo tempo (churn,
    próxima categoria, gasto em 90 dias, próxima online e próxima com
    desconto; `configs/hm_clientes_f3h1.yaml`).
  - Taxa de aprendizado de 5×10⁻⁵ no backbone e 3×10⁻³ na cabeça, com
    aquecimento de 5% e decaimento linear, como na F2-H2.
  - 3 épocas sobre as âncoras de treino; a época é escolhida pela AUC de churn
    na validação; 3 sementes; teste avaliado uma vez.
  - Cada âncora usa o mesmo contexto da extração dos vetores (janela de 512
    posições, causal).
  - Script: `pipeline/common/v2_ajuste_conjunto.py`.
- **Régua.** T: o LightGBM da H10 (`gbdt/h10_trad_lgbm`) em churn, categoria,
  gasto e próxima online; o da F3-H1 (`gbdt/f3h1_trad_lgbm`) em desconto.
  Referência do vetor congelado: a fusão conjunta do v2 (`v2_fusao_s*`, F3-H3)
  e a fusão de desconto da F3-H5 (`f3h5_desc_s*`).
- **Critério, escrito antes de rodar.**
  - Principal: o ajuste conjunto empata com T (limite inferior do IC 95% de
    E − T acima de −0,01) em churn, próxima online e desconto, as três
    perguntas de AUC da bateria.
  - Secundário: supera o vetor congelado por pelo menos 0,005, com IC sem o
    zero, em pelo menos duas das cinco tarefas.
  - Esforço: horas de GPU por semente. O ajuste é um treino por conjunto de
    perguntas; uma pergunta nova exige ajustar de novo. Isso entra no registro
    de esforço, ao lado dos 100 s da cabeça leve.
- **O que muda com o resultado.** Se o principal for atendido, o caminho E
  passa a empatar nas perguntas supervisionadas da H&M, com mais custo por
  pergunta que a cabeça leve. O veredito de M1 na H&M não muda sozinho: a
  recomendação e a clusterização seguem como estão.
- **Custo estimado.** Uma época de ajuste perto de uma época de pré-treino
  (cerca de 7 min de L4) mais a validação; menos de 2 h de L4 nas 3 sementes.

## 5. O que sai da fila

- **Ablações de arquitetura do backbone** (tamanho, posição, normalização). A
  H8 mostrou que não mudam o downstream.
- **Novas comparações de AUC nas 4 tarefas antigas com o motor v1,** depois da
  etapa 0.
- **F2-H4 e F2-H5** deixam de existir como hipóteses separadas: viram a
  F3-H5 e a F3-H4.
- **F2-H8 (transferência entre empresas)** continua adiada.
- **F2-H9 (base da empresa)** sai dos experimentos: a empresa recebe o código
  no fim (etapa 4).
- **Fusão conjunta com as 97 features** (sugerida na avaliação de 13/11) vira
  o braço J da bateria, e não uma hipótese própria.

## 6. Ordem, custo e pontos de decisão

| Etapa | O que roda | Custo estimado | Decisão ao fim |
|---|---|---|---|
| 0 | F2-H10 e F2-H11 | 9 a 10 h de L4 | Continuar ou parar a linha de LLM |
| 1 | F3-H1 e F3-H2 | CPU local; 15 min de L4 | Onde o motor atual já empata |
| 2 | Motor v2; F3-H3, F3-H4, F3-H5, F3-H6 | Implementação; menos de 1 h de L4 por pré-treino | Veredito de M1 na H&M |
| 3 | F3-H7 e F3-H8 | A estimar com as bases | Motor pronto para várias fontes e para as formas da empresa |
| 4 | Entrega do código | Sem GPU | Levar para a empresa |

Os custos da etapa 2 em diante são estimativas, sem medição.

Pontos de decisão:

1. **Depois da etapa 1.** Se a clusterização com o embedding atual já empatar,
   o v2 não precisa mirar nela.
2. **Depois da F3-H4.** Se a recomendação direta não superar nem o R1, a
   promessa da solução única exclui recomendação, que segue com o ranqueador
   específico.
3. **Depois da F3-H6.** Veredito de M1 na H&M. Se a solução única não empatar
   nem como híbrida, a estratégia é revista antes das etapas 3 e 4.

## 7. Quando o código está pronto para ir à empresa

1. **M1 julgada em bases reais.** Veredito da F3-H6 na H&M e da F3-H7 na
   segunda base real, pelo critério de empate.
2. **M2 demonstrada.** O mesmo código roda em todas as bases da seção 3 só
   com adapter e YAML, incluindo vários tipos de dado do mesmo cliente
   (F3-H7).
3. **Formas da empresa testadas.** F3-H8 atendida na base sintética B2B, com a
   projeção de custo para o volume da empresa.
4. **Reprodutibilidade.** Testes de regressão, de vazamento e de causalidade
   do motor v2 passando; sementes e configurações versionadas.
5. **Independência do Colab.** O pipeline roda de ponta a ponta com um único
   comando por etapa, com retomada, em qualquer máquina com GPU.
6. **Roteiro de implantação.** Modelo de adapter e de YAML, o registro de
   esforço e um guia para definir os 4 problemas com os donos de cada um.

**Situação em 23/11/2026.**

| Critério | Situação |
|---|---|
| 1. M1 julgada em bases reais | feito: não atendida na H&M (F3-H6) nem na Synerise (F3-H7) |
| 2. M2 demonstrada | feito: quatro bases, quatro formatos, só adapter e YAML |
| 3. Formas da empresa testadas | forma e custo atendidos (F3-H8), com projeção de custo; estrutura não |
| 4. Reprodutibilidade | feito: `testes/testar.py` regenera a base mini e compara 28 artefatos com a referência, testa vazamento das variáveis e causalidade, censura e aprendizado do v2; versões em `requirements.txt` |
| 5. Independência do Colab | feito: `motor.py <config> <etapa>` (ou `tudo`), com retomada e registro de tempo por etapa; validado de ponta a ponta na CPU (base mini, 19 min) |
| 6. Roteiro de implantação | feito: `GUIA_IMPLANTACAO.md`, `configs/modelo_empresa.yaml`, `adapters/modelo_adapter.py` |

## 8. Decisões tomadas e pendências

Decididas em 2026-11-13:

1. **Margens de empate:** as da seção 3.
2. **Base da empresa:** fora dos experimentos; vai só o código.
3. **Base sintética:** com viés de perfumaria e beleza de varejo.
4. **Base da Synerise:** aprovada para a etapa 3.

Pendências operacionais:

1. **Persistência no Colab.** A pasta no Google Drive foi criada; falta
   confirmar a cópia num runtime acessível (`scripts/colab_persistencia.sh`).
2. **Licença do TabPFN-3.5.** Informada como aceita; a fila confere antes de
   rodar o TabPFN.
