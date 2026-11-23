---
name: protocolo-avaliacao
description: Protocolo de avaliação do Customer Foundation Model — braços T/E/H/J, critério escrito antes de rodar, margens de empate, bootstrap Poisson por cliente, teste avaliado uma vez, registro de esforço, índice de clusterização e MAP@12 — com os erros de avaliação que já aconteceram no projeto.
---

# Protocolo de avaliação

Vale para toda hipótese nova (fase 3 em diante; texto-fonte: `PLANO_FASE3.md`,
seção 3). O veredito é sempre contra um critério **escrito antes de rodar**.

## 1. Antes de rodar

1. Registrar a hipótese em `PLANO_FASE3.md` (ou no plano da fase vigente):
   pergunta, braços, métrica, margem, critério de empate e o que conta como
   "não atendida". Commit antes de ter qualquer número de teste.
2. Escolhas de desenho (época, hiperparâmetros, forma do espaço de
   clusterização, beta/gama de uma mistura) são feitas **na validação** e
   ficam fixas no teste.
3. Variante criada depois de ver um resultado (ex.: F3-H3b) é registrada como
   variante nova, com data, e não substitui o veredito original.

## 2. Braços

| Braço | O que é |
|---|---|
| T | o modelo tradicional específico do problema, como a empresa faz hoje (LightGBM com as 97 features genéricas e HPO de 40 tentativas; R2 em recomendação; K-means sobre RFM em clusterização) |
| E | o vetor aprendido com uma cabeça leve (fusão DCNv2), sem engenharia de features |
| H | T + PCA-32 do vetor, mesma HPO |
| J | ajuste fino conjunto (bônus; um treino por problema) |

Controle obrigatório quando se mede o valor de uma fonte de dado: retirar a
fonte de **T também** (`--params_de` + `--sem_prefixo`). Na Synerise a navegação
ajudou E (+0,022 em churn), mas ajudou T tanto quanto (+0,027).

## 3. Margens e decisão

- 0,01 em AUC e acurácia (o dobro do ruído entre sementes e períodos, ~0,005).
- 5% relativos em MAP@12 e no índice de clusterização.
- **Empate** = limite inferior do IC 95% de (solução − T) acima de −margem.
- **Ganho** = limite inferior acima de 0 (significativo) e acima de +margem
  para ser ganho prático. Diferença significativa abaixo da margem é empate,
  e o texto deve dizer isso ("significativo, mas dentro da margem").
- Clusterização: índice = média de I(segmento; desfecho)/H(desfecho) sobre os
  desfechos futuros; empate exige índice ≥ 95% do melhor T **e** ARI entre
  sementes até 0,05 abaixo (`f3h2_clusterizacao.py`).
- Recomendação: MAP@12 da semana seguinte, protocolo da F2-H3
  (`f2h3_ranquear.py`, `f3h4_avaliar.py`).

## 4. Execução

- Mesmas âncoras e splits em todos os braços; nas comparações entre tarefas,
  âncoras comuns **por tarefa** (cruzar entre tarefas tira do churn justamente
  quem não voltou).
- 3 sementes; as predições das sementes são promediadas antes do bootstrap.
- IC 95% por **bootstrap Poisson por cliente** (`avaliacao.py`, B = 300;
  `f2h10_avaliar.py` para a média das 4 tarefas). Bootstrap por linha dá IC
  estreito demais.
- Teste avaliado uma vez. Reavaliar o teste só para corrigir erro de
  protocolo, com errata explícita.
- Registro de esforço por problema e braço: linhas de código fora do motor,
  linhas de YAML, horas de CPU/GPU, peças a manter, custo fixo à parte.

## 5. Erros que já aconteceram (procurar ativamente)

| Erro | Como apareceu | Regra |
|---|---|---|
| Censura unilateral | âncoras com janela de churn aberta no fim do teste entravam só se o cliente voltou (rótulo 0); taxa publicada 16% contra 25,4% real | âncora só é determinável se a janela fechou, para os dois rótulos |
| Régua enfraquecida | o R1 da F2-H3 deixava sem lista quem não tinha recompra (0,0229 → 0,0259) | conferir que T responde para 100% das âncoras avaliadas |
| Âncora por split | usar só a última transação de cada split inflava churn para 83% | uma âncora por evento rotulado |
| Duplo deslocamento | rótulos já deslocados passados ao HF, que desloca de novo | conferir a perda contra o acaso (ln do vocabulário) antes de treinar longo |
| Calendário no vetor | segmentos explicavam 57–71% do mês da compra | diagnóstico de calendário (`f3h2_diagnostico_calendario.py`) em toda clusterização |
| Tarefa degenerada | LTV da Synerise colapsou em 2 classes; nota/atraso da Olist perto do acaso | relatar, não usar no veredito |

## 6. Como relatar (`EXPERIMENTOS_HM.md`)

Tabela de métricas, tabela de diferenças com IC, veredito contra o critério
escrito, o que funcionou, o que não funcionou e por quê, custos, limitações,
caminhos dos arquivos de resultado e o script de reprodução.
