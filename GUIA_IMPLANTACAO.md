# Guia de implantação na empresa

Como levar o motor à base da empresa e julgar, com o mesmo protocolo dos experimentos, se o modelo
fundacional vale para ela. Público: o time de dados da empresa. Para o porquê de cada escolha, ver o
artigo (seção 11) e `PLANO_FASE3.md`.

**O que o estudo deixou como recomendação.** O motor genérico está pronto: conecta a base por adaptador
e YAML e gera automaticamente as variáveis do modelo tradicional (T) para qualquer pergunta. O vetor do
cliente (motor v2) ainda não pagou o seu custo nas bases públicas: ao lado das variáveis, não mudou o
resultado; sozinho, ficou abaixo. Na empresa ele é avaliado de novo, problema a problema, com critério
fixado antes.

## 1. Antes de qualquer dado: os 4 problemas e seus donos

Para cada problema, uma conversa curta com quem decide sobre ele. Registre as respostas na tabela abaixo
**antes** de rodar qualquer coisa: elas viram o critério de sucesso.

| Problema | Dono | Pergunta exata (o rótulo) | Horizonte | Modelo atual e como é medido | Métrica e margem de empate | O que muda na decisão se o modelo novo empatar |
|---|---|---|---|---|---|---|
| Recomendação (upsell, cross-sell) | | ex.: itens do próximo pedido | | | MAP@12; 5% relativo | |
| Churn | | ex.: sem pedido em 90 dias | 90 dias | | AUC; 0,01 | |
| Propensão a desconto | | ex.: o próximo pedido usa desconto | | | AUC; 0,01 | |
| Clusterização | | ex.: segmentos para campanha | | | índice e estabilidade; 5% e ARI −0,05 | |

Perguntas que costumam faltar:
- **O rótulo existe nos dados?** Se a pergunta depende de algo que só se sabe depois (devolução, nota),
  ele entra como campo de rótulo, nunca como entrada.
- **Qual o horizonte?** A janela de teste tem de caber o maior horizonte, com folga (`splits` no YAML).
- **Qual é a régua hoje?** O modelo atual da empresa entra como mais um braço T, se as previsões dele
  puderem ser geradas nas mesmas datas.

## 2. Adaptador e YAML

1. Copie `adapters/modelo_adapter.py` para `adapters/adapter_<empresa>.py` e `configs/modelo_empresa.yaml`
   para `configs/<empresa>.yaml`. Os dois arquivos são comentados linha a linha.
2. O adaptador só lê, filtra, renomeia e grava. Converta ids de texto em inteiros antes de qualquer
   junção grande. Tudo o que depende de frequência (catálogo top-N, categorias top-K) usa só dados
   anteriores ao corte da validação.
3. Cada fonte de dado do cliente vira um bloco do YAML, numa de quatro naturezas: `itens` (pedido com
   linhas; uma delas é o alvo), `evento` (busca, visita, carrinho; agrupados em sessões de 30 min),
   `estado` (vale até mudar: limite de crédito, nível de incentivo) e atributos fixos do cliente.
4. Peça uma revisão independente do adaptador e do YAML (o agente `auditor-vazamento`, em `.claude/agents/`,
   ou um colega) antes de olhar qualquer resultado de teste.

Exemplos reais: `configs/synerise.yaml` (navegação em volume), `configs/olist_v2.yaml` (os quatro
formatos), `configs/beleza_b2b_pequeno.yaml` (pedidos, navegação, crédito e incentivo).

## 3. Rodar

Instale as dependências (`requirements.txt`; o PyTorch com GPU pela instrução oficial) e rode:

```bash
python3 testes/testar.py configs/beleza_b2b_mini.yaml     # o motor está íntegro nesta máquina?
python3 motor.py configs/<empresa>.yaml tudo               # o pipeline inteiro
```

Ou etapa por etapa (`preparar`, `serializar`, `pretreinar`, `extrair`, `fusao`, `gbdt`, `clusterizar`,
`avaliar`). Cada etapa pula o que já está pronto, então o mesmo comando retoma depois de uma queda; o
pré-treino retoma do último checkpoint. Opções úteis: `--dispositivo cuda`, `--epocas 8`,
`--n_trials 40`, `--threads N`.

**Custo esperado.** O pré-treino cresce em linha reta com o número de linhas de pedido e sessões. Na
proporção da base sintética grande (47 posições por cliente), 2 milhões de clientes dariam cerca de 45
minutos por época numa GPU L4, cerca de 6 horas para 8 épocas. É uma projeção; meça a primeira época.
O ajuste do LightGBM leva de 30 minutos a 2 horas de CPU por pergunta.

**Memória.** Numa máquina de pouca RAM, rode as etapas pesadas com `scripts/rodar_limitado.sh -m 7G`.

## 4. Ler o resultado

Saem, em `runs/<empresa>/`:

- `motor_bootstrap_test.json`: para cada pergunta, E (vetor + cabeça leve) e H (T + vetor) contra T, com
  intervalo de 95% por cliente;
- `motor_cluster/resultado_test.json`: índice e estabilidade dos segmentos de RFM, T, E e H;
- `motor_registro.json`: o tempo de cada etapa (registro de esforço).

Regra de leitura, a mesma dos experimentos:

- **Empate** quando o pior caso do intervalo (o limite inferior) fica acima de −0,01 (ou −5% em MAP@12
  e no índice). Diferença significativa abaixo da margem continua sendo empate.
- **E empata com menos esforço** → o vetor substitui o modelo daquele problema.
- **Só H empata** → o vetor não paga o seu custo naquele problema; fica o T.
- **Nada empata** → fica o T. O motor continua útil: as variáveis e o T já saem por configuração.

Registre o veredito de cada problema junto com o critério escrito no passo 1. Não troque o critério
depois de ver o resultado; uma ideia nova vira uma hipótese nova, testada num período ainda não visto.

## 5. Registro de esforço

Preencha por problema, com os números do `motor_registro.json` e do repositório:

| Problema | Caminho | Linhas de código específicas | Linhas de YAML | Horas de CPU | Horas de GPU | Peças a manter |
|---|---|---|---|---|---|---|
| | T | | | | | |
| | E | | | | | |
| | H | | | | | |

O custo fixo (pré-treino e variáveis) é contado à parte, uma vez para todos os problemas.

## 6. Antes de entregar uma mudança no código

```bash
python3 testes/testar.py configs/beleza_b2b_mini.yaml
```

Regenera a base mini do zero e compara 28 artefatos com a referência (`testes/referencia/`), roda o
teste de vazamento das variáveis e os testes de causalidade, censura e aprendizado do motor v2. Se a
mudança altera um artefato de propósito, grave a referência nova (`--gravar_referencia`) no mesmo
commit e explique o porquê.
