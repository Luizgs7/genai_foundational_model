---
name: motor-v2-nova-base
description: Como colocar uma base de clientes nova no motor v2 (evento composto) só com adapter e YAML — fontes e naturezas (itens, evento, estado, estático), catálogo, ordem do pipeline, testes e as armadilhas já encontradas na H&M, Synerise, Olist e base sintética B2B.
---

# Base nova no motor v2: adapter + YAML

O motor v2 (`MOTOR_V2.md`) recebe qualquer base de clientes sem mudar código do
motor. A parte específica de cada empresa é só o **adapter**
(`adapters/adapter_<base>.py`) e o **YAML** (`configs/<base>.yaml`). Se uma base
exigir mudar `pipeline/common/`, isso é um achado (anotar em
`EXPERIMENTOS_HM.md`), não um ajuste silencioso: a promessa M2 do projeto é
justamente "só configuração".

## 1. O adapter

Saídas, sempre:
- `runs/<base>/eventos_canonicos.parquet`: a fonte-alvo (a que vira rótulos,
  features tradicionais e o alvo de itens do v2). Colunas mínimas:
  `evento_id` (Int64, único e ordenado por entidade e data), `entidade_id`
  (Int32), `data_evento` (datetime), o id do grupo (pedido/cesta), o item, os
  campos categóricos e numéricos do YAML, o campo de valor e os campos estáticos.
- `dados_<base>/preparado/<fonte>.parquet`: uma tabela por fonte extra, com
  `entidade_id` e a coluna de data declarada no YAML, ordenada por entidade e data.

Regras:
- Ids de texto viram inteiros **antes** de qualquer join grande (ver
  `adapters/compactar_hm.py`); a máquina local tem 15 GB, e um join de dezenas
  de milhões de linhas com texto já derrubou o VS Code. Jobs pesados locais
  rodam com `scripts/rodar_limitado.sh -m 7G`.
- Tudo o que o adapter deriva (ex.: `faixa10 = preco // 10`, catálogo top-N)
  é calculado **só com dados anteriores ao corte** quando depende de
  frequência, para não vazar o futuro (Synerise: catálogo = top 100 mil SKUs
  por compra+carrinho antes do corte).
- Campos que só existem depois do evento (nota de avaliação, atraso de
  entrega) podem ir na fonte-alvo **apenas como campo de rótulo**
  (`campo_futuro`), nunca como feature ou entrada da mesma posição.
- Bases com licença restrita (Synerise CC BY-NC 4.0) ficam no `.gitignore`.

## 2. O YAML

Blocos (exemplos reais: `configs/synerise.yaml`, `configs/olist_v2.yaml`):
- `campos_categoricos` (`fechado` ou `top_n_outros` com `top_n`),
  `campos_numericos` (`n_buckets`, opcional `agrupar_por`), estáticos,
  `ancora` (`cesta`), `splits` (`val_window_days`, `test_window_days`).
- `tarefas_downstream`: receitas `churn` (`dias_escolhido`), `campo_futuro`
  (`campo_fonte`, `limiar`, `operador`), `ltv`. Pergunta nova = entrada nova aqui.
- `features_tradicionais`: o braço T genérico (`janelas_dias`,
  `campos_composicao`, `fontes_extras: v2` para agregar as fontes extras).
- `v2`: `janela` (512 posições), `catalogo` (item + atributos) e `fontes`.

As quatro naturezas de fonte:

| Natureza | Quando usar | Chaves principais |
|---|---|---|
| `itens` | evento com várias linhas (pedido, cesta) | `agrupar_por`, `item`, `numericos_linha`, `valor_linha`, `alvo: true` na fonte-alvo |
| `evento` | eventos simples de alta frequência (busca, visita, carrinho, pagamento) | `sessao_minutos: 30`, `categoricos_linha`, opcional `item` |
| `estado` | valor que vale até mudar (limite de crédito, atraso da última entrega) | `numericos`, `categoricos`; não ocupa posição |
| `estatico` | atributo fixo do cliente | vai em `campos_estaticos_*`, fora da sequência |

## 3. Ordem do pipeline (cada etapa é idempotente)

```
adapter → tarefa7_splits → tarefa4_discretizacao → tarefa8_rotulos
        → features_tradicionais → v2_serializacao --dados v2
        → v2_pretreino --retomar → v2_extrair
        → tarefa13_treinar_fusao (E) · tarefa14_gbdt --features trad (T) · --features hibrido (H)
        → avaliacao.py (bootstrap por cliente)
```
Modelo de fila: `scripts/fila_colab_synerise.sh` (com e sem fontes extras via
`--sem_fonte`, T sem as fontes via `--params_de` + `--sem_prefixo`).
Preparo local: `scripts/preparar_beleza.sh`.

## 4. Conferências antes de olhar resultado

1. `python3 pipeline/common/v2_testes.py`: causalidade, censura, aprendizado mínimo.
2. `meta.json` da serialização: entidades, posições, linhas, janela; a cobertura
   da janela deve ser 100% das âncoras (ou anotar quantas truncam).
3. Taxa de positivos de cada tarefa e distribuição de classes (o LTV da
   Synerise colapsou em 2 classes; tarefa degenerada não entra no veredito).
4. Linha da fonte-alvo **sem item do catálogo** não pode virar alvo de item (nem
   no treino nem na validação: já deu perda de validação infinita na Synerise).
5. Curva de validação do pré-treino: com todas as fontes, a Synerise sobreajustou
   depois da época 2; a época é escolhida pelo critério de validação.

## 5. Colab (GPU)

- Túnel cloudflared muda a cada sessão; o runtime some. Toda fila chama
  `persistir` (Drive) a cada etapa e a cada 10 min; um runtime novo roda
  `restaurar`. Pastas novas de resultado precisam entrar em
  `scripts/colab_persistencia.sh`.
- Treinos longos aceitam `--retomar`; ao retomar, o estado do RNG volta para a CPU.
- Nunca dois LightGBM de 12 threads ao mesmo tempo (contenção: os dois ficam
  muito lentos). `pkill -f <nome>` não vai dentro da linha de comando do ssh.
- O tar de código enviado no início da sessão não contém arquivos criados
  depois: enviar scripts e configs novos à parte.
- Credenciais (ex.: chave do TabPFN) só em arquivo `chmod 600` no runtime;
  nunca em commit, log ou memória.
