# genai
Repositório destinado ao projeto de conclusão de curso Especialização em Inteligência Artificial Generativa, da Universidade Federal do Paraná, da turma 2026-27


# To Dos

# Definir Prof. orientador

# Estrutura de Dados

1 - Alinhamento interno com lideranças para liberação de dados internos para treino


# Kickoff — Customer Foundation Model
**Especialização em Gen AI · UFPR · Time de 5 pessoas**

> Este documento cobre apenas o início do projeto. O objetivo não é seguir um roteiro fechado, mas garantir que o time chegue à Fase 1 com as perguntas certas e os dados na mão.

---

## Leituras obrigatórias

Todos os membros devem ler antes do primeiro encontro técnico.

**Arquitetura e motivação**
- [ ] Nubank — Visão geral do Customer Foundation Model → https://building.nubank.com/pt-br/entendendo-as-financas-dos-nossos-clientes-por-meio-de-modelos-fundacionais/
- [ ] Nubank — Interface de serialização de transações → https://building.nubank.com/pt-br/definindo-uma-interface-entre-dados-de-transacoes-e-modelos-fundacionais/
- [ ] Nubank — Fine-tuning e joint fusion → https://building.nubank.com/pt-br/ajuste-fino-de-modelos-de-usuarios-de-transacoes/
- [ ] NeoSpace → https://www.neospace.ai/neoldmpaper

**Papers técnicos** (leitura diagonal — abstract, introdução e conclusão são suficientes para o kickoff)
- [ ] TabBERT (IBM, 2021) — Transformer para sequências de transações → https://arxiv.org/abs/2011.01843
- [ ] NoPE (Kazemnejad et al., 2023) — Remoção de positional encoding → https://arxiv.org/abs/2305.19466
- [ ] FlashAttention (Dao et al., 2022) — Atenção eficiente para contexto longo → https://arxiv.org/abs/2205.14135

---

## Captação e mapeamento dos dados

Antes de escrever qualquer código, o time precisa responder: **quais dados existem de fato e em qual formato estão?**

### Dados Sintéticos

- [] Dados https://we.tl/t-UsfDMKTdUOSgE65n

---

## Setup inicial

### Repositório e ambiente
- [ ] Criar conta Google para assinar Colab Pro
- [ ] Criar repositório Git com `CLAUDE.md` e `ROADMAP.md` na raiz
- [ ] Configurar `.gitignore` — dados reais nunca no repositório
- [ ] Testar ambiente GPU: Colab Pro ou cloud spot (A100/T4)
- [ ] Instalar dependências base: `polars`, `torch`, `transformers`, `tokenizers`, `accelerate`, `wandb`
- [ ] Criar conta W&B e projeto `customer-embedding` — todos os membros com acesso

### Divisão inicial de responsabilidades
- [ ] Johnny: levantamento e acesso às fontes de dados
- [ ] Membro 3: exploração da serialização (protótipo em notebook)
- [ ] Membro 4: leitura aprofundada dos papers de arquitetura
- [ ] Membro 5: pesquisa de ferramentas e baselines (TabPFNv2, LightGBM, DCNv2)

---

## Perguntas técnicas em aberto

O time deve pesquisar, experimentar e trazer respostas para o próximo encontro. Não há resposta certa definida — são decisões que dependem dos dados reais.

### Sobre os dados

1. **Qual é a densidade de eventos por cliente?** Clientes com poucos eventos (<5) devem ser excluídos ou tratados de forma especial?
2. **Qual o horizonte histórico ideal para a sequência?** Usar tudo ou só os últimos N meses? Qual N?
3. **Como tratar clientes novos** (poucos eventos) vs. clientes antigos (sequências muito longas)?
4. **As fontes têm o mesmo `cliente_id`?** Se não, como fazer o match entre elas?
5. **Quais features do CRM são estáticas** (entram no DCNv2) e quais são dinâmicas (poderiam entrar na sequência)?

### Sobre a serialização

6. **Quantos buckets usar para o valor da transação?** 5, 10 ou 20? O que a distribuição real dos dados sugere?
7. **A descrição textual da transação** (ex: "Netflix", "Mercado Livre") deve entrar como texto livre (BPE) ou como categoria mapeada? O vocabulário de descrições é grande demais para tokens especiais?
8. **Como representar ausência de dados?** Um cliente sem navegação digital — como isso aparece (ou não aparece) na sequência?
9. **Qual a granularidade temporal ideal?** Eventos por dia, por semana, por transação individual?

### Sobre a arquitetura

10. **Qual tamanho de modelo cabe no Colab Pro (16GB VRAM)?** Quantas camadas, qual dimensão de embedding, qual batch size — com FlashAttention ativo e bfloat16?
11. **NoPE funciona melhor que positional encoding padrão no nosso domínio?** Vale fazer o experimento de ablação logo no início ou apenas no final?
12. **Qual o critério de parada do pré-treino?** Número de épocas, perplexidade mínima, ou estabilização da loss?
13. **Como avaliar a qualidade do embedding antes do fine-tuning?** Clustering faz sentido? Que métricas usar?

### Sobre as tarefas downstream

14. **Como definir churn para este negócio?** Ausência de compra em 30, 60 ou 90 dias? Depende do segmento?
15. **Temos rótulos de churn históricos disponíveis** para supervisionar o fine-tuning?
16. **Para demanda de SKU: qual granularidade?** SKU individual, categoria, ou família de produtos?
17. **O sinal do Customer Embedding realmente ajuda na previsão de demanda agregada**, ou o volume de vendas é dominado por fatores externos (promoção, sazonalidade)?

### Sobre baselines e comparação

18. **LightGBM com features manuais do CRM** vai ser o baseline principal — quais features construir para que seja um baseline justo (não fraco demais)?
19. **Vale implementar o TabPFNv2 como baseline** mesmo com o limite de 10k amostras, para ter um número comparável da literatura?

---

## Próximos passos concretos

Ao final do kickoff, o time deve sair com:

- [ ] Inventário de dados preenchido
- [ ] Acesso confirmado a pelo menos uma fonte real (ou gerador sintético pronto)
- [ ] Ambiente GPU funcionando com `import torch; torch.cuda.is_available()` retornando `True`
- [ ] Cada membro com W&B configurado
- [ ] Data do próximo encontro marcada (sugestão: 1 semana)
- [ ] Cada pergunta técnica em aberto atribuída a um membro responsável por pesquisar

---

*Próximo documento após o kickoff: `ROADMAP.md` com as fases detalhadas de implementação.*
