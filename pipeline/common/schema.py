"""Carregador do config declarativo por empresa/contexto (YAML).

Ver PLANO_PRODUTO.md pro desenho completo. Um config declara como as
colunas de uma empresa mapeiam pro schema canônico que todo o resto do
motor genérico (`pipeline/common/`) consome — nenhum script downstream
deste ponto conhece nomes de coluna de uma empresa específica.

Formato esperado (ver `configs/synthetic_agro.yaml` e
`configs/olist_sellers.yaml` pra exemplos completos):

    nome: str                       # identifica o run (pasta em runs/<nome>/)
    entidade_label: str             # rótulo humano ("cliente", "vendedor", ...)
    fonte_canonica: str             # caminho do CSV canônico já adaptado

    campos_categoricos:
      - nome: str
        estrategia: fechado | top_n_outros
        top_n: int                  # só se estrategia == top_n_outros

    campos_numericos:
      - nome: str
        agrupar_por: str | null      # campo categórico pra bucket por grupo
        n_buckets: int

    campos_estaticos_categoricos: [str, ...]   # pode ser vazia
    campos_estaticos_numericos: [str, ...]     # pode ser vazia

    splits:
      val_window_days: int
      test_window_days: int

    tarefas_downstream: [{tipo: str, ...params da receita...}, ...]

    fusao:
      campos_estaticos_categoricos: [str, ...]
      campos_estaticos_numericos: [str, ...]
"""

import yaml

CAMPOS_OBRIGATORIOS = [
    "nome", "entidade_label", "fonte_canonica",
    "campos_categoricos", "campos_numericos", "splits", "tarefas_downstream",
]


def carregar_config(path):
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    faltando = [c for c in CAMPOS_OBRIGATORIOS if c not in config]
    if faltando:
        raise ValueError(f"Config {path} sem campos obrigatórios: {faltando}")

    config.setdefault("campos_estaticos_categoricos", [])
    config.setdefault("campos_estaticos_numericos", [])
    config.setdefault("usa_recencia", True)
    config.setdefault("usa_mes", True)

    for campo in config["campos_categoricos"]:
        if campo.get("estrategia") == "top_n_outros" and "top_n" not in campo:
            raise ValueError(f"Campo categórico {campo['nome']!r} usa top_n_outros mas não define top_n")

    return config


def nomes_campos_categoricos(config):
    return [c["nome"] for c in config["campos_categoricos"]]


def nomes_campos_numericos(config):
    return [c["nome"] for c in config["campos_numericos"]]


def run_dir(config, subpasta):
    return f"runs/{config['nome']}/{subpasta}"


def tarefas_ativas(config):
    return config["tarefas_downstream"]


def tarefa_por_tipo(config, tipo):
    for t in config["tarefas_downstream"]:
        if t["tipo"] == tipo:
            return t
    return None
