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


# ---------------------------------------------------------------- I/O tabular
#
# A partir do experimento H&M (31,8M eventos) o motor grava os artefatos
# tabulares em Parquet (bem mais rápido e compacto que CSV nessa escala). A
# leitura aceita os dois formatos, então runs antigos em CSV continuam
# legíveis.

def ler_tabela(path, colunas=None):
    import pandas as pd

    if path.endswith(".parquet"):
        return pd.read_parquet(path, columns=colunas)
    return pd.read_csv(path, usecols=colunas)


def ler_fonte(config, colunas=None):
    """Lê o CSV/Parquet canônico do config (só as colunas pedidas)."""
    return ler_tabela(config["fonte_canonica"], colunas)


def caminho_tabela(config, subpasta, nome):
    """Caminho do artefato tabular `nome` (sem extensão) de um run: prefere o
    `.parquet` (formato atual); cai pro `.csv` se só existir o legado."""
    import os

    base = os.path.join(run_dir(config, subpasta), nome)
    if os.path.exists(base + ".parquet") or not os.path.exists(base + ".csv"):
        return base + ".parquet"
    return base + ".csv"


def gravar_tabela(df, config, subpasta, nome):
    import os

    out_dir = run_dir(config, subpasta)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, nome + ".parquet")
    df.to_parquet(path, index=False)
    return path


def config_pretreino(config):
    """Hiperparâmetros do pré-treino (seção opcional `pretreino:` do YAML).
    Os padrões reproduzem as constantes usadas nos runs synthetic_agro/olist."""
    padrao = {
        "batch_size": 32,             # nº de sequências por lote (se tokens_por_lote não for definido)
        "tokens_por_lote": None,      # orçamento de tokens por lote (lotes agrupados por tamanho)
        "n_epochs": 30,
        "lr": 3e-4,
        "warmup_steps": 100,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "seed": 0,
        "log_a_cada_n_passos": 20,
        "precisao": "bf16_autocast",  # bf16_autocast (pesos fp32) | bf16_puro (legado)
        "janelas_treino": "ultima",   # ultima (legado: últimos n_positions tokens) | todas
        "val_a_cada_passos": None,    # validação parcial periódica (amostra fixa)
        "val_amostra_entidades": 2000,
        "checkpoint_a_cada_passos": None,
        "paciencia_epocas": None,     # early stopping por loss de validação
        "max_passos": None,           # orçamento de passos (ablações com budget fixo)
    }
    padrao.update(config.get("pretreino", {}) or {})
    return padrao


def juntar_por_evento(df, outro, colunas):
    """Acrescenta `colunas` de `outro` em `df` casando por evento_id. Se os
    dois já estão na mesma ordem de evento_id (o caso normal: splits e
    discretizado são gerados na ordem da fonte), atribui as colunas direto —
    sem a cópia inteira do DataFrame que um merge faz (memória em 30M+
    linhas). Senão, cai num merge um-pra-um."""
    import numpy as np

    a, b = df["evento_id"].to_numpy(), outro["evento_id"].to_numpy()
    if len(a) == len(b) and np.array_equal(a, b):
        for c in colunas:
            df[c] = outro[c].to_numpy() if not hasattr(outro[c], "cat") else outro[c].values
        return df
    return df.merge(outro[["evento_id"] + colunas], on="evento_id", validate="one_to_one")


def reordenar(df, chaves):
    """sort_values estável por `chaves`, mas reordenando coluna a coluna
    (pico de memória = df + 1 coluna, em vez de df + cópia inteira)."""
    import numpy as np

    perm = np.lexsort([df[c].to_numpy() for c in reversed(chaves)])
    for c in df.columns:
        df[c] = df[c].values.take(perm) if hasattr(df[c].values, "take") else df[c].to_numpy()[perm]
    df.index = np.arange(len(df))
    return df


def mem(etapa):
    """Com DEBUG_MEM=1, imprime a memória residente do processo (diagnóstico
    de pico em bases grandes; ver scripts/rodar_limitado.sh)."""
    import os

    if os.environ.get("DEBUG_MEM"):
        rss = int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
        print(f"  [mem] {etapa}: {rss:.2f} GB", flush=True)
