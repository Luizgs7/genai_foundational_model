"""Motor genérico — Catálogo de rótulos downstream (Tarefa 8 generalizada).

Ver PLANO_PRODUTO.md pra desenho completo e critério de qualidade de cada
receita. Duas receitas genéricas cobrem quase todo o catálogo:

  - `campo_futuro`: shift(-1) por entidade sobre um campo qualquer, com
    limiar/operador opcional (vira booleano) — cobre `next_category`,
    `next_value`, `risco_review_negativo`, `risco_atraso_entrega`.
  - `ltv`: soma causal de um campo numérico numa janela futura de H dias,
    com a mesma disciplina de censura do churn.

`churn` continua com lógica própria (candidatos de N variados, censura por
"não tem próximo E janela não fechou") — é o único caso booleano onde o
evento em si (não um campo específico) é o alvo.

Uso: python3 pipeline/common/tarefa8_rotulos.py <config.yaml>

Gera runs/<nome>/rotulos_downstream/{rotulos.parquet, relatorio.json}.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "pipeline/common")
from schema import caminho_tabela, carregar_config, gravar_tabela, ler_fonte, ler_tabela, run_dir, tarefas_ativas  # noqa: E402


# ---------------------------------------------------------------- métricas

def auc_roc(y_true, score):
    y_true = pd.Series(y_true).reset_index(drop=True)
    score = pd.Series(score).reset_index(drop=True)
    mask = y_true.notna() & score.notna()
    y_true, score = y_true[mask], score[mask]
    n_pos, n_neg = (y_true == 1).sum(), (y_true == 0).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = score.rank(method="average")
    soma_ranks_pos = ranks[y_true == 1].sum()
    return float((soma_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def accuracy(y_true, y_pred):
    y_true, y_pred = pd.Series(y_true).reset_index(drop=True), pd.Series(y_pred).reset_index(drop=True)
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return float("nan")
    return float((y_true[mask] == y_pred[mask]).mean())


def f1_macro(y_true, y_pred):
    y_true, y_pred = pd.Series(y_true).reset_index(drop=True), pd.Series(y_pred).reset_index(drop=True)
    mask = y_true.notna() & y_pred.notna()
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return float("nan")
    classes = pd.unique(pd.concat([y_true, y_pred]))
    f1s = []
    for c in classes:
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        precisao = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precisao * recall / (precisao + recall) if (precisao + recall) > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def metrica_por_split(df, mascara_valida, y_true_col, y_pred_col, metrica_fn, **kwargs):
    saida = {}
    for split_name in ["train", "val", "test"]:
        sub = df[mascara_valida & (df["split"] == split_name)]
        saida[split_name] = metrica_fn(sub[y_true_col], sub[y_pred_col], **kwargs) if len(sub) else float("nan")
    todos = df[mascara_valida]
    saida["geral"] = metrica_fn(todos[y_true_col], todos[y_pred_col], **kwargs) if len(todos) else float("nan")
    return saida


def moda_causal_por_grupo(df, grupo_col, valor_col):
    """Moda de `valor_col` considerando só linhas ANTERIORES do mesmo grupo
    (causal). NaN em `valor_col` não contamina a contagem, mas a linha ainda
    recebe o melhor palpite corrente.

    Desempate: vence o valor que PRIMEIRO atingiu a contagem máxima corrente
    (a versão original, com laço Python, só trocava a moda quando um valor
    passava estritamente da contagem máxima). Vetorizado: a moda muda só nas
    linhas "recorde" (contagem acumulada do próprio valor > máximo anterior
    do grupo); a moda vigente é o valor do último recorde, deslocada 1
    linha pra ser estritamente causal.
    """
    grupo = df[grupo_col]
    valor = df[valor_col]
    valido = valor.notna()

    contagem_valor = pd.Series(np.nan, index=df.index)
    contagem_valor[valido] = df[valido].groupby([grupo_col, valor_col], sort=False, observed=True).cumcount() + 1
    maximo = contagem_valor.fillna(0).groupby(grupo, sort=False).cummax()
    maximo_anterior = maximo.groupby(grupo, sort=False).shift(1).fillna(0)
    recorde = valido & (contagem_valor > maximo_anterior)

    moda_apos = valor.where(recorde).groupby(grupo, sort=False).ffill()
    saida = moda_apos.groupby(grupo, sort=False).shift(1)
    if pd.api.types.is_numeric_dtype(valor):
        return saida.astype(float)  # mesmo dtype que o legado produzia (float com NaN)
    return saida.astype(object).where(saida.notna(), None)


def media_expandida_causal(df, grupo_col, valor_col):
    """Média expandida (só passado) de `valor_col` por grupo — mesma lógica
    do RFM (`baseline_churn_score`), generalizada pra qualquer campo
    numérico/booleano usado como taxa histórica. Vetorizado via somas e
    contagens acumuladas (antes: groupby.apply(expanding().mean()))."""
    grupo = df[grupo_col]
    anterior = df.groupby(grupo_col, sort=False)[valor_col].shift(1).astype(float)
    soma = anterior.fillna(0.0).groupby(grupo, sort=False).cumsum()
    n = anterior.notna().astype(np.int64).groupby(grupo, sort=False).cumsum()
    return (soma / n).where(n > 0)


# ---------------------------------------------------------------- receitas

def _rotular_churn(df, n_dias, dataset_end):
    tem_next = df["next_date"].notna()
    gap = df["gap_dias"]
    dias_ate_fim = (dataset_end - df["data_evento"]).dt.days
    censura_fechada = dias_ate_fim >= n_dias
    determinavel = tem_next | censura_fechada
    rotulo = pd.Series(np.nan, index=df.index)
    rotulo.loc[tem_next] = (gap[tem_next] > n_dias).astype(float)
    rotulo.loc[~tem_next & censura_fechada] = 1.0
    return rotulo, determinavel


def receita_churn(df, cfg, dataset_end):
    candidatos = cfg.get("candidatos_dias", [cfg["dias_escolhido"]])
    print(f"  [churn] varredura de candidatos de N (dias): {candidatos}")
    for n in candidatos:
        rotulo, determinavel = _rotular_churn(df, n, dataset_end)
        taxa = rotulo[determinavel].mean()
        print(f"    N={n:>3}: taxa_churn={taxa:.4f}  n_determinavel={int(determinavel.sum())}")
    n_escolhido = cfg["dias_escolhido"]
    rotulo, determinavel = _rotular_churn(df, n_escolhido, dataset_end)
    rotulo = rotulo.where(determinavel)
    baseline = media_expandida_causal(df, "entidade_id", "gap_dias")
    fallback = df.loc[df["split"] == "train", "gap_dias"].dropna().mean()
    baseline = baseline.fillna(fallback)
    return rotulo, baseline, {"n_dias": n_escolhido}


def receita_campo_futuro(df, cfg):
    campo_fonte = cfg["campo_fonte"]
    proximo = df.groupby("entidade_id")[campo_fonte].shift(-1)

    if "limiar" in cfg:
        operador = cfg.get("operador", "<=")
        if operador == "<=":
            rotulo = (proximo <= cfg["limiar"]).astype(float)
            alvo_bruto = (df[campo_fonte] <= cfg["limiar"]).astype(float)
        elif operador == ">":
            rotulo = (proximo > cfg["limiar"]).astype(float)
            alvo_bruto = (df[campo_fonte] > cfg["limiar"]).astype(float)
        else:
            raise ValueError(f"operador desconhecido: {operador}")
        rotulo = rotulo.where(proximo.notna())
        alvo_bruto = alvo_bruto.where(df[campo_fonte].notna())
    else:
        rotulo = proximo
        alvo_bruto = df[campo_fonte]

    if rotulo.dtype == bool:
        rotulo = rotulo.astype(float)

    # baseline: decidido pelo tipo do RÓTULO (já limiarizado, se houver
    # `limiar`) -> taxa histórica causal quando o alvo é booleano (mesma
    # lógica do churn); moda histórica causal quando é categórico/bucket.
    # Importante: usa `alvo_bruto` (mesma transformação do rótulo, mas SEM
    # o shift(-1)) -- media_expandida_causal já desloca 1 posição pra trás
    # por conta própria; aplicar em cima do rótulo (já deslocado pra
    # frente) reintroduziria o próprio evento atual na média causal.
    if eh_binaria(rotulo):
        baseline = media_expandida_causal(df.assign(_alvo_bin=alvo_bruto), "entidade_id", "_alvo_bin")
        fallback = alvo_bruto[df["split"] == "train"].dropna().mean()
        baseline = baseline.fillna(fallback)
    else:
        baseline = moda_causal_por_grupo(df, "entidade_id", campo_fonte)
        fallback = df.loc[df["split"] == "train", campo_fonte].mode()
        baseline = baseline.fillna(fallback.iloc[0] if len(fallback) else np.nan)

    return rotulo, baseline, {}


def receita_ltv(df, cfg, dataset_end):
    """Soma de `campo_valor` dos eventos SEGUINTES da entidade com data em
    (data_i, data_i + H] — mais os eventos posteriores de mesma data —, só
    quando a janela já fechou até o fim do dataset. Vetorizado: soma
    acumulada por entidade + searchsorted numa chave (entidade, segundos)
    ordenada (df já vem ordenado por entidade, data, evento_id)."""
    campo_valor = cfg["campo_valor"]
    horizonte = cfg["horizonte_dias"]

    ent_cod = pd.factorize(df["entidade_id"])[0].astype(np.int64)
    segundos = df["data_evento"].to_numpy().astype("datetime64[s]").astype(np.int64)
    assert (df["data_evento"].to_numpy().astype("datetime64[ns]").astype(np.int64) % 10**9 == 0).all(), \
        "receita_ltv assume datas com resolução de segundos"
    assert np.all(np.diff(ent_cod) >= 0), "df precisa estar ordenado por entidade"
    deslocamento = int(segundos.max() - segundos.min()) + 1
    chave = ent_cod * deslocamento + (segundos - segundos.min())
    limite = chave + horizonte * 86400
    j = np.searchsorted(chave, limite, side="right")      # 1º índice global fora da janela

    valores = df[campo_valor].to_numpy(dtype=float)
    assert not np.isnan(valores).any(), f"receita_ltv: {campo_valor} tem valores nulos"
    cum_incl = pd.Series(valores).groupby(ent_cod).cumsum().to_numpy()
    # soma das linhas i+1 .. j-1 (todas da mesma entidade, pela construção da chave)
    soma_futuro = np.where(j - 1 > np.arange(len(df)), cum_incl[np.maximum(j - 1, 0)] - cum_incl, 0.0)

    dias_ate_fim = (dataset_end - df["data_evento"]).dt.days.to_numpy()
    rotulo = pd.Series(np.where(dias_ate_fim >= horizonte, soma_futuro, np.nan), index=df.index)

    n_buckets = cfg.get("n_buckets", 10)
    train_vals = rotulo[df["split"] == "train"].dropna()
    edges = np.quantile(train_vals.values, np.linspace(0, 100, n_buckets + 1)[1:-1] / 100) if len(train_vals) else []
    rotulo_bucket = pd.Series(np.searchsorted(edges, rotulo.to_numpy(), side="right"), index=df.index).astype(float)
    rotulo_bucket = rotulo_bucket.where(rotulo.notna())

    baseline = moda_causal_por_grupo(df.assign(_ltv_bucket=rotulo_bucket), "entidade_id", "_ltv_bucket")
    fallback = rotulo_bucket[df["split"] == "train"].mode()
    baseline = baseline.fillna(fallback.iloc[0] if len(fallback) else np.nan)

    return rotulo_bucket, baseline, {"horizonte_dias": horizonte, "n_buckets": n_buckets}


# ---------------------------------------------------------------- runner

def eh_binaria(rotulo):
    valores = set(rotulo.dropna().unique())
    return valores <= {0.0, 1.0}


def main(config_path):
    config = carregar_config(config_path)

    colunas_bucket = {f"{c['nome']}_bucket" for c in config["campos_numericos"]}
    colunas_extra = set()
    for t in tarefas_ativas(config):
        if t["tipo"] == "campo_futuro":
            colunas_extra.add(t["campo_fonte"])
        elif t["tipo"] == "ltv":
            colunas_extra.add(t["campo_valor"])
    # campos *_bucket vêm do merge com discretizado.csv, não do canônico
    colunas_extra = list(colunas_extra - colunas_bucket)

    fonte = ler_fonte(config, ["evento_id", "entidade_id", "data_evento"] + colunas_extra)
    fonte["data_evento"] = pd.to_datetime(fonte["data_evento"])

    splits = ler_tabela(caminho_tabela(config, "splits", "splits"), ["evento_id", "split"])
    df = fonte.merge(splits, on="evento_id", validate="one_to_one")

    if config["campos_numericos"]:
        cols_bucket = ["evento_id"] + [f"{c['nome']}_bucket" for c in config["campos_numericos"]]
        disc = ler_tabela(caminho_tabela(config, "discretizacao", "discretizado"), cols_bucket)
        df = df.merge(disc, on="evento_id", validate="one_to_one")

    df = df.sort_values(["entidade_id", "data_evento", "evento_id"]).reset_index(drop=True)
    df["next_date"] = df.groupby("entidade_id")["data_evento"].shift(-1)
    df["gap_dias"] = (df["next_date"] - df["data_evento"]).dt.days
    dataset_end = df["data_evento"].max()

    relatorio = {}
    saida = df[["evento_id", "entidade_id", "split"]].copy()
    # Feature causal genérica (não amarrada a nenhuma tarefa): quantos
    # eventos essa entidade já teve ANTES deste. Consumida pela Tarefa 13
    # como sinal de "maturidade"/histórico disponível na cabeça DCNv2.
    saida["n_eventos_anteriores"] = df.groupby("entidade_id").cumcount()

    for tarefa_cfg in tarefas_ativas(config):
        nome = tarefa_cfg.get("nome", tarefa_cfg["tipo"])
        print(f"\n=== Tarefa downstream: {nome} ===")

        if tarefa_cfg["tipo"] == "churn":
            rotulo, baseline, extra = receita_churn(df, tarefa_cfg, dataset_end)
        elif tarefa_cfg["tipo"] == "campo_futuro":
            rotulo, baseline, extra = receita_campo_futuro(df, tarefa_cfg)
        elif tarefa_cfg["tipo"] == "ltv":
            rotulo, baseline, extra = receita_ltv(df, tarefa_cfg, dataset_end)
        else:
            raise ValueError(f"tipo de tarefa desconhecido: {tarefa_cfg['tipo']}")

        saida[f"{nome}_rotulo"] = rotulo
        saida[f"{nome}_baseline"] = baseline

        determinavel = rotulo.notna()
        n_det = int(determinavel.sum())
        n_total = len(rotulo)
        info = {
            "n_linhas_determinaveis": n_det,
            "n_linhas_total": n_total,
            "taxa_determinabilidade": n_det / n_total if n_total else float("nan"),
            **extra,
        }

        if eh_binaria(rotulo):
            info["base_rate_geral"] = float(rotulo[determinavel].mean()) if n_det else float("nan")
            info["base_rate_por_split"] = {
                s: float(rotulo[determinavel & (df["split"] == s)].mean())
                if (determinavel & (df["split"] == s)).sum() else float("nan")
                for s in ["train", "val", "test"]
            }
            info["baseline_auc_por_split"] = metrica_por_split(
                saida, determinavel, f"{nome}_rotulo", f"{nome}_baseline", auc_roc
            )
            print(f"  base_rate_geral={info['base_rate_geral']:.4f}  "
                  f"base_rate_teste={info['base_rate_por_split']['test']:.4f}  "
                  f"baseline_auc_teste={info['baseline_auc_por_split']['test']:.4f}")
        else:
            pred_col = f"{nome}_baseline"
            info["baseline_accuracy_por_split"] = metrica_por_split(
                saida, determinavel, f"{nome}_rotulo", pred_col, accuracy
            )
            info["baseline_f1_macro_por_split"] = metrica_por_split(
                saida, determinavel, f"{nome}_rotulo", pred_col, f1_macro
            )
            print(f"  n_determinavel={n_det}  "
                  f"baseline_accuracy_teste={info['baseline_accuracy_por_split']['test']:.4f}")

        relatorio[nome] = info

    out_dir = run_dir(config, "rotulos_downstream")
    rotulos_csv = gravar_tabela(saida, config, "rotulos_downstream", "rotulos")

    def nan_para_none(obj):
        if isinstance(obj, dict):
            return {k: nan_para_none(v) for k, v in obj.items()}
        if isinstance(obj, float) and np.isnan(obj):
            return None
        return obj

    relatorio_json = os.path.join(out_dir, "relatorio.json")
    with open(relatorio_json, "w", encoding="utf-8") as f:
        json.dump(nan_para_none(relatorio), f, ensure_ascii=False, indent=2, allow_nan=False)

    print(f"\nGravado {rotulos_csv} e {relatorio_json}")


if __name__ == "__main__":
    main(sys.argv[1])
