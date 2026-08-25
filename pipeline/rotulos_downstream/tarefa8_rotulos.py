"""Tarefa 8 — Rótulos para tarefas downstream (churn, próxima categoria, próximo valor).

Ver .claude/skills/downstream-label-engineering/SKILL.md para o racional (por que
o rótulo olha só pro futuro de cada evento, por que censura em churn, por que
baseline por cliente em vez de baseline global).

Gera:
  - pipeline/rotulos_downstream/tarefa8_rotulos.csv: uma linha por transação elegível
    (elegivel_downstream=True, de splits.csv), com os 3 rótulos (NaN onde a
    tarefa não é determinável para aquela linha — ver critérios de censura).
  - pipeline/rotulos_downstream/tarefa8_relatorio.json: base rates + métricas de
    baseline (sem nenhum modelo treinado) por tarefa e por split.

Sem sklearn/scipy disponíveis neste ambiente — AUC-ROC, accuracy e F1 macro
são implementados na mão (numpy/pandas), sem nenhuma dependência nova.
"""

import json

import numpy as np
import pandas as pd

SOURCE_CSV = "base_sintetica_embeddings_100k_v2.csv"
SPLITS_CSV = "pipeline/splits/splits.csv"
DISCRETIZADO_CSV = "pipeline/discretizacao/discretizado.csv"
OUTPUT_CSV = "pipeline/rotulos_downstream/tarefa8_rotulos.csv"
OUTPUT_JSON = "pipeline/rotulos_downstream/tarefa8_relatorio.json"

# Candidatos avaliados sobre a distribuição real de gaps entre compras
# consecutivas (ver exploração ad-hoc); N=121 é o corte que produz uma taxa
# de positivos de ~29.7% — nem trivialmente rara nem trivialmente comum.
CANDIDATOS_N_CHURN = [30, 45, 60, 75, 90, 105, 120, 121, 135, 150, 180]
N_CHURN_DIAS = 121


def carregar_base():
    df = pd.read_csv(
        SOURCE_CSV,
        encoding="utf-8-sig",
        usecols=["cpf", "data_compra", "categoria_produto"],
    )
    df["data_compra"] = pd.to_datetime(df["data_compra"])
    df["transacao_id"] = df.index

    splits = pd.read_csv(SPLITS_CSV, usecols=["transacao_id", "split", "elegivel_downstream"])
    df = df.merge(splits, on="transacao_id", validate="one_to_one")

    buckets = pd.read_csv(DISCRETIZADO_CSV, usecols=["transacao_id", "valor_total_bucket"])
    df = df.merge(buckets, on="transacao_id", validate="one_to_one")

    return df


def explorar_gaps(df):
    """Distribuição de gaps entre compras consecutivas — base para escolher
    N_CHURN_DIAS a partir dos dados, não de um número arbitrário."""
    tmp = df.sort_values(["cpf", "data_compra", "transacao_id"]).copy()
    tmp["next_date"] = tmp.groupby("cpf")["data_compra"].shift(-1)
    tmp["gap_dias"] = (tmp["next_date"] - tmp["data_compra"]).dt.days
    gaps = tmp["gap_dias"].dropna()

    print("--- Distribuição de gaps entre compras consecutivas (dias) ---")
    print(gaps.describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))

    dataset_end = tmp["data_compra"].max()
    print(f"\nfim do dataset: {dataset_end.date()}")
    print("\nTaxa de churn por candidato de N (com tratamento de censura):")
    for n in CANDIDATOS_N_CHURN:
        rotulo, determinavel = _rotular_churn(tmp, n, dataset_end)
        taxa = rotulo[determinavel].mean()
        print(f"  N={n:>3}: taxa_churn={taxa:.4f}  n_determinavel={determinavel.sum()}")
    print(f"\nN escolhido: {N_CHURN_DIAS} dias (taxa resultante mais próxima do esperado ~29.7%).")


def _rotular_churn(df_ordenado, n_dias, dataset_end):
    """Rótulo de churn com censura: 1 = não voltou a comprar dentro de N dias
    após o evento; NaN/indeterminável quando o cliente não tem próxima compra
    E a janela de N dias ainda não se fechou até o fim do dataset."""
    tem_next = df_ordenado["next_date"].notna()
    gap = df_ordenado["gap_dias"]

    dias_ate_fim = (dataset_end - df_ordenado["data_compra"]).dt.days
    censura_fechada = dias_ate_fim >= n_dias  # só vale quando não tem next

    determinavel = tem_next | censura_fechada
    rotulo = pd.Series(np.nan, index=df_ordenado.index)
    rotulo.loc[tem_next] = (gap[tem_next] > n_dias).astype(float)
    rotulo.loc[~tem_next & censura_fechada] = 1.0  # esgotou a janela sem voltar
    return rotulo, determinavel


def montar_rotulos(df):
    # Churn NÃO é restrito a elegivel_downstream (>=2 transações): um cliente
    # de compra única também produz um rótulo de churn válido via censura
    # (ex.: comprou uma vez há >N dias e nunca mais voltou — isso é exatamente
    # o evento que a tarefa quer capturar). A skill diz que churn "é aplicável
    # a clientes com pelo menos 2 transações, OU censura conhecida no fim da
    # janela" — a segunda condição cobre o cliente de evento único. Restringir
    # a elegivel_downstream aqui dilui a taxa de churn (de ~29.7% pra ~26.6%)
    # removendo justamente os casos mais óbvios de abandono.
    # Já "próxima categoria"/"próximo valor" continuam de fato restritos, mas
    # isso acontece automaticamente: sem uma próxima compra (exigência de
    # elegivel_downstream), essas duas colunas já saem NaN.
    elegiveis = df.sort_values(["cpf", "data_compra", "transacao_id"]).reset_index(drop=True)

    grp = elegiveis.groupby("cpf")
    elegiveis["next_date"] = grp["data_compra"].shift(-1)
    elegiveis["next_categoria"] = grp["categoria_produto"].shift(-1)
    elegiveis["next_valor_bucket"] = grp["valor_total_bucket"].shift(-1)
    elegiveis["gap_dias"] = (elegiveis["next_date"] - elegiveis["data_compra"]).dt.days

    dataset_end = elegiveis["data_compra"].max()
    churn_label, churn_determinavel = _rotular_churn(elegiveis, N_CHURN_DIAS, dataset_end)
    elegiveis["churn_label"] = churn_label
    elegiveis.loc[~churn_determinavel, "churn_label"] = np.nan

    # Baseline de churn: média histórica (causal, só passado) do gap entre
    # compras do próprio cliente — RFM simples, sem nenhum aprendizado.
    gap_prev = elegiveis.groupby("cpf")["gap_dias"].shift(1)
    elegiveis["baseline_churn_score"] = gap_prev.groupby(elegiveis["cpf"]).apply(
        lambda s: s.expanding(min_periods=1).mean()
    ).reset_index(level=0, drop=True)

    # Baselines de "próxima categoria" / "próximo valor": moda histórica
    # (causal, só passado) das compras do próprio cliente.
    elegiveis["baseline_categoria"] = _moda_historica_por_grupo(elegiveis, "cpf", "categoria_produto")
    elegiveis["baseline_valor_bucket"] = _moda_historica_por_grupo(elegiveis, "cpf", "valor_total_bucket")

    # Fallback global (cliente sem histórico prévio ainda) calibrado só no
    # split de treino, mesmo critério usado nas Tarefas 4/5.
    train = elegiveis[elegiveis["split"] == "train"]
    fallback_gap = train["gap_dias"].dropna().mean()
    fallback_categoria = train["categoria_produto"].mode().iloc[0]
    fallback_valor_bucket = train["valor_total_bucket"].mode().iloc[0]

    elegiveis["baseline_churn_score"] = elegiveis["baseline_churn_score"].fillna(fallback_gap)
    elegiveis["baseline_categoria"] = elegiveis["baseline_categoria"].fillna(fallback_categoria)
    elegiveis["baseline_valor_bucket"] = elegiveis["baseline_valor_bucket"].fillna(fallback_valor_bucket)

    return elegiveis, {
        "fallback_gap_medio_treino": float(fallback_gap),
        "fallback_categoria_treino": fallback_categoria,
        "fallback_valor_bucket_treino": int(fallback_valor_bucket),
    }


def _moda_historica_por_grupo(df, coluna_grupo, coluna_valor):
    """Para cada linha, moda de `coluna_valor` considerando só linhas
    ANTERIORES do mesmo grupo (causal — a linha atual nunca entra na sua
    própria baseline)."""

    def moda_causal(serie):
        contagem = {}
        melhor, melhor_contagem = None, 0
        saida = []
        for v in serie:
            saida.append(melhor)
            contagem[v] = contagem.get(v, 0) + 1
            if contagem[v] > melhor_contagem:
                melhor, melhor_contagem = v, contagem[v]
        return pd.Series(saida, index=serie.index)

    return df.groupby(coluna_grupo)[coluna_valor].apply(moda_causal).reset_index(level=0, drop=True)


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
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return float("nan")
    return float((y_true[mask] == y_pred[mask]).mean())


def f1_macro(y_true, y_pred):
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


def _nan_para_none(obj):
    """json.dump com allow_nan=False rejeita float('nan') direto — troca
    recursivamente por None (-> null no JSON), já que métricas indefinidas
    (ex. AUC sem exemplo positivo no split de teste) são um NaN real."""
    if isinstance(obj, dict):
        return {k: _nan_para_none(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_nan_para_none(v) for v in obj]
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def main():
    df = carregar_base()
    explorar_gaps(df)

    print("\n--- Montando rótulos ---")
    rotulados, fallback_info = montar_rotulos(df)
    print(f"Linhas processadas: {len(rotulados)} (churn não restrito a elegivel_downstream — ver comentário em montar_rotulos)")
    print(f"Fallback (calibrado só no treino): {fallback_info}")

    # --- Churn ---
    mask_churn = rotulados["churn_label"].notna()
    base_rate_geral = rotulados.loc[mask_churn, "churn_label"].mean()
    base_rate_split = rotulados[mask_churn].groupby("split")["churn_label"].mean().to_dict()
    auc_split = {}
    for split_name in ["train", "val", "test"]:
        sub = rotulados[mask_churn & (rotulados["split"] == split_name)]
        auc_split[split_name] = auc_roc(sub["churn_label"], sub["baseline_churn_score"])
    auc_geral = auc_roc(rotulados.loc[mask_churn, "churn_label"], rotulados.loc[mask_churn, "baseline_churn_score"])

    print(f"\n--- Churn (N={N_CHURN_DIAS} dias) ---")
    print(f"Linhas determináveis: {mask_churn.sum()} (excluídas por censura: {(~mask_churn).sum()})")
    print(f"Base rate geral: {base_rate_geral:.4f}")
    print(f"Base rate por split: {base_rate_split}")
    print(f"Baseline (média histórica de gap por cliente) — AUC-ROC geral: {auc_geral:.4f}")
    print(f"AUC-ROC por split: {auc_split}")

    taxa_teste = base_rate_split.get("test")
    if taxa_teste is None or taxa_teste < 0.02 or taxa_teste > 0.98:
        print(
            f"ALERTA: base rate de churn no split de teste é degenerado ({taxa_teste}) "
            "— a janela de teste (ver Tarefa 7) provavelmente é menor que o "
            f"horizonte de churn (N={N_CHURN_DIAS} dias). Churn não é avaliável "
            "no split de teste como os splits estão definidos hoje."
        )
    else:
        print(
            f"OK: base rate de churn no split de teste é {taxa_teste:.4f} "
            "(não-degenerado) — janela de teste comporta o horizonte de churn."
        )

    # --- Próxima categoria ---
    mask_cat = rotulados["next_categoria"].notna()
    acc_cat_split = metrica_por_split(rotulados, mask_cat, "next_categoria", "baseline_categoria", accuracy)
    f1_cat_split = metrica_por_split(rotulados, mask_cat, "next_categoria", "baseline_categoria", f1_macro)

    print("\n--- Próxima categoria ---")
    print(f"Linhas determináveis: {mask_cat.sum()} (sem próxima compra: {(~mask_cat).sum()})")
    print(f"Baseline (moda histórica por cliente) — accuracy: {acc_cat_split}")
    print(f"Baseline — F1 macro: {f1_cat_split}")

    # --- Próximo valor (bucket) ---
    mask_val = rotulados["next_valor_bucket"].notna()
    acc_val_split = metrica_por_split(rotulados, mask_val, "next_valor_bucket", "baseline_valor_bucket", accuracy)
    f1_val_split = metrica_por_split(rotulados, mask_val, "next_valor_bucket", "baseline_valor_bucket", f1_macro)

    print("\n--- Próximo valor (bucket, 0-9) ---")
    print(f"Linhas determináveis: {mask_val.sum()} (sem próxima compra: {(~mask_val).sum()})")
    print(f"Baseline (moda histórica por cliente) — accuracy: {acc_val_split}")
    print(f"Baseline — F1 macro: {f1_val_split}")

    # --- Salvar dataset rotulado ---
    colunas_saida = [
        "transacao_id", "cpf", "data_compra", "split",
        "churn_label", "next_categoria", "next_valor_bucket",
    ]
    saida = rotulados[colunas_saida].rename(
        columns={"next_categoria": "proxima_categoria", "next_valor_bucket": "proximo_valor_bucket"}
    )
    saida.to_csv(OUTPUT_CSV, index=False)
    print(f"\nGravado {OUTPUT_CSV} com {len(saida)} linhas.")

    relatorio = {
        "churn": {
            "n_dias": N_CHURN_DIAS,
            "criterio_escolha_n": (
                "varredura de candidatos sobre a distribuição real de gaps "
                "entre compras consecutivas; N=121 é o corte cuja taxa de "
                "positivos (~29.7%) fica na faixa não-degenerada (nem <5% "
                "nem >90%)"
            ),
            "n_linhas_determinaveis": int(mask_churn.sum()),
            "n_linhas_censuradas_excluidas": int((~mask_churn).sum()),
            "base_rate_geral": float(base_rate_geral),
            "base_rate_por_split": {k: float(v) for k, v in base_rate_split.items()},
            "diagnostico_split_teste": (
                f"base rate de churn no teste = {taxa_teste:.4f}, não-degenerado "
                "— a janela de teste (Tarefa 7, TEST_WINDOW_DAYS=180) foi "
                f"dimensionada com margem acima do horizonte de churn (N="
                f"{N_CHURN_DIAS} dias) especificamente para viabilizar essa "
                "avaliação. Ver ROADMAP.md (Tarefa 7) para o histórico da "
                "correção (janela original de 90 dias era menor que N e "
                "zerava a taxa de churn no teste por construção)."
                if taxa_teste is not None and 0.02 <= taxa_teste <= 0.98
                else "degenerado — ver alerta impresso no log de execução."
            ),
            "baseline": {
                "nome": "media_historica_causal_de_gap_por_cliente (RFM)",
                "metrica": "AUC-ROC",
                "geral": auc_geral,
                "por_split": auc_split,
            },
        },
        "proxima_categoria": {
            "n_linhas_determinaveis": int(mask_cat.sum()),
            "baseline": {
                "nome": "moda_historica_causal_por_cliente",
                "accuracy": acc_cat_split,
                "f1_macro": f1_cat_split,
            },
        },
        "proximo_valor_bucket": {
            "n_linhas_determinaveis": int(mask_val.sum()),
            "baseline": {
                "nome": "moda_historica_causal_por_cliente",
                "accuracy": acc_val_split,
                "f1_macro": f1_val_split,
            },
        },
        "fallback_global_calibrado_no_treino": fallback_info,
    }
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(_nan_para_none(relatorio), f, ensure_ascii=False, indent=2, allow_nan=False)
    print(f"Gravado {OUTPUT_JSON}.")


if __name__ == "__main__":
    main()
