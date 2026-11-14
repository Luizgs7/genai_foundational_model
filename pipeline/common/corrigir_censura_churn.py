"""Correção, nos artefatos já gerados, da censura unilateral do rótulo de churn.

A receita de churn (tarefa8_rotulos._rotular_churn) foi corrigida em 2026-11-14:
só é determinável a âncora cuja janela de N dias fechou no fim do dataset. Este
script aplica a mesma regra aos artefatos existentes, sem refazer o pipeline:
`churn_rotulo` vira nulo onde (fim do dataset − data da âncora) < N dias.

É exatamente equivalente a regerar os rótulos com a receita corrigida: nas
âncoras de janela fechada nada muda; nas de janela aberta o rótulo anterior só
podia ser 0 (havia cesta seguinte) e passa a ser nulo.

Arquivos corrigidos em runs/<nome>/ (uma cópia do original fica ao lado, com o
sufixo .antes_correcao_churn):
  rotulos_downstream/rotulos.parquet (e o bloco de churn do relatorio.json)
  features_tradicionais/features.parquet
  f2h11/amostra_features.parquet, se existir

Uso: python3 pipeline/common/corrigir_censura_churn.py configs/hm_clientes.yaml
"""

import json
import os
import shutil
import sys

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, "pipeline/common")
from avaliacao import metrica  # noqa: E402
from schema import carregar_config, run_dir, tarefa_por_tipo  # noqa: E402


def main():
    config = carregar_config(sys.argv[1])
    n_dias = tarefa_por_tipo(config, "churn")["dias_escolhido"]
    base = run_dir(config, "")
    datas = pl.read_parquet(config["fonte_canonica"], columns=["evento_id", "data_evento"])
    fim = datas["data_evento"].max()
    datas = datas.with_columns((pl.lit(fim) - pl.col("data_evento")).dt.total_days().alias("_dias_ate_fim")).select(["evento_id", "_dias_ate_fim"])
    print(f"fim do dataset: {fim}; janela de churn: {n_dias} dias")

    abertas = pa.array(datas.filter(pl.col("_dias_ate_fim") < n_dias)["evento_id"].to_numpy())
    del datas

    def resumo(arq):
        return (pl.scan_parquet(arq).select(["split", "churn_rotulo"]).group_by("split")
                .agg(pl.col("churn_rotulo").is_not_null().sum().alias("n"), pl.col("churn_rotulo").mean().alias("taxa"))
                .collect().with_columns(pl.col("split").cast(pl.Utf8)).sort("split"))

    def corrigir(rel):
        """reescreve o arquivo bloco a bloco (mesma ordem de linhas e mesmo esquema), anulando o rótulo de janela aberta."""
        arq = os.path.join(base, rel)
        if not os.path.exists(arq):
            print(f"  {rel}: não existe, pulando")
            return False
        copia = arq + ".antes_correcao_churn"
        if os.path.exists(copia):
            print(f"  {rel}: já corrigido (cópia do original presente), pulando")
            return True
        pf = pq.ParquetFile(arq)
        tmp, w, n_anuladas = arq + ".tmp", None, 0
        for g in range(pf.num_row_groups):
            t = pf.read_row_group(g)
            col = t["churn_rotulo"]
            aberta = pc.is_in(t["evento_id"], value_set=abertas)
            # nas âncoras de janela aberta o rótulo antigo só pode ser 0 ou nulo
            antigos = pc.filter(col, aberta)
            assert pc.max(antigos).as_py() in (0.0, None), "rótulo 1 com janela aberta: receita diferente da esperada"
            n_anuladas += len(antigos) - antigos.null_count
            t = t.set_column(t.schema.get_field_index("churn_rotulo"), t.schema.field("churn_rotulo"),
                             pc.if_else(aberta, pa.scalar(None, type=col.type), col))
            if w is None:
                w = pq.ParquetWriter(tmp, t.schema, compression="zstd")
            w.write_table(t)
        w.close()
        antes, depois = resumo(arq), resumo(tmp)
        print(f"  {rel}: {n_anuladas:,} rótulos anulados")
        print(antes.join(depois, on="split", suffix="_depois"))
        shutil.copy2(arq, copia)
        os.replace(tmp, arq)
        return True

    ok = corrigir("rotulos_downstream/rotulos.parquet")
    corrigir("features_tradicionais/features.parquet")
    corrigir("f2h11/amostra_features.parquet")

    # bloco de churn do relatório de rótulos, recalculado
    if ok:
        rel_arq = os.path.join(base, "rotulos_downstream", "relatorio.json")
        rel = json.load(open(rel_arq, encoding="utf-8"))
        rot = pl.read_parquet(os.path.join(base, "rotulos_downstream", "rotulos.parquet"),
                              columns=["evento_id", "split", "churn_rotulo", "churn_baseline"])
        det = rot.filter(pl.col("churn_rotulo").is_not_null()).with_columns(pl.col("split").cast(pl.Utf8))
        bloco = rel["churn"]
        if "antes_da_correcao_de_2026-11-14" not in bloco:
            bloco["antes_da_correcao_de_2026-11-14"] = {k: bloco[k] for k in ("n_linhas_determinaveis", "taxa_determinabilidade",
                                                                               "base_rate_geral", "base_rate_por_split", "baseline_auc_por_split")
                                                        if k in bloco}
        bloco["n_linhas_determinaveis"] = det.height
        bloco["taxa_determinabilidade"] = det.height / bloco["n_linhas_total"]
        bloco["base_rate_geral"] = float(det["churn_rotulo"].mean())
        bloco["base_rate_por_split"] = {s: float(det.filter(pl.col("split") == s)["churn_rotulo"].mean()) for s in ("train", "val", "test")}

        def auc(d):
            # o baseline é o intervalo médio histórico: intervalo maior = mais risco
            y = d["churn_rotulo"].to_numpy().astype(float)
            s = d["churn_baseline"].to_numpy().astype(float)
            return float(metrica("binaria", y, s, np.ones(len(y))))
        bloco["baseline_auc_por_split"] = {**{s: auc(det.filter(pl.col("split") == s)) for s in ("train", "val", "test")}, "geral": auc(det)}
        bloco["regra"] = "determinável só com a janela de N dias fechada no fim do dataset (corrigido em 2026-11-14)"
        json.dump(rel, open(rel_arq, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print("relatório de rótulos atualizado:", {k: bloco[k] for k in ("n_linhas_determinaveis", "base_rate_por_split", "baseline_auc_por_split")})


if __name__ == "__main__":
    main()
