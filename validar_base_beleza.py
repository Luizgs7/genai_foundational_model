"""Confere se a estrutura plantada pelo gerador da base sintética de beleza B2B
aparece de fato nos dados gerados. Usa as tabelas e o gabarito; não treina nada.

Para cada efeito plantado, compara o grupo em que o efeito deveria aparecer com
um grupo de referência. Se algum efeito não aparecer, o gabarito não vale e as
hipóteses F3-H7 e F3-H8 não podem ser avaliadas nessa base.

Uso: python3 validar_base_beleza.py dados_beleza_b2b/mini
Gera <pasta>/gabarito/validacao.json
"""

import json
import os
import sys

import numpy as np
import polars as pl

CICLO = 21
DIA0 = np.datetime64("2019-01-01")


def main():
    pasta = sys.argv[1]
    ler = lambda n: pl.read_parquet(os.path.join(pasta, n + ".parquet"))  # noqa: E731
    ped, itens, nav = ler("pedidos"), ler("itens_pedido"), ler("navegacao")
    cred, inc, cli, prod = ler("credito"), ler("incentivo"), ler("clientes"), ler("produtos")
    lat, regras = ler("gabarito/clientes_latentes"), ler("gabarito/regras_compra_conjunta")
    g_ped, g_it = ler("gabarito/pedidos_origem"), ler("gabarito/itens_origem")
    res = {}

    # 1. formas, por tipo de cliente
    pc = ped.group_by("cliente_id").agg(pl.len().alias("n_pedidos")).join(cli.select(["cliente_id", "tipo_cliente"]), on="cliente_id")
    res["formas"] = {
        t: {"clientes": int(d.height), "pedidos_por_cliente_p50": float(d["n_pedidos"].median()), "pedidos_por_cliente_max": int(d["n_pedidos"].max())}
        for (t,), d in pc.group_by("tipo_cliente")}
    lp = ped.join(cli.select(["cliente_id", "tipo_cliente"]), on="cliente_id").group_by("tipo_cliente").agg(
        pl.col("n_linhas").median().alias("p50"), pl.col("n_linhas").quantile(0.99).alias("p99"), pl.col("n_linhas").max().alias("max"),
        pl.col("valor_total").median().alias("valor_p50"))
    for r in lp.iter_rows(named=True):
        res["formas"][r["tipo_cliente"]].update({"linhas_por_pedido_p50": r["p50"], "linhas_por_pedido_p99": r["p99"],
                                                 "linhas_por_pedido_max": r["max"], "valor_do_pedido_p50": round(r["valor_p50"], 2)})

    # 2. segmento latente -> categoria dominante das compras
    mix = (itens.join(prod.select(["sku", "categoria"]), on="sku").join(lat.select(["cliente_id", "segmento"]), on="cliente_id")
           .group_by(["segmento", "categoria"]).agg(pl.len().alias("n"))
           .with_columns((pl.col("n") / pl.col("n").sum().over("segmento")).alias("parcela"))
           .sort(["segmento", "parcela"], descending=[False, True]))
    res["segmento_categoria_dominante"] = {s: {"categoria": d["categoria"][0], "parcela": round(float(d["parcela"][0]), 3)}
                                           for (s,), d in mix.group_by("segmento", maintain_order=True)}

    # 3. sensibilidade a desconto -> parcela de linhas em promoção
    pr = itens.group_by("cliente_id").agg(pl.col("em_promocao").mean().alias("parcela_promo"), pl.len().alias("n")).filter(pl.col("n") >= 20)
    pr = pr.join(lat.select(["cliente_id", "sens_desconto"]), on="cliente_id")
    res["desconto"] = {"correlacao_sensibilidade_x_parcela_promo": round(float(np.corrcoef(pr["sens_desconto"], pr["parcela_promo"])[0, 1]), 3),
                       "parcela_promo_tercil_baixo": round(float(pr.filter(pl.col("sens_desconto") < 0.35)["parcela_promo"].mean()), 3),
                       "parcela_promo_tercil_alto": round(float(pr.filter(pl.col("sens_desconto") > 0.65)["parcela_promo"].mean()), 3)}

    # 4. compra conjunta: P(B no pedido | A no pedido) contra P(B no pedido)
    n_ped = ped.height
    freq = itens.group_by("sku").agg(pl.col("pedido_id").n_unique().alias("n_b"))
    ab = (itens.select(["pedido_id", pl.col("sku").alias("sku_a")]).join(regras, on="sku_a")
          .join(itens.select(["pedido_id", pl.col("sku").alias("sku_b"), pl.lit(1).alias("tem_b")]), on=["pedido_id", "sku_b"], how="left"))
    p_cond = float(ab["tem_b"].fill_null(0).mean())
    p_base = float(ab.join(freq, left_on="sku_b", right_on="sku", how="left")["n_b"].fill_null(0).mean() / n_ped)
    res["compra_conjunta"] = {"p_B_dado_A": round(p_cond, 4), "p_B_base": round(p_base, 5), "lift": round(p_cond / max(p_base, 1e-9), 1)}

    # 5. navegação -> compra (junções por proximidade no tempo, lineares)
    vistas = (nav.filter(pl.col("tipo") == "ver_produto").select(["cliente_id", "sku", pl.col("datahora").alias("t_vista")])
              .drop_nulls("sku").sort("t_vista"))
    lin = itens.select(["cliente_id", "sku", pl.col("data").alias("t_pedido")]).sort("t_pedido")
    # (a) a linha do pedido foi vista pelo mesmo cliente nas 72 h anteriores?
    a_ = lin.join_asof(vistas, left_on="t_pedido", right_on="t_vista", by=["cliente_id", "sku"], strategy="backward")
    vista_antes = ((a_["t_pedido"] - a_["t_vista"]).dt.total_hours() <= 72).fill_null(False)
    # (b) a vista virou linha de pedido do mesmo cliente nos 30 dias seguintes?
    b_ = vistas.join_asof(lin, left_on="t_vista", right_on="t_pedido", by=["cliente_id", "sku"], strategy="forward")
    comprou = ((b_["t_pedido"] - b_["t_vista"]).dt.total_days() <= 30).fill_null(False)
    # referência: chance de um SKU qualquer entrar nos pedidos do cliente numa janela de 30 dias
    skus_30d = lin.with_columns(pl.col("t_pedido").dt.truncate("30d").alias("j")).group_by(["cliente_id", "j"]).agg(pl.col("sku").n_unique().alias("k"))
    res["navegacao"] = {"linhas_de_pedido_vistas_antes": round(float(vista_antes.mean()), 3),
                        "vistas_que_viram_compra_em_30_dias": round(float(comprou.mean()), 3),
                        "referencia_sku_ao_acaso": round(float(skus_30d["k"].mean()) / prod.height, 4)}
    del a_, b_, vistas, lin

    # 6. crédito: saída depois de ciclo bloqueado; tamanho do pedido com pouco crédito
    ult = ped.group_by("cliente_id").agg(pl.col("ciclo").max().alias("ultimo_ciclo"))
    cr = cred.with_columns(((pl.col("data").cast(pl.Datetime("ms")) - pl.lit(DIA0.astype("datetime64[ms]").item())).dt.total_days() // CICLO).cast(pl.Int16).alias("ciclo"))
    cr = cr.join(lat.select(["cliente_id", "ciclo_saida"]), on="cliente_id")
    cr = cr.with_columns((pl.col("ciclo_saida") == pl.col("ciclo")).alias("saiu_no_ciclo"))
    res["credito"] = {"taxa_saida_ciclo_bloqueado": round(float(cr.filter(pl.col("bloqueado"))["saiu_no_ciclo"].mean()), 4),
                      "taxa_saida_ciclo_normal": round(float(cr.filter(~pl.col("bloqueado"))["saiu_no_ciclo"].mean()), 4),
                      "ciclos_bloqueados": int(cr["bloqueado"].sum())}
    pl_cr = ped.join(cr.select(["cliente_id", "ciclo", "disponivel", "limite"]), on=["cliente_id", "ciclo"], how="inner")
    med = ped.group_by("cliente_id").agg(pl.col("n_linhas").mean().alias("media_linhas"))
    pl_cr = pl_cr.join(med, on="cliente_id").with_columns((pl.col("n_linhas") / pl.col("media_linhas")).alias("rel"),
                                                          (pl.col("disponivel") / pl.col("limite")).alias("folga"))
    res["credito"]["tamanho_relativo_do_pedido_com_folga_menor_que_40pct"] = round(float(pl_cr.filter(pl.col("folga") < 0.4)["rel"].mean()), 3)
    res["credito"]["tamanho_relativo_do_pedido_com_folga_maior_que_90pct"] = round(float(pl_cr.filter(pl.col("folga") > 0.9)["rel"].mean()), 3)
    del ult

    # 7. incentivo: pedido nos últimos 4 dias do ciclo, para quem estava perto da meta
    dia_no_ciclo = ((pl.col("data").cast(pl.Datetime("ms")) - pl.lit(DIA0.astype("datetime64[ms]").item())).dt.total_days() % CICLO)
    pd_ = ped.with_columns(dia_no_ciclo.alias("dia_ciclo")).join(g_ped, on="pedido_id")
    ini = inc.filter(pl.col("evento") == "inicio_ciclo").select(["cliente_id", "ciclo", "meta_ciclo"])
    antes = (pd_.filter(pl.col("dia_ciclo") < CICLO - 4).group_by(["cliente_id", "ciclo"]).agg(pl.col("valor_total").sum().alias("pontos_antes")))
    fim = (pd_.filter(pl.col("dia_ciclo") >= CICLO - 4).group_by(["cliente_id", "ciclo"]).agg(pl.len().alias("n_fim")))
    b = ini.join(antes, on=["cliente_id", "ciclo"], how="left").join(fim, on=["cliente_id", "ciclo"], how="left").with_columns(
        pl.col("pontos_antes").fill_null(0), (pl.col("n_fim").fill_null(0) > 0).alias("pediu_no_fim"))
    b = b.with_columns((pl.col("pontos_antes") / pl.col("meta_ciclo")).alias("razao"))
    res["incentivo"] = {"pedidos_de_complemento": int(g_ped["pedido_de_complemento"].sum()),
                        "pediu_nos_ultimos_4_dias_com_75_a_100pct_da_meta": round(float(b.filter(pl.col("razao").is_between(0.75, 1.0, closed="left"))["pediu_no_fim"].mean()), 3),
                        "pediu_nos_ultimos_4_dias_com_menos_de_50pct": round(float(b.filter(pl.col("razao") < 0.5)["pediu_no_fim"].mean()), 3),
                        "pediu_nos_ultimos_4_dias_com_meta_ja_batida": round(float(b.filter(pl.col("razao") >= 1.0)["pediu_no_fim"].mean()), 3)}

    # 8. reposição: origem das linhas
    orig = g_it.group_by("origem").agg(pl.len().alias("n")).with_columns((pl.col("n") / pl.col("n").sum()).alias("parcela"))
    res["origem_das_linhas"] = {r["origem"]: round(r["parcela"], 3) for r in orig.iter_rows(named=True)}

    json.dump(res, open(os.path.join(pasta, "gabarito", "validacao.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
