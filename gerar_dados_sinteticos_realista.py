"""Gera uma versão mais realista de base_sintetica_embeddings_100k.csv.

Corrige, em relação ao arquivo original:
- Ausência de recorrência de cliente (98,6% dos CPFs apareciam em só 1 transação).
- cpf/email/nome/data_nascimento inconsistentes entre linhas do mesmo cliente.
- valor_compra/valor_total com distribuição uniforme (agora log-normal por categoria).
- desconto decorativo, sem efeito em valor_total (agora aplicado de fato).
- latitude/longitude sem relação com a cidade do registro (agora geradas em torno
  do centro real de cada cidade).
- Ausência de padrões comportamentais por cliente (agora cada cliente tem
  preferências de categoria/marca/forma de pagamento/canal de venda).

Mantém: schema de 25 colunas, mapeamento fixo cod_produto -> descricao_produto ->
categoria_produto, independência entre marca/fabricante e produto, ~100.000 linhas.
"""

import csv
from datetime import date, timedelta

import numpy as np
import pandas as pd
from faker import Faker

SEED = 42
SOURCE_CSV = "base_sintetica_embeddings_100k.csv"
OUTPUT_CSV = "base_sintetica_embeddings_100k_v2.csv"

N_CUSTOMERS = 15_000
N_TRANSACTIONS = 100_000
MAX_TRANSACOES_POR_CLIENTE = 150

DATE_START = date(2023, 7, 29)
DATE_END = date(2026, 7, 28)
TOTAL_DAYS = (DATE_END - DATE_START).days

COLUMNS = [
    "nome_completo", "sexo", "cpf", "email", "telefone", "data_nascimento",
    "data_compra", "cod_produto", "descricao_produto", "categoria_produto",
    "marca", "fabricante", "valor_compra", "quantidade", "desconto",
    "valor_total", "forma_pagamento", "canal_venda", "vendedor", "cidade",
    "uf", "cep", "latitude", "longitude", "descricao_compra",
]

CITY_COORDS = {
    ("Cascavel", "PR"): (-24.9578, -53.4595),
    ("Londrina", "PR"): (-23.3045, -51.1696),
    ("Maringá", "PR"): (-23.4205, -51.9331),
    ("Luís Eduardo Magalhães", "BA"): (-12.0958, -45.7986),
    ("Passo Fundo", "RS"): (-28.2576, -52.4092),
    ("Rio Verde", "GO"): (-17.7979, -50.9188),
    ("Sorriso", "MT"): (-12.5453, -55.7217),
    ("Uberlândia", "MG"): (-18.9186, -48.2772),
}
CITIES = list(CITY_COORDS.keys())

# Peso relativo de compra por mês (1=jan ... 12=dez), calibrado ao calendário
# agrícola do centro-sul do Brasil (plantio ago-out, safra nov-mar).
CATEGORY_SEASONALITY = {
    "Sementes":      [0.5, 0.5, 0.5, 0.5, 1.0, 2.0, 3.0, 3.0, 2.5, 1.0, 0.5, 0.5],
    "Fertilizantes": [0.5, 0.5, 0.5, 1.0, 2.0, 3.0, 3.0, 2.5, 1.5, 0.5, 0.5, 0.5],
    "Defensivos":    [0.5, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.0, 2.0, 1.0, 0.5, 0.5],
    "Máquinas":      [1.0] * 12,
}

# (mu, sigma) de log(valor_compra) por categoria -> log-normal com escala de
# preço diferente por categoria (Máquinas = ticket alto).
CATEGORY_PRICE_PARAMS = {
    "Sementes": (6.8, 0.5),
    "Fertilizantes": (7.0, 0.5),
    "Defensivos": (6.9, 0.55),
    "Máquinas": (9.0, 0.6),
}


def ler_tabelas_referencia(path):
    produtos = {}  # cod_produto -> (descricao_produto, categoria_produto)
    marcas, fabricantes, formas_pagamento, canais = set(), set(), set(), set()
    with open(path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            produtos[row["cod_produto"]] = (row["descricao_produto"], row["categoria_produto"])
            marcas.add(row["marca"])
            fabricantes.add(row["fabricante"])
            formas_pagamento.add(row["forma_pagamento"])
            canais.add(row["canal_venda"])
    por_categoria = {}
    for cod, (desc, cat) in produtos.items():
        por_categoria.setdefault(cat, []).append(cod)
    return produtos, por_categoria, sorted(marcas), sorted(fabricantes), sorted(formas_pagamento), sorted(canais)


def gerar_cpf(rng):
    digitos = list(rng.integers(0, 10, size=9))

    def dv(digitos, pesos):
        s = sum(d * p for d, p in zip(digitos, pesos))
        r = (s * 10) % 11
        return 0 if r == 10 else r

    d1 = dv(digitos, range(10, 1, -1))
    d2 = dv(digitos + [d1], range(11, 1, -1))
    d = digitos + [d1, d2]
    return f"{d[0]}{d[1]}{d[2]}.{d[3]}{d[4]}{d[5]}.{d[6]}{d[7]}{d[8]}-{d[9]}{d[10]}"


def preferencia_ponderada(opcoes, preferido_idx, peso_preferido, rng, n):
    pesos = np.full(len(opcoes), (1 - peso_preferido) / (len(opcoes) - 1))
    pesos[preferido_idx] = peso_preferido
    idx = rng.choice(len(opcoes), size=n, p=pesos)
    return [opcoes[i] for i in idx]


def gerar_clientes(rng, faker, marcas, formas_pagamento, canais, categorias):
    cpfs_usados = set()
    clientes = []
    for _ in range(N_CUSTOMERS):
        cpf = gerar_cpf(rng)
        while cpf in cpfs_usados:
            cpf = gerar_cpf(rng)
        cpfs_usados.add(cpf)

        ordinal_max = DATE_END.replace(year=DATE_END.year - 18).toordinal()
        ordinal_min = DATE_END.replace(year=DATE_END.year - 80).toordinal()
        nascimento = date.fromordinal(int(rng.integers(ordinal_min, ordinal_max)))

        cidade, uf = CITIES[rng.integers(0, len(CITIES))]
        lat0, lon0 = CITY_COORDS[(cidade, uf)]

        clientes.append({
            "nome_completo": faker.name(),
            "sexo": rng.choice(["M", "F"]),
            "cpf": cpf,
            "email": faker.unique.email(),
            "telefone": faker.phone_number(),
            "data_nascimento": nascimento.isoformat(),
            "cidade": cidade,
            "uf": uf,
            "cep": f"{rng.integers(10000, 99999)}-{rng.integers(100, 999)}",
            "latitude": round(lat0 + rng.uniform(-0.08, 0.08), 6),
            "longitude": round(lon0 + rng.uniform(-0.08, 0.08), 6),
            "pref_categoria": categorias[rng.integers(0, len(categorias))],
            "peso_categoria": rng.uniform(0.55, 0.75),
            "pref_marca": marcas[rng.integers(0, len(marcas))],
            "peso_marca": rng.uniform(0.5, 0.7),
            "pref_pagamento": formas_pagamento[rng.integers(0, len(formas_pagamento))],
            "peso_pagamento": rng.uniform(0.5, 0.75),
            "pref_canal": canais[rng.integers(0, len(canais))],
            "peso_canal": rng.uniform(0.5, 0.75),
        })
    return clientes


def gerar_num_transacoes(rng):
    # Gamma-Poisson (binomial negativa): sobredisperso, gera cauda longa.
    taxa = rng.gamma(shape=0.5, scale=11.33, size=N_CUSTOMERS)
    n = 1 + rng.poisson(taxa)
    n = np.clip(n, 1, MAX_TRANSACOES_POR_CLIENTE)

    diff = N_TRANSACTIONS - n.sum()
    idx_ajustaveis = np.arange(N_CUSTOMERS)
    while diff != 0:
        i = rng.choice(idx_ajustaveis)
        if diff > 0 and n[i] < MAX_TRANSACOES_POR_CLIENTE:
            n[i] += 1
            diff -= 1
        elif diff < 0 and n[i] > 1:
            n[i] -= 1
            diff += 1
    return n


def gerar_datas_cliente(rng, n):
    if n == 1:
        return [DATE_START + timedelta(days=int(rng.integers(0, TOTAL_DAYS + 1)))]
    gaps = rng.gamma(shape=0.7, scale=1.0, size=n + 1)
    offsets = np.cumsum(gaps / gaps.sum() * TOTAL_DAYS)[:-1]
    return [DATE_START + timedelta(days=int(o)) for o in offsets]


def gerar_transacoes(rng, faker, clientes, produtos, por_categoria, marcas, fabricantes, formas_pagamento, canais, categorias):
    num_transacoes = gerar_num_transacoes(rng)
    linhas = []

    for cliente, n in zip(clientes, num_transacoes):
        datas = gerar_datas_cliente(rng, int(n))

        idx_cat_pref = categorias.index(cliente["pref_categoria"])
        idx_marca_pref = marcas.index(cliente["pref_marca"])
        idx_pgto_pref = formas_pagamento.index(cliente["pref_pagamento"])
        idx_canal_pref = canais.index(cliente["pref_canal"])

        pesos_categoria_base = np.full(len(categorias), (1 - cliente["peso_categoria"]) / (len(categorias) - 1))
        pesos_categoria_base[idx_cat_pref] = cliente["peso_categoria"]

        marca_amostrada = preferencia_ponderada(marcas, idx_marca_pref, cliente["peso_marca"], rng, len(datas))
        pagamento_amostrado = preferencia_ponderada(formas_pagamento, idx_pgto_pref, cliente["peso_pagamento"], rng, len(datas))
        canal_amostrado = preferencia_ponderada(canais, idx_canal_pref, cliente["peso_canal"], rng, len(datas))

        for i, data_compra in enumerate(datas):
            mes = data_compra.month
            pesos_sazonais = np.array([CATEGORY_SEASONALITY[c][mes - 1] for c in categorias])
            pesos_finais = pesos_categoria_base * pesos_sazonais
            pesos_finais /= pesos_finais.sum()
            categoria = categorias[rng.choice(len(categorias), p=pesos_finais)]

            cod_produto = por_categoria[categoria][rng.integers(0, len(por_categoria[categoria]))]
            descricao_produto, _ = produtos[cod_produto]
            fabricante = fabricantes[rng.integers(0, len(fabricantes))]

            mu, sigma = CATEGORY_PRICE_PARAMS[categoria]
            valor_compra = round(float(rng.lognormal(mu, sigma)), 2)
            quantidade = int(rng.integers(1, 11))
            subtotal = valor_compra * quantidade
            pct_desconto = rng.triangular(0.0, 0.05, 0.15)
            desconto = round(subtotal * pct_desconto, 2)
            valor_total = round(subtotal - desconto, 2)

            linhas.append({
                "nome_completo": cliente["nome_completo"],
                "sexo": cliente["sexo"],
                "cpf": cliente["cpf"],
                "email": cliente["email"],
                "telefone": cliente["telefone"],
                "data_nascimento": cliente["data_nascimento"],
                "data_compra": data_compra.isoformat(),
                "cod_produto": cod_produto,
                "descricao_produto": descricao_produto,
                "categoria_produto": categoria,
                "marca": marca_amostrada[i],
                "fabricante": fabricante,
                "valor_compra": valor_compra,
                "quantidade": quantidade,
                "desconto": desconto,
                "valor_total": valor_total,
                "forma_pagamento": pagamento_amostrado[i],
                "canal_venda": canal_amostrado[i],
                "vendedor": faker.name(),
                "cidade": cliente["cidade"],
                "uf": cliente["uf"],
                "cep": cliente["cep"],
                "latitude": cliente["latitude"],
                "longitude": cliente["longitude"],
                "descricao_compra": f"Cliente comprou {quantidade} unidade(s) de {descricao_produto}.",
            })

    return pd.DataFrame(linhas, columns=COLUMNS)


def self_check(df):
    print("\n--- self-check ---")
    por_cpf = df.groupby("cpf").size()
    print(f"CPFs únicos: {por_cpf.shape[0]} (linhas: {len(df)})")
    print("Transações por cliente -> "
          f"min={por_cpf.min()} mediana={por_cpf.median():.1f} "
          f"p90={por_cpf.quantile(0.9):.1f} p99={por_cpf.quantile(0.99):.1f} max={por_cpf.max()}")

    inconsist = df.groupby("cpf")[["nome_completo", "email", "data_nascimento"]].nunique()
    print("CPFs com >1 nome/email/nascimento associado:",
          (inconsist["nome_completo"] > 1).sum(),
          (inconsist["email"] > 1).sum(),
          (inconsist["data_nascimento"] > 1).sum())

    esperado = (df["valor_compra"] * df["quantidade"] - df["desconto"]).round(2)
    print("Máx. divergência valor_total vs valor_compra*quantidade-desconto:",
          (df["valor_total"] - esperado).abs().max())

    for (cidade, uf), (lat0, lon0) in CITY_COORDS.items():
        sub = df[(df["cidade"] == cidade) & (df["uf"] == uf)]
        if len(sub):
            dist = ((sub["latitude"] - lat0).abs().max(), (sub["longitude"] - lon0).abs().max())
            print(f"{cidade}/{uf}: desvio máx. lat/lon = {dist}")

    print("\nDescribe valor_compra:")
    print(df["valor_compra"].describe())


def main():
    rng = np.random.default_rng(SEED)
    faker = Faker("pt_BR")
    faker.seed_instance(SEED)

    produtos, por_categoria, marcas, fabricantes, formas_pagamento, canais = ler_tabelas_referencia(SOURCE_CSV)
    categorias = sorted(por_categoria.keys())

    clientes = gerar_clientes(rng, faker, marcas, formas_pagamento, canais, categorias)
    df = gerar_transacoes(rng, faker, clientes, produtos, por_categoria, marcas, fabricantes, formas_pagamento, canais, categorias)
    df = df.sort_values("data_compra").reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"Gravado {OUTPUT_CSV} com {len(df)} linhas.")

    self_check(df)


if __name__ == "__main__":
    main()
