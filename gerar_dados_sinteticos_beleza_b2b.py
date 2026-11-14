"""Base sintética de beleza B2B — uma empresa fictícia de perfumaria e beleza
que vende para revendedores e lojas, com TODOS os tipos de dado do mesmo cliente:

  pedidos.parquet        cabeçalho do pedido (canal, pagamento, valor)
  itens_pedido.parquet   linhas do pedido (SKU, quantidade, preço, desconto)
  navegacao.parquet      busca, visualização, carrinho e abandono (aplicativo e site)
  credito.parquet        limite, disponível, atraso e bloqueio (estado por ciclo)
  incentivo.parquet      programa de incentivo: nível e meta no início do ciclo, pontos no fim
  clientes.parquet       cadastro
  produtos.parquet       catálogo, com lançamento e descontinuação
  campanhas.parquet      SKUs em promoção em cada ciclo de campanha

Especificação em PLANO_FASE3.md (seção 3, "Base sintética de beleza B2B"). Serve
às hipóteses F3-H7 (vários tipos de dado do mesmo cliente) e F3-H8 (formas
extremas e estrutura conhecida). Não é dado real: prova capacidade do motor,
não valor de negócio.

Estrutura plantada, com gabarito em gabarito/ (NUNCA entra no motor nem na régua):
  - segmento latente do cliente: define o mix de categorias e a faixa de preço;
  - itens de reposição: cada cliente recompra um conjunto próprio de SKUs quando passa o
    período de cada um desde a última compra (até 60% das linhas do primeiro pedido do ciclo);
  - compra conjunta: regras SKU A -> SKU B dentro da mesma linha (perfume e loção, xampu e condicionador);
  - navegação que antecede a compra: parte do que é visto entra no pedido seguinte
    e parte vira interesse, comprado no ciclo seguinte;
  - crédito: pouco limite disponível encolhe o pedido; atraso e bloqueio aumentam o churn;
  - incentivo: quem termina o ciclo perto da meta faz um pedido de complemento nos últimos dias;
  - sensibilidade a desconto: define a parcela de itens comprados em promoção.

Formas: pedidos de 1 a centenas de linhas; clientes de 1 a milhares de pedidos;
catálogo com lançamentos e descontinuações a cada ciclo de 21 dias; 7 anos.

Uso: python3 gerar_dados_sinteticos_beleza_b2b.py --tamanho mini|pequeno|grande [--saida dados_beleza_b2b] [--seed 0]
"""

import argparse
import json
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

TAMANHOS = {  # clientes, SKUs (alvo)
    "mini": (2_000, 1_500),
    "pequeno": (15_000, 6_000),
    "grande": (100_000, 30_000),
}
DIA0 = np.datetime64("2019-01-01")
N_DIAS = 2557            # 2019-01-01 a 2025-12-31
CICLO = 21
N_CICLOS = -(-N_DIAS // CICLO)

CATS = ["perfumaria", "maquiagem", "cabelos", "corpo_banho", "pele", "masculino", "infantil", "kits"]
FAMILIAS = {   # família da linha -> (peso, [(tipo de produto, categoria, preço médio em R$)])
    "fragrancia_feminina": (0.22, [("Perfume", "perfumaria", 150), ("Body Splash", "perfumaria", 60),
                                   ("Loção Hidratante", "corpo_banho", 40), ("Sabonete", "corpo_banho", 15), ("Kit Presente", "kits", 190)]),
    "fragrancia_masculina": (0.10, [("Perfume Masculino", "masculino", 160), ("Desodorante Masculino", "masculino", 30),
                                    ("Pós-barba", "masculino", 55), ("Sabonete", "corpo_banho", 15), ("Kit Presente", "kits", 180)]),
    "capilar": (0.18, [("Xampu", "cabelos", 32), ("Condicionador", "cabelos", 34), ("Máscara Capilar", "cabelos", 48),
                       ("Leave-in", "cabelos", 40), ("Óleo Capilar", "cabelos", 45)]),
    "maquiagem": (0.20, [("Base", "maquiagem", 60), ("Pó Compacto", "maquiagem", 50), ("Batom", "maquiagem", 35),
                         ("Máscara de Cílios", "maquiagem", 45), ("Delineador", "maquiagem", 32), ("Sombra", "maquiagem", 38),
                         ("Blush", "maquiagem", 40), ("Esmalte", "maquiagem", 12)]),
    "pele": (0.12, [("Sérum Facial", "pele", 95), ("Hidratante Facial", "pele", 70), ("Protetor Solar", "pele", 65),
                    ("Água Micelar", "pele", 40), ("Tônico", "pele", 45)]),
    "corpo": (0.12, [("Loção Hidratante", "corpo_banho", 40), ("Óleo Corporal", "corpo_banho", 50), ("Esfoliante", "corpo_banho", 42),
                     ("Desodorante", "corpo_banho", 18), ("Creme para Mãos", "corpo_banho", 25), ("Sabonete", "corpo_banho", 15)]),
    "infantil": (0.06, [("Xampu Infantil", "infantil", 28), ("Colônia Infantil", "infantil", 40),
                        ("Sabonete Infantil", "infantil", 14), ("Loção Infantil", "infantil", 30)]),
}
COMPLEMENTOS = [  # dentro da mesma linha: tipo A -> tipo B
    ("Perfume", "Loção Hidratante"), ("Perfume", "Body Splash"), ("Perfume", "Sabonete"),
    ("Perfume Masculino", "Desodorante Masculino"), ("Perfume Masculino", "Pós-barba"),
    ("Xampu", "Condicionador"), ("Condicionador", "Máscara Capilar"), ("Xampu", "Leave-in"),
    ("Base", "Pó Compacto"), ("Máscara de Cílios", "Delineador"), ("Batom", "Blush"),
    ("Sérum Facial", "Hidratante Facial"), ("Hidratante Facial", "Protetor Solar"),
    ("Xampu Infantil", "Sabonete Infantil"), ("Loção Hidratante", "Sabonete"),
]
NOMES_LINHA = ["Aurora", "Luar", "Brisa", "Jardim", "Solar", "Serena", "Vívida", "Encanto", "Âmbar", "Orquídea", "Maré", "Dália",
               "Íris", "Cedro", "Safira", "Néctar", "Aura", "Pétala", "Fluir", "Raiz", "Essência", "Onda", "Veludo", "Lume"]
MARCAS = ["Florata", "Vitalis", "Nativa", "Belavie", "Dermax", "Capilé", "Urbano", "Pequenos", "Matiz", "Savana"]
SEGMENTOS = ["perfumaria_feminina", "masculino", "maquiagem", "cabelos", "dia_a_dia", "pele", "presentes", "multicategoria"]
#                 perf  maq   cab   corpo pele  masc  inf   kits
PREF_CAT = np.array([
    [0.50, 0.10, 0.04, 0.20, 0.04, 0.01, 0.01, 0.10],   # perfumaria_feminina
    [0.10, 0.01, 0.10, 0.15, 0.03, 0.50, 0.01, 0.10],   # masculino
    [0.10, 0.60, 0.03, 0.05, 0.15, 0.01, 0.01, 0.05],   # maquiagem
    [0.04, 0.05, 0.60, 0.15, 0.10, 0.03, 0.02, 0.01],   # cabelos
    [0.10, 0.10, 0.25, 0.35, 0.03, 0.02, 0.15, 0.00],   # dia_a_dia
    [0.04, 0.15, 0.05, 0.15, 0.55, 0.02, 0.02, 0.02],   # pele
    [0.30, 0.04, 0.02, 0.10, 0.02, 0.15, 0.02, 0.35],   # presentes
    [0.18, 0.18, 0.16, 0.16, 0.10, 0.10, 0.05, 0.07],   # multicategoria
])
PREF_FAIXA = np.array([   # preferência por faixa de preço (popular, intermediária, premium)
    [0.2, 0.4, 0.4], [0.3, 0.4, 0.3], [0.4, 0.4, 0.2], [0.4, 0.4, 0.2],
    [0.6, 0.3, 0.1], [0.1, 0.4, 0.5], [0.2, 0.4, 0.4], [0.35, 0.4, 0.25]])
TIPOS_CLIENTE = ["revendedor", "lider", "loja"]
P_TIPO = [0.86, 0.10, 0.04]
P_SEG_TIPO = np.array([
    [0.20, 0.10, 0.18, 0.14, 0.18, 0.08, 0.10, 0.02],
    [0.15, 0.10, 0.15, 0.12, 0.15, 0.08, 0.05, 0.20],
    [0.05, 0.03, 0.05, 0.07, 0.10, 0.05, 0.05, 0.60]])
CANAIS = ["aplicativo", "site", "representante", "televendas"]
PAGAMENTOS = ["credito_empresa", "pix", "cartao"]
NAV_TIPOS = ["busca", "ver_produto", "ver_campanha", "add_carrinho", "abandono_carrinho"]
UFS = ["SP", "MG", "RJ", "PR", "RS", "BA", "SC", "PE", "GO", "CE", "PA", "ES", "MT", "MS", "DF", "AM"]
P_UF = np.array([22, 11, 9, 7, 7, 7, 5, 5, 4, 4, 4, 3, 3, 3, 3, 3], dtype=float)
MAX_REPOSICAO = 160


def sazonalidade():
    """fator de demanda por ciclo (datas de presente do varejo brasileiro) e reforço de kits."""
    fator, kits = np.ones(N_CICLOS), np.ones(N_CICLOS)
    mult = {1: 0.80, 5: 1.25, 6: 1.15, 8: 1.10, 11: 1.30, 12: 1.45}
    for c in range(N_CICLOS):
        mes = (DIA0 + np.timedelta64(min(c * CICLO + CICLO // 2, N_DIAS - 1), "D")).astype("datetime64[M]").astype(int) % 12 + 1
        fator[c] = mult.get(mes, 1.0)
        kits[c] = 2.5 if mes in (5, 6, 8, 12) else 1.0
    return fator, kits


def gerar_catalogo(rng, n_sku_alvo):
    fams = list(FAMILIAS)
    pesos = np.array([FAMILIAS[f][0] for f in fams])
    cols = {k: [] for k in ("linha_id", "familia", "tipo_produto", "categoria", "marca", "faixa", "variante", "preco_lista", "nome")}
    linha = 0
    while len(cols["linha_id"]) < n_sku_alvo:
        f = fams[rng.choice(len(fams), p=pesos / pesos.sum())]
        marca = MARCAS[rng.randint(len(MARCAS))]
        faixa = int(rng.choice(3, p=[0.4, 0.4, 0.2]))
        nome_linha = f"{NOMES_LINHA[rng.randint(len(NOMES_LINHA))]} {linha}"
        for tipo, cat, preco in FAMILIAS[f][1]:
            if rng.rand() > 0.8:
                continue
            for v in range(rng.randint(1, 5)):
                cols["linha_id"].append(linha); cols["familia"].append(f); cols["tipo_produto"].append(tipo)
                cols["categoria"].append(cat); cols["marca"].append(marca); cols["faixa"].append(faixa); cols["variante"].append(v)
                cols["preco_lista"].append(round(float(preco * [0.7, 1.0, 1.6][faixa] * np.exp(rng.normal(0, 0.15)) * (1 + 0.25 * v)), 2))
                cols["nome"].append(f"{tipo} {nome_linha} {marca} v{v + 1}")
        linha += 1
    n = len(cols["linha_id"])
    cat = {k: np.array(v) for k, v in cols.items()}
    cat["sku"] = np.arange(n, dtype=np.int32)
    cat["cat_id"] = np.array([CATS.index(c) for c in cat["categoria"]], dtype=np.int8)
    # lançamento: 60% já existiam; o resto entra ao longo dos ciclos. Vida útil média de ~40 ciclos.
    lanc = np.where(rng.rand(n) < 0.6, 0, rng.randint(1, N_CICLOS, n)).astype(np.int16)
    vida = np.maximum(6, rng.gamma(2.0, 20.0, n)).astype(np.int32)
    perene = rng.rand(n) < 0.25
    cat["ciclo_lancamento"] = lanc
    cat["ciclo_fim"] = np.where(perene, N_CICLOS + 1, np.minimum(lanc + vida, N_CICLOS + 1)).astype(np.int16)
    cat["popularidade"] = np.exp(rng.normal(0, 1.0, n))
    # regras de compra conjunta dentro da linha
    ra, rb, rp = [], [], []
    por_linha = {}
    for i in range(n):
        por_linha.setdefault(int(cat["linha_id"][i]), []).append(i)
    pares = set(COMPLEMENTOS)
    for skus in por_linha.values():
        for a in skus:
            for b in skus:
                if (cat["tipo_produto"][a], cat["tipo_produto"][b]) in pares and rng.rand() < 0.6:
                    ra.append(a); rb.append(b); rp.append(rng.uniform(0.3, 0.7))
    ordem = np.argsort(ra, kind="stable")
    cat["regra_a"] = np.array(ra, dtype=np.int32)[ordem]
    cat["regra_b"] = np.array(rb, dtype=np.int32)[ordem]
    cat["regra_p"] = np.array(rp)[ordem]
    cont = np.bincount(cat["regra_a"], minlength=n)
    cat["regra_ini"] = np.r_[0, np.cumsum(cont)[:-1]]
    cat["regra_n"] = cont
    return cat


def gerar_clientes(rng, n):
    cli = {"cliente_id": np.arange(n, dtype=np.int32)}
    tipo = rng.choice(3, n, p=P_TIPO).astype(np.int8)
    cli["tipo"] = tipo
    cli["segmento"] = np.array([rng.choice(8, p=P_SEG_TIPO[t]) for t in tipo], dtype=np.int8)
    # 30% da base já existe no início; o resto entra com crescimento ao longo dos 7 anos
    entrada = np.where(rng.rand(n) < 0.30, 0, np.sqrt(rng.rand(n)) * (N_CICLOS - 2) + 1).astype(np.int16)
    cli["ciclo_entrada"] = entrada
    # frequência (pedidos por ciclo enquanto ativo), linhas por pedido e quantidade por linha
    freq = np.where(tipo == 0, rng.gamma(2.0, 0.30, n), np.where(tipo == 1, rng.gamma(3.0, 0.6, n), rng.gamma(2.0, 1.0, n)))
    rede = (tipo == 2) & (rng.rand(n) < 0.03)            # grandes redes: pedido quase diário -> milhares de pedidos
    freq = np.where(rede, rng.uniform(10, 18, n), freq)
    cli["freq"] = np.maximum(freq, 0.05)
    lin = np.where(tipo == 0, np.clip(np.exp(rng.normal(np.log(4), 0.5, n)), 1, 40),
                   np.where(tipo == 1, np.clip(np.exp(rng.normal(np.log(15), 0.5, n)), 5, 120),
                            np.clip(np.exp(rng.normal(np.log(45), 0.8, n)), 10, 450)))
    cli["linhas"] = lin
    cli["qtd"] = np.where(tipo == 0, 1.4, np.where(tipo == 1, 3.0, 10.0)) * np.exp(rng.normal(0, 0.2, n))
    cli["sens_desconto"] = rng.beta(2, 2, n)
    cli["engajamento"] = rng.beta(2, 2.5, n)
    cli["risco_credito"] = rng.beta(1.5, 7, n)
    # risco de saída por ciclo: muito heterogêneo (há quem saia logo e quem fique os 7 anos)
    risco = np.where(tipo == 0, rng.beta(1.2, 25, n), np.where(tipo == 1, rng.beta(1.2, 60, n), rng.beta(1.2, 120, n)))
    cli["risco_saida"] = np.where(rede, 0.2 * risco, risco)
    cli["so_um_pedido"] = (tipo == 0) & (rng.rand(n) < 0.10)
    cli["rede"] = rede
    cli["ciclo_entrada"] = np.where(rede & (rng.rand(n) < 0.5), 0, entrada).astype(np.int16)   # metade das redes já existe no início
    cli["usa_credito"] = rng.rand(n) < np.array([0.6, 0.85, 0.95])[tipo]
    cli["incentivo"] = rng.rand(n) < np.array([0.5, 0.9, 0.3])[tipo]
    cli["uf"] = rng.choice(len(UFS), n, p=P_UF / P_UF.sum()).astype(np.int8)
    cli["canal_aquisicao"] = rng.choice(4, n, p=[0.35, 0.25, 0.3, 0.1]).astype(np.int8)   # indicação, digital, representante, evento
    cli["idade"] = np.where(tipo == 2, np.nan, np.clip(rng.normal(38, 11, n), 18, 75).round())
    cli["n_reposicao"] = np.clip((lin * rng.uniform(1.2, 2.5, n)).round(), 1, MAX_REPOSICAO).astype(np.int16)
    return cli


class Escritor:
    def __init__(self, pasta, nome):
        self.arq, self.w, self.n = os.path.join(pasta, f"{nome}.parquet"), None, 0

    def escrever(self, colunas):
        if not len(next(iter(colunas.values()))):
            return
        t = pa.table(colunas)
        if self.w is None:
            self.w = pq.ParquetWriter(self.arq, t.schema, compression="zstd")
        self.w.write_table(t)
        self.n += t.num_rows

    def fechar(self):
        if self.w is not None:
            self.w.close()


def amostrar(rng, pesos, n):
    """n sorteios com reposição proporcionais a `pesos` (retorna índices)."""
    cdf = np.cumsum(pesos)
    return np.minimum(np.searchsorted(cdf, rng.rand(n) * cdf[-1]), len(pesos) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tamanho", default="pequeno", choices=list(TAMANHOS))
    ap.add_argument("--saida", default="dados_beleza_b2b")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    rng = np.random.RandomState(a.seed)
    n_cli, n_sku_alvo = TAMANHOS[a.tamanho]
    pasta = os.path.join(a.saida, a.tamanho)
    os.makedirs(os.path.join(pasta, "gabarito"), exist_ok=True)
    cat = gerar_catalogo(rng, n_sku_alvo)
    cli = gerar_clientes(rng, n_cli)
    n_sku = len(cat["sku"])
    saz, saz_kits = sazonalidade()
    KITS = CATS.index("kits")

    # itens de reposição de cada cliente (matriz com preenchimento -1), período em ciclos e fase
    rep_sku = np.full((n_cli, MAX_REPOSICAO), -1, dtype=np.int32)
    rep_per = rng.choice([1, 2, 3, 4], (n_cli, MAX_REPOSICAO), p=[0.3, 0.35, 0.2, 0.15]).astype(np.int8)
    rep_ult = np.full((n_cli, MAX_REPOSICAO), -100, dtype=np.int16)      # ciclo da última compra de cada item de reposição
    rep_mask = np.arange(MAX_REPOSICAO)[None, :] < cli["n_reposicao"][:, None]

    # estado por cliente
    ativo = np.zeros(n_cli, bool)
    ciclos_ativo = np.zeros(n_cli, np.int32)
    n_pedidos = np.zeros(n_cli, np.int64)
    ciclo_saida = np.full(n_cli, -1, np.int16)
    nivel = np.zeros(n_cli, np.int8)
    falhas = np.zeros(n_cli, np.int8)
    metas_bimestre = np.zeros(n_cli, np.int8)
    gasto_medio = np.zeros(n_cli)          # média móvel do gasto por ciclo (base do limite e da meta)
    fatura = np.zeros(n_cli)               # compras a crédito do ciclo anterior, ainda em aberto
    atraso = np.zeros(n_cli, np.int16)
    bloqueado = np.zeros(n_cli, bool)
    interesse_cli, interesse_sku = np.zeros(0, np.int32), np.zeros(0, np.int32)

    esc = {n: Escritor(pasta, n) for n in ("pedidos", "itens_pedido", "navegacao", "credito", "incentivo", "campanhas")}
    esc["g_pedidos"] = Escritor(os.path.join(pasta, "gabarito"), "pedidos_origem")
    esc["g_itens"] = Escritor(os.path.join(pasta, "gabarito"), "itens_origem")
    prox_pedido, prox_nav = 0, 0
    hist_linhas, hist_valor = [], []

    for c in range(N_CICLOS):
        d0 = c * CICLO
        dias_ciclo = min(CICLO, N_DIAS - d0)
        sku_ativo = (cat["ciclo_lancamento"] <= c) & (cat["ciclo_fim"] > c)
        ids_ativos = np.flatnonzero(sku_ativo)
        # peso de cada SKU ativo por segmento: a preferência por categoria é do segmento; dentro da categoria
        # pesam a faixa de preço, a popularidade e o lançamento recente (kits ganham peso nas datas de presente)
        ca = cat["cat_id"][ids_ativos]
        base = PREF_FAIXA[:, cat["faixa"][ids_ativos]] * (cat["popularidade"][ids_ativos]
                                                         * np.where(c - cat["ciclo_lancamento"][ids_ativos] < 2, 3.0, 1.0))[None, :]
        soma = np.stack([base[:, ca == k].sum(1) for k in range(len(CATS))], axis=1)
        pref_c = PREF_CAT.copy()
        pref_c[:, KITS] *= saz_kits[c]
        w_ciclo = pref_c[:, ca] * base / np.maximum(soma[:, ca], 1e-12)
        # campanha do ciclo: ~6% dos SKUs ativos em promoção
        em_promo = np.zeros(n_sku, bool)
        promo_ids = rng.choice(ids_ativos, max(1, int(0.06 * len(ids_ativos))), replace=False)
        em_promo[promo_ids] = True
        desc_promo = np.zeros(n_sku)
        desc_promo[promo_ids] = rng.choice([0.10, 0.15, 0.20, 0.30, 0.40], len(promo_ids), p=[0.25, 0.25, 0.25, 0.15, 0.10])
        esc["campanhas"].escrever({"ciclo": np.full(len(promo_ids), c, np.int16), "sku": promo_ids.astype(np.int32),
                                   "desconto_pct": desc_promo[promo_ids].astype(np.float32),
                                   "data_inicio": np.full(len(promo_ids), DIA0 + np.timedelta64(d0, "D")).astype("datetime64[ms]")})
        pos_promo = np.flatnonzero(em_promo[ids_ativos])            # posições dos promocionais dentro de ids_ativos

        # entradas e itens de reposição (novos clientes; SKUs descontinuados são substituídos)
        novos = np.flatnonzero(cli["ciclo_entrada"] == c)
        ativo[novos] = True
        trocar = rep_mask & ((rep_sku < 0) | ~sku_ativo[np.maximum(rep_sku, 0)]) & ativo[:, None]
        if trocar.any():
            li, co = np.nonzero(trocar)
            for s in range(8):
                m = cli["segmento"][li] == s
                if m.any():
                    rep_sku[li[m], co[m]] = ids_ativos[amostrar(rng, w_ciclo[s], int(m.sum()))]

        # ---- crédito: pagamento da fatura do ciclo anterior, atraso e bloqueio
        cred = ativo & cli["usa_credito"]
        tinha_fatura = cred & (fatura > 0)
        atrasou = tinha_fatura & (rng.rand(n_cli) < np.clip(cli["risco_credito"] * 2.5, 0, 0.9))
        atraso = np.where(atrasou, (5 + rng.exponential(20, n_cli)).astype(np.int16), 0).astype(np.int16)
        bloqueado = atrasou & (atraso > 30)
        limite = np.maximum(500.0, 2.5 * gasto_medio + np.array([800.0, 4000.0, 30000.0])[cli["tipo"]]) * (1 + 0.01 * np.minimum(ciclos_ativo, 60))
        em_aberto = np.where(atrasou, fatura, 0.0)       # quem pagou em dia começa o ciclo sem saldo devedor
        disponivel = np.maximum(limite - em_aberto, 0.0)
        ic = np.flatnonzero(cred)
        esc["credito"].escrever({"cliente_id": ic.astype(np.int32),
                                 "data": np.full(len(ic), DIA0 + np.timedelta64(d0, "D")).astype("datetime64[ms]"),
                                 "limite": limite[ic].round(2), "utilizado": em_aberto[ic].round(2), "disponivel": disponivel[ic].round(2),
                                 "dias_atraso": atraso[ic], "bloqueado": bloqueado[ic]})

        # ---- incentivo: nível e meta no início do ciclo
        inc = ativo & cli["incentivo"]
        meta = np.maximum(300.0, 1.10 * gasto_medio) * (1 + 0.25 * nivel)
        ii = np.flatnonzero(inc)
        esc["incentivo"].escrever({"cliente_id": ii.astype(np.int32),
                                   "data": np.full(len(ii), DIA0 + np.timedelta64(d0, "D")).astype("datetime64[ms]"),
                                   "ciclo": np.full(len(ii), c, np.int16), "evento": np.full(len(ii), "inicio_ciclo"),
                                   "nivel": nivel[ii], "meta_ciclo": meta[ii].round(2),
                                   "pontos_ciclo": np.full(len(ii), np.nan), "atingiu_meta": np.full(len(ii), np.nan)})

        # ---- pedidos regulares do ciclo
        seg_pres = cli["segmento"] == SEGMENTOS.index("presentes")
        taxa = cli["freq"] * np.where(seg_pres, saz[c] ** 2, saz[c]) * np.where(bloqueado, 0.3, 1.0) * (dias_ciclo / CICLO)
        n_ord = np.where(ativo, rng.poisson(taxa), 0)
        n_ord[novos] = np.maximum(n_ord[novos], 1)                # todo cliente novo faz ao menos o primeiro pedido
        n_ord[cli["so_um_pedido"] & (n_pedidos > 0)] = 0
        ped_cli = np.repeat(np.arange(n_cli, dtype=np.int32), n_ord)
        ped_dia = rng.randint(0, dias_ciclo, len(ped_cli))
        ped_comp = np.zeros(len(ped_cli), bool)

        def montar(ped_cli, ped_dia, linhas_alvo, com_reposicao):
            """sorteia as linhas dos pedidos -> (índice do pedido, sku, origem). origem: 0 pref, 1 promo, 2 interesse, 3 reposição, 4 compra conjunta"""
            n_p = len(ped_cli)
            if not n_p:
                return np.zeros(0, np.int64), np.zeros(0, np.int32), np.zeros(0, np.int8)
            partes_o, partes_s, partes_f = [], [], []
            n_rep_usado = np.zeros(n_p, np.int64)
            if com_reposicao:
                # itens de reposição vencidos entram no 1º pedido do cliente no ciclo
                primeiro = np.r_[True, ped_cli[1:] != ped_cli[:-1]]
                op = np.flatnonzero(primeiro)
                cp = ped_cli[op]
                vence = rep_mask[cp] & (rep_sku[cp] >= 0) & ((c - rep_ult[cp]) >= rep_per[cp])
                # no máximo 60% das linhas do pedido; a prioridade entre os vencidos gira a cada ciclo
                giro = (np.arange(MAX_REPOSICAO)[None, :] + rng.randint(0, MAX_REPOSICAO, len(cp))[:, None]) % MAX_REPOSICAO
                vg = np.take_along_axis(vence, giro, axis=1)
                vg &= np.cumsum(vg, axis=1) <= np.ceil(0.6 * linhas_alvo[op])[:, None]
                lo, cg = np.nonzero(vg)
                co = giro[lo, cg]
                rep_ult[cp[lo], co] = c
                partes_o.append(op[lo]); partes_s.append(rep_sku[cp[lo], co]); partes_f.append(np.full(len(lo), 3, np.int8))
                n_rep_usado[op] = vg.sum(1)
            resto = np.maximum(linhas_alvo - n_rep_usado, 0)
            o = np.repeat(np.arange(n_p), resto)
            ci = ped_cli[o]
            u = rng.rand(len(o))
            p_promo = 0.10 + 0.45 * cli["sens_desconto"][ci]
            tem_int = np.zeros(n_cli, np.int64)
            if len(interesse_cli):
                tem_int = np.bincount(interesse_cli, minlength=n_cli)
            p_int = np.where(tem_int[ci] > 0, 0.15, 0.0)
            fonte = np.where(u < p_promo, 1, np.where(u < p_promo + p_int, 2, 0)).astype(np.int8)
            sku = np.zeros(len(o), np.int32)
            seg = cli["segmento"][ci]
            for s in range(8):
                m = (seg == s) & (fonte == 0)
                if m.any():
                    sku[m] = ids_ativos[amostrar(rng, w_ciclo[s], int(m.sum()))]
                m = (seg == s) & (fonte == 1)
                if m.any():
                    sku[m] = ids_ativos[pos_promo[amostrar(rng, w_ciclo[s][pos_promo], int(m.sum()))]]
            m = fonte == 2
            if m.any():
                ini = np.r_[0, np.cumsum(tem_int)[:-1]]
                esc_i = ini[ci[m]] + (rng.rand(int(m.sum())) * tem_int[ci[m]]).astype(np.int64)
                sku[m] = interesse_sku[esc_i]
            partes_o.append(o); partes_s.append(sku); partes_f.append(fonte)
            o, sku, fonte = np.concatenate(partes_o), np.concatenate(partes_s), np.concatenate(partes_f)
            # compra conjunta: A no pedido puxa B com a probabilidade da regra
            tem = cat["regra_n"][sku] > 0
            if tem.any():
                r = cat["regra_ini"][sku[tem]] + (rng.rand(int(tem.sum())) * cat["regra_n"][sku[tem]]).astype(np.int64)
                vai = (rng.rand(len(r)) < cat["regra_p"][r]) & sku_ativo[cat["regra_b"][r]]
                o = np.r_[o, o[tem][vai]]; sku = np.r_[sku, cat["regra_b"][r][vai]]; fonte = np.r_[fonte, np.full(int(vai.sum()), 4, np.int8)]
            # um SKU por pedido (fica a primeira origem)
            _, idx = np.unique(o.astype(np.int64) * n_sku + sku, return_index=True)
            return o[idx], sku[idx], fonte[idx]

        aperto = np.clip(disponivel / np.maximum(gasto_medio, 200.0), 0.25, 1.0)        # pouco crédito encolhe o pedido
        alvo = cli["linhas"][ped_cli] * np.exp(rng.normal(0, 0.35, len(ped_cli))) * np.where(cli["usa_credito"][ped_cli], aperto[ped_cli], 1.0)
        alvo = np.maximum(alvo.round(), np.where(cli["tipo"][ped_cli] == 2, 2, 1)).astype(np.int64)
        ordem = np.lexsort((ped_dia, ped_cli))
        ped_cli, ped_dia, alvo = ped_cli[ordem], ped_dia[ordem], alvo[ordem]
        lo, lsku, lfonte = montar(ped_cli, ped_dia, alvo, True)

        def precificar(lo, lsku, ped_cli):
            ci = ped_cli[lo]
            qtd = 1 + rng.poisson(np.maximum(cli["qtd"][ci] - 1, 0.05))
            volume = np.array([0.0, 0.06, 0.12])[cli["tipo"][ci]]
            preco = cat["preco_lista"][lsku] * (1 - desc_promo[lsku]) * (1 - volume)
            return qtd.astype(np.int32), preco.round(2)

        lqtd, lpreco = precificar(lo, lsku, ped_cli)
        valor_ped = np.bincount(lo, weights=lqtd * lpreco, minlength=len(ped_cli))
        pontos = np.bincount(ped_cli, weights=valor_ped, minlength=n_cli)

        # ---- incentivo: pedido de complemento nos últimos dias, para quem ficou perto da meta
        perto = inc & (pontos >= 0.75 * meta) & (pontos < meta) & ~bloqueado & (dias_ciclo == CICLO)
        comp = np.flatnonzero(perto & (rng.rand(n_cli) < 0.6)).astype(np.int32)
        if len(comp):
            falta = meta[comp] - pontos[comp]
            valor_linha = np.maximum(cli["qtd"][comp] * 45.0, 30.0)
            alvo_c = np.clip(np.ceil(1.15 * falta / valor_linha), 1, 60).astype(np.int64)
            dia_c = rng.randint(CICLO - 4, CICLO, len(comp))
            lo2, lsku2, lf2 = montar(comp, dia_c, alvo_c, False)
            lq2, lp2 = precificar(lo2, lsku2, comp)
            base_o = len(ped_cli)
            ped_cli, ped_dia, ped_comp = np.r_[ped_cli, comp], np.r_[ped_dia, dia_c], np.r_[ped_comp, np.ones(len(comp), bool)]
            lo, lsku, lfonte, lqtd, lpreco = np.r_[lo, lo2 + base_o], np.r_[lsku, lsku2], np.r_[lfonte, lf2], np.r_[lqtd, lq2], np.r_[lpreco, lp2]
            valor_ped = np.bincount(lo, weights=lqtd * lpreco, minlength=len(ped_cli))
            pontos = np.bincount(ped_cli, weights=valor_ped, minlength=n_cli)
        n_p = len(ped_cli)

        # ---- cabeçalho dos pedidos
        digital = rng.rand(n_p) < (0.15 + 0.75 * cli["engajamento"][ped_cli])
        canal = np.where(digital, np.where(rng.rand(n_p) < 0.6, 0, 1), np.where((cli["tipo"][ped_cli] == 2) & (rng.rand(n_p) < 0.4), 3, 2)).astype(np.int8)
        a_credito = cli["usa_credito"][ped_cli] & ~bloqueado[ped_cli]
        pagamento = np.where(a_credito, 0, np.where(rng.rand(n_p) < 0.6, 1, 2)).astype(np.int8)
        minuto = rng.randint(7 * 60, 22 * 60, n_p)
        ts = (DIA0 + (d0 + ped_dia).astype("timedelta64[D]")).astype("datetime64[m]") + minuto.astype("timedelta64[m]")
        pid = prox_pedido + np.arange(n_p, dtype=np.int64)
        prox_pedido += n_p
        n_lin = np.bincount(lo, minlength=n_p)
        lista = np.bincount(lo, weights=lqtd * cat["preco_lista"][lsku], minlength=n_p)
        promo_lin = em_promo[lsku]
        esc["pedidos"].escrever({"pedido_id": pid, "cliente_id": ped_cli.astype(np.int32), "data": ts.astype("datetime64[ms]"),
                                 "ciclo": np.full(n_p, c, np.int16), "canal": np.array(CANAIS)[canal], "forma_pagamento": np.array(PAGAMENTOS)[pagamento],
                                 "n_linhas": n_lin.astype(np.int32), "valor_total": valor_ped.round(2), "valor_lista": lista.round(2),
                                 "tem_item_em_promocao": np.bincount(lo, weights=promo_lin, minlength=n_p) > 0})
        esc["itens_pedido"].escrever({"pedido_id": pid[lo], "cliente_id": ped_cli[lo].astype(np.int32), "data": ts[lo].astype("datetime64[ms]"),
                                      "sku": lsku, "quantidade": lqtd, "preco_unitario": lpreco,
                                      "desconto_pct": desc_promo[lsku].astype(np.float32), "em_promocao": promo_lin})
        esc["g_pedidos"].escrever({"pedido_id": pid, "pedido_de_complemento": ped_comp})
        esc["g_itens"].escrever({"pedido_id": pid[lo], "sku": lsku,
                                 "origem": np.array(["preferencia", "promocao", "interesse", "reposicao", "compra_conjunta"])[lfonte]})
        hist_linhas.append(n_lin); hist_valor.append(valor_ped)

        # ---- navegação: sessões antes dos pedidos (parte do que é visto é o que será comprado) e sessões sem compra
        eng = cli["engajamento"][ped_cli]
        com_sessao = rng.rand(n_p) < (0.25 + 0.7 * eng)
        nav = {k: [] for k in ("cli", "ts", "tipo", "sku", "cat")}

        def registrar(cli_, ts_, tipo_, sku_):
            nav["cli"].append(cli_.astype(np.int32)); nav["ts"].append(ts_); nav["tipo"].append(np.full(len(cli_), tipo_, np.int8))
            nav["sku"].append(sku_.astype(np.int32)); nav["cat"].append(np.where(sku_ >= 0, cat["cat_id"][np.maximum(sku_, 0)], -1).astype(np.int8))

        # cada pedido com navegação tem 1 ou 2 sessões nas 72 h anteriores; os eventos de uma sessão ficam a minutos uns dos outros
        sess = np.stack([rng.randint(20, 30 * 60, n_p), rng.randint(31 * 60, 72 * 60, n_p)], axis=1)
        duas = rng.rand(n_p) < 0.4

        def quando(o):
            """instante de um evento de navegação do pedido `o`: início de uma das sessões + alguns minutos."""
            k = np.where(duas[o], rng.randint(0, 2, len(o)), 0)
            return ts[o] - sess[o, k].astype("timedelta64[m]") + rng.randint(0, 15, len(o)).astype("timedelta64[m]")

        # vistas do que será comprado: cada linha de pedido com sessão é vista com probabilidade 0,35 (no máximo ~30 por pedido)
        m = com_sessao[lo] & (rng.rand(len(lo)) < np.minimum(0.35, 30.0 / np.maximum(n_lin[lo], 1)))
        registrar(ped_cli[lo][m], quando(lo[m]), 1, lsku[m])
        # carrinho: pedidos pelo aplicativo ou site
        m2 = m & (canal[lo] <= 1) & (rng.rand(len(lo)) < 0.7)
        registrar(ped_cli[lo][m2], ts[lo][m2] - rng.randint(1, 15, int(m2.sum())).astype("timedelta64[m]"), 3, lsku[m2])
        # vistas de interesse (não compradas agora): alimentam o pedido do ciclo seguinte
        op = np.flatnonzero(com_sessao)
        n_ruido = rng.poisson(3 + 10 * eng[op])
        oi = np.repeat(op, n_ruido)
        sk = np.zeros(len(oi), np.int32)
        seg_i = cli["segmento"][ped_cli[oi]]
        for s in range(8):
            mm = seg_i == s
            if mm.any():
                sk[mm] = ids_ativos[amostrar(rng, w_ciclo[s], int(mm.sum()))]
        registrar(ped_cli[oi], quando(oi), 1, sk)
        novo_int_cli, novo_int_sku = [ped_cli[oi]], [sk]
        # buscas (por categoria) e visualização de campanha
        nb = rng.poisson(1.0, len(op))
        ob = np.repeat(op, nb)
        registrar(ped_cli[ob], quando(ob), 0, np.full(len(ob), -1))
        nav["cat"][-1] = np.array([rng.choice(8, p=PREF_CAT[s] / PREF_CAT[s].sum()) for s in cli["segmento"][ped_cli[ob]]], dtype=np.int8) if len(ob) else np.zeros(0, np.int8)
        oc = op[rng.rand(len(op)) < 0.3]
        registrar(ped_cli[oc], quando(oc), 2, np.full(len(oc), -1))
        # sessões sem compra: clientes ativos e engajados que não pediram no ciclo; carrinho abandonado vira interesse forte
        sem = np.flatnonzero(ativo & (np.bincount(ped_cli, minlength=n_cli) == 0) & (rng.rand(n_cli) < 0.3 * cli["engajamento"]))
        if len(sem):
            nv = 1 + rng.poisson(2, len(sem))
            si = np.repeat(sem, nv)
            sk2 = np.zeros(len(si), np.int32)
            usa_promo = rng.rand(len(si)) < 0.4
            for s in range(8):
                mm = (cli["segmento"][si] == s) & ~usa_promo
                if mm.any():
                    sk2[mm] = ids_ativos[amostrar(rng, w_ciclo[s], int(mm.sum()))]
                mm = (cli["segmento"][si] == s) & usa_promo
                if mm.any():
                    sk2[mm] = ids_ativos[pos_promo[amostrar(rng, w_ciclo[s][pos_promo], int(mm.sum()))]]
            ini_s = (DIA0 + (d0 + rng.randint(0, dias_ciclo, len(sem))).astype("timedelta64[D]")).astype("datetime64[m]") + rng.randint(7 * 60, 22 * 60, len(sem)).astype("timedelta64[m]")
            ts_s = np.repeat(ini_s, nv) + rng.randint(0, 15, len(si)).astype("timedelta64[m]")        # uma sessão por cliente
            registrar(si, ts_s, 1, sk2)
            ab = rng.rand(len(si)) < 0.35
            registrar(si[ab], ts_s[ab] + np.timedelta64(3, "m"), 3, sk2[ab])
            registrar(si[ab], ts_s[ab] + np.timedelta64(40, "m"), 4, sk2[ab])
            novo_int_cli += [si[ab], si[ab]]; novo_int_sku += [sk2[ab], sk2[ab]]      # peso dobrado
        if nav["cli"]:
            nc, nt = np.concatenate(nav["cli"]), np.concatenate(nav["ts"])
            ntipo, nsku, ncat = np.concatenate(nav["tipo"]), np.concatenate(nav["sku"]), np.concatenate(nav["cat"])
            ordem = np.lexsort((nt, nc))
            plataforma = np.where(rng.rand(len(nc)) < 0.65, "aplicativo", "site")
            esc["navegacao"].escrever({"evento_id": prox_nav + np.arange(len(nc), dtype=np.int64), "cliente_id": nc[ordem],
                                       "datahora": nt[ordem].astype("datetime64[ms]"), "plataforma": plataforma,
                                       "tipo": np.array(NAV_TIPOS)[ntipo[ordem]],
                                       "sku": pa.array(nsku[ordem], mask=nsku[ordem] < 0),
                                       "categoria": pa.array(np.array(CATS + ["?"])[ncat[ordem]], mask=ncat[ordem] < 0)})
            prox_nav += len(nc)
        ic_, is_ = np.concatenate(novo_int_cli), np.concatenate(novo_int_sku)
        ordem = np.argsort(ic_, kind="stable")
        interesse_cli, interesse_sku = ic_[ordem].astype(np.int32), is_[ordem].astype(np.int32)

        # ---- fim do ciclo: incentivo, crédito e saída
        atingiu = inc & (pontos >= meta)
        falhas = np.where(inc, np.where(atingiu, 0, np.minimum(falhas + 1, 100)), falhas).astype(np.int8)
        metas_bimestre = (metas_bimestre + atingiu).astype(np.int8)
        if (c + 1) % 6 == 0:                                   # revisão de nível a cada 6 ciclos
            nivel = np.clip(nivel + np.where(metas_bimestre >= 4, 1, np.where(metas_bimestre <= 1, -1, 0)), 0, 3).astype(np.int8)
            metas_bimestre[:] = 0
        esc["incentivo"].escrever({"cliente_id": ii.astype(np.int32),
                                   "data": np.full(len(ii), DIA0 + np.timedelta64(d0 + dias_ciclo - 1, "D")).astype("datetime64[ms]"),
                                   "ciclo": np.full(len(ii), c, np.int16), "evento": np.full(len(ii), "fim_ciclo"),
                                   "nivel": nivel[ii], "meta_ciclo": meta[ii].round(2),
                                   "pontos_ciclo": pontos[ii].round(2), "atingiu_meta": atingiu[ii].astype(np.float64)})
        gasto_ciclo = pontos
        gasto_medio = np.where(ativo, np.where(ciclos_ativo == 0, gasto_ciclo, 0.8 * gasto_medio + 0.2 * gasto_ciclo), gasto_medio)
        fatura = np.bincount(ped_cli, weights=valor_ped * a_credito, minlength=n_cli)
        n_pedidos += np.bincount(ped_cli, minlength=n_cli)
        ciclos_ativo += ativo
        risco = cli["risco_saida"] * np.where(ciclos_ativo <= 2, 1.5, 1.0) * np.where(bloqueado, 3.0, 1.0) \
            * np.where(atraso > 15, 2.0, 1.0) * np.where(falhas >= 3, 1.5, 1.0)
        sai = ativo & (rng.rand(n_cli) < np.clip(risco, 0, 0.95))
        sai |= ativo & cli["so_um_pedido"] & (n_pedidos > 0)
        ciclo_saida[sai] = c
        ativo[sai] = False
        if c % 20 == 0 or c == N_CICLOS - 1:
            print(f"ciclo {c:3d}: {int(ativo.sum()):,} clientes ativos, {prox_pedido:,} pedidos, {esc['itens_pedido'].n:,} linhas, "
                  f"{prox_nav:,} eventos de navegação ({time.time() - t0:.0f}s)", flush=True)

    for e in esc.values():
        e.fechar()

    # ---- cadastro, catálogo e gabarito
    datas_cad = (DIA0 + (cli["ciclo_entrada"].astype(np.int64) * CICLO).astype("timedelta64[D]")).astype("datetime64[ms]")
    pq.write_table(pa.table({"cliente_id": cli["cliente_id"], "tipo_cliente": np.array(TIPOS_CLIENTE)[cli["tipo"]],
                             "uf": np.array(UFS)[cli["uf"]], "data_cadastro": datas_cad,
                             "canal_aquisicao": np.array(["indicacao", "digital", "representante", "evento"])[cli["canal_aquisicao"]],
                             "idade": cli["idade"], "usa_credito": cli["usa_credito"], "inscrito_incentivo": cli["incentivo"]}),
                   os.path.join(pasta, "clientes.parquet"))
    data_ciclo = lambda v: (DIA0 + (np.minimum(v.astype(np.int64) * CICLO, N_DIAS)).astype("timedelta64[D]")).astype("datetime64[ms]")  # noqa: E731
    pq.write_table(pa.table({"sku": cat["sku"], "nome": cat["nome"], "categoria": cat["categoria"], "tipo_produto": cat["tipo_produto"],
                             "familia": cat["familia"], "linha_id": cat["linha_id"].astype(np.int32), "marca": cat["marca"],
                             "faixa_preco": np.array(["popular", "intermediaria", "premium"])[cat["faixa"]],
                             "preco_lista": cat["preco_lista"], "data_lancamento": data_ciclo(cat["ciclo_lancamento"]),
                             "data_descontinuacao": pa.array(data_ciclo(cat["ciclo_fim"]), mask=cat["ciclo_fim"] > N_CICLOS)}),
                   os.path.join(pasta, "produtos.parquet"))
    pq.write_table(pa.table({"cliente_id": cli["cliente_id"], "segmento": np.array(SEGMENTOS)[cli["segmento"]],
                             "frequencia_base": cli["freq"], "linhas_base": cli["linhas"], "sens_desconto": cli["sens_desconto"],
                             "engajamento": cli["engajamento"], "risco_credito": cli["risco_credito"], "risco_saida": cli["risco_saida"],
                             "so_um_pedido": cli["so_um_pedido"], "grande_rede": cli["rede"],
                             "ciclo_entrada": cli["ciclo_entrada"], "ciclo_saida": ciclo_saida,
                             "n_pedidos": n_pedidos}),
                   os.path.join(pasta, "gabarito", "clientes_latentes.parquet"))
    pq.write_table(pa.table({"sku_a": cat["regra_a"], "sku_b": cat["regra_b"], "probabilidade": cat["regra_p"]}),
                   os.path.join(pasta, "gabarito", "regras_compra_conjunta.parquet"))

    lin, val = np.concatenate(hist_linhas), np.concatenate(hist_valor)
    np_cli = n_pedidos[n_pedidos > 0]
    q = lambda v, ps: {f"p{p}": float(np.percentile(v, p)) for p in ps} | {"max": float(v.max())}  # noqa: E731
    resumo = {"tamanho": a.tamanho, "seed": a.seed, "clientes": int(n_cli), "clientes_com_pedido": int(len(np_cli)), "skus": int(n_sku),
              "ciclos": N_CICLOS, "pedidos": int(prox_pedido), "linhas_de_pedido": int(esc["itens_pedido"].n),
              "eventos_navegacao": int(prox_nav), "linhas_credito": int(esc["credito"].n), "linhas_incentivo": int(esc["incentivo"].n),
              "regras_compra_conjunta": int(len(cat["regra_a"])),
              "linhas_por_pedido": q(lin, [50, 90, 99]), "valor_do_pedido": q(val, [5, 50, 90, 99]),
              "pedidos_por_cliente": q(np_cli, [50, 90, 99]), "clientes_com_1_pedido": float((np_cli == 1).mean()),
              "clientes_com_mil_pedidos_ou_mais": int((np_cli >= 1000).sum()), "pedidos_com_100_linhas_ou_mais": int((lin >= 100).sum()),
              "segundos": round(time.time() - t0, 1)}
    json.dump(resumo, open(os.path.join(pasta, "resumo.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(resumo, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
