"""Motor v2 (evento composto) — leitura do esquema no YAML. Desenho em MOTOR_V2.md.

O config declara, na seção `v2:`, o catálogo de itens e uma lista de fontes.
Cada fonte tem uma natureza:

  itens    evento formado por várias linhas (pedido e seus itens)
  evento   evento simples, agrupado em sessões (navegação)
  estado   valor que vale até mudar (crédito, nível de incentivo): não ocupa
           posição; o valor vigente é anexado a cada posição

Exemplo completo: configs/beleza_b2b_mini.yaml. Para um config do v1 basta um
atalho, que vira uma única fonte `itens` agrupada por entidade e dia:

    v2:
      item: artigo
      atributos_item: [indice, grupo_produto, tipo_produto, cor, linha_vestuario]
      numericos_linha: [preco]
      cabecalho_categoricos: [canal]
      valor_linha: [preco]
"""

PADRAO_JANELA = 512
JANELAS_DIAS = [30, 90, 180]
# faixas fixas dos alvos (limites superiores, inclusive; a última faixa é aberta)
FAIXAS_INTERVALO_DIAS = [1, 2, 3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 180, 365]
FAIXAS_N_LINHAS = [1, 2, 3, 5, 8, 12, 20, 35, 60, 100, 200]
FAIXAS_CONTAGEM = [0, 1, 2, 3, 5, 9]


def esquema(config):
    v2 = dict(config.get("v2") or {})
    if not v2:
        raise ValueError("o config não tem a seção v2 (ver pipeline/common/v2_esquema.py)")
    if "fontes" not in v2:      # atalho para um config do v1: uma fonte `itens` por entidade e dia
        fonte = {"nome": "compra", "natureza": "itens", "alvo": True, "tabela": config["fonte_canonica"],
                 "entidade": "entidade_id", "data": "data_evento", "agrupar_por": ["entidade_id", "_dia"],
                 "item": v2["item"], "numericos_linha": v2.get("numericos_linha", []),
                 "categoricos_linha": v2.get("categoricos_linha", []), "valor_linha": v2.get("valor_linha", []),
                 "id_linha": "evento_id",
                 "cabecalho": {"derivado": True, "categoricos": v2.get("cabecalho_categoricos", []), "numericos": []}}
        v2 = {"janela": v2.get("janela", PADRAO_JANELA), "fontes": [fonte],
              "catalogo": {"derivado_de": "compra", "item": v2["item"], "categoricos": v2.get("atributos_item", []), "numericos": []}}
    v2.setdefault("janela", PADRAO_JANELA)
    v2.setdefault("janelas_dias", JANELAS_DIAS)
    alvo = [f for f in v2["fontes"] if f.get("alvo")]
    if len(alvo) != 1 or alvo[0]["natureza"] != "itens":
        raise ValueError("exatamente uma fonte de natureza `itens` deve ter alvo: true")
    for f in v2["fontes"]:
        f.setdefault("numericos_linha", []); f.setdefault("categoricos_linha", []); f.setdefault("valor_linha", [])
        f.setdefault("numericos", []); f.setdefault("categoricos", []); f.setdefault("item", None)
        f.setdefault("sessao_minutos", 30); f.setdefault("id_linha", None); f.setdefault("filtro", None)
        if f["natureza"] not in ("itens", "evento", "estado"):
            raise ValueError(f"natureza desconhecida na fonte {f['nome']}: {f['natureza']}")
    v2["fonte_alvo"] = alvo[0]["nome"]
    return v2
