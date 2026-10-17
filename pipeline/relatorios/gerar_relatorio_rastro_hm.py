"""Gera o relatório visual "rastro ponta a ponta" de um cliente da H&M, no
mesmo formato do relatório do cliente 860.703.096-50 da base sintética.

Entradas (runs/hm_clientes/relatorios/):
  rastro_<id>.json            etapas de dados (rastro_cliente_hm_dados.py)
  rastro_<id>_modelo.json     embeddings das âncoras e predições (extraído no Colab)
  rastro_<id>_percentis.json  scores e percentis no split de validação
  predicoes_<id>.json         previsões token a token do backbone (gerar_predicoes_entidade.py)

Uso: python3 pipeline/relatorios/gerar_relatorio_rastro_hm.py 836302
Gera runs/hm_clientes/relatorios/rastro_cliente_<id>.html
"""

import json
import sys
from datetime import date

import pandas as pd

R = "runs/hm_clientes/"


def carregar(ent):
    base = json.load(open(f"{R}relatorios/rastro_{ent}.json", encoding="utf-8"))
    modelo = json.load(open(f"{R}relatorios/rastro_{ent}_modelo.json", encoding="utf-8"))
    perc = json.load(open(f"{R}relatorios/rastro_{ent}_percentis.json", encoding="utf-8"))
    ntp = json.load(open(f"{R}relatorios/predicoes_{ent}.json", encoding="utf-8"))
    vocab = json.load(open(f"{R}vocabulario/vocab.json", encoding="utf-8"))
    info = json.load(open(f"{R}features_tradicionais/features_info.json", encoding="utf-8"))
    resumo_fusao = json.load(open(f"{R}fusao_h4_h9_resumo.json", encoding="utf-8"))
    gb = json.load(open(f"{R}gbdt/trad_lgbm/relatorio.json", encoding="utf-8"))
    return base, modelo, perc, ntp, vocab, info, resumo_fusao, gb


def nomes_categorias():
    df = pd.read_parquet(f"{R}eventos_canonicos.parquet", columns=["indice", "grupo_produto"])
    return {c: list(df[c].cat.categories) for c in ["indice", "grupo_produto"]}


def montar(ent):
    base, modelo, perc, ntp, vocab, info, resumo_fusao, gb = carregar(ent)
    cats = nomes_categorias()
    seq = base["tokens"]["seq_full"]

    # bloco de tokens de cada compra (posição do último token -> fatia)
    compras = base["compras"]
    pos = base["posicoes"]
    inicio = 1
    for c in compras:
        fim = pos[str(c["evento_id"])]
        c["tokens"] = seq[inicio:fim + 1]
        c["pos_ini"], c["pos_fim"] = inicio, fim
        inicio = fim + 1

    # cestas: data, gap até a próxima, rótulos
    datas_cesta = []
    for c in compras:
        if not datas_cesta or datas_cesta[-1] != c["data"]:
            datas_cesta.append(c["data"])
    cestas = base["cestas"]
    for i, ce in enumerate(cestas):
        ce["data"] = datas_cesta[i]
        ce["gap_proxima"] = (date.fromisoformat(datas_cesta[i + 1]) - date.fromisoformat(datas_cesta[i])).days if i + 1 < len(cestas) else None
        ce["itens"] = [c for c in compras if c["data"] == ce["data"]]

    # features tradicionais: nomes legíveis
    def legivel(nome):
        for campo in ("indice", "grupo_produto"):
            if nome.startswith(campo + "_") and "_share" in nome:
                cod = int(nome[len(campo) + 1:].split("_share")[0])
                return f"{'linha' if campo == 'indice' else 'grupo'} {cats[campo][cod]}{' (90d)' if nome.endswith('_90d') else ''}"
        return nome
    feats = {}
    for f in base["features"]:
        feats[str(f["evento_id"])] = {legivel(k): v for k, v in f.items()}

    # previsões finais nas âncoras de validação/teste
    fus_d = {str(r["evento_id"]): r for r in modelo["predicoes"].get("fusao_d_completo_s0/predicoes_val.parquet", [])}
    fus_e = {str(r["evento_id"]): r for r in modelo["predicoes"].get("fusao_e_cad_caus_s0/predicoes_val.parquet", [])}

    # acurácia do backbone por tipo de token
    def campo(tok):
        for p in ["GRUPO_PRODUTO", "TIPO_PRODUTO", "LINHA_VESTUARIO", "PRECO_BUCKET", "INDICE", "COR", "CANAL", "ARTIGO", "RECENCIA", "MES", "EVT", "EOS"]:
            if tok.startswith(p):
                return p
        return "OUTRO"
    por_campo = {}
    for split in ("validacao", "teste"):
        for r in ntp[split]["previsoes"]:
            k = campo(r["alvo_real"])
            a = por_campo.setdefault(k, {"validacao": [0, 0], "teste": [0, 0]})
            a[split][0] += int(r["acertou_top1"])
            a[split][1] += 1

    return {
        "ent": ent, "customer_id": base["customer_id"], "cadastro": base["cadastro"], "cortes": base["cortes"],
        "periodo": base["periodo"], "compras": compras, "cestas": cestas, "seq": seq,
        "len_train": base["tokens"]["len_train"], "len_train_val": base["tokens"]["len_train_val"],
        "vocab_size": base["vocab_size"], "n_positions": base["n_positions"],
        "vocab_ids": {t: vocab[t] for t in set(seq)},
        "features": feats, "n_features": info["n_features"],
        "embeddings": modelo["embeddings"], "ntp": ntp, "ntp_por_campo": por_campo,
        "perc": perc, "fus_d": fus_d, "fus_e": fus_e,
        "resumo_fusao": resumo_fusao,
        "gb_trad": {t: {k: v["media"] for k, v in x["val_resumo"].items()} for t, x in gb["tarefas"].items()},
    }




def main(ent):
    dados = montar(ent)
    with open("pipeline/relatorios/rastro_hm_template.html", encoding="utf-8") as f:
        tpl = f.read()
    html = tpl.replace("__DADOS__", json.dumps(dados, ensure_ascii=False, default=str))
    out = f"{R}relatorios/rastro_cliente_{ent}.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Gravado {out} ({len(html)/1024:.0f} KB)")


if __name__ == "__main__":
    main(sys.argv[1])
