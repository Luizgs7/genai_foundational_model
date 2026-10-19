"""Consolida os resultados do experimento H&M (H3–H10) num único JSON —
fonte das tabelas do EXPERIMENTOS_HM.md e do relatório final.

Lê só os relatórios .json pequenos (versionados) de runs/hm_clientes/ e grava
runs/hm_clientes/resultados_consolidados.json com:
  - pretreino: H3 (referência) e backbone final (3 sementes)
  - ablacoes: H8 (14 ablações, 1 época, 1 semente)
  - validacao: média ± desvio por modelo e tarefa (H4–H9)
  - teste: idem, só para os modelos congelados da H10 (avaliação única)
  - bootstrap_teste: deltas com IC 95% por cliente (h10_bootstrap.json)

Uso: python3 pipeline/relatorios/consolidar_hm.py
"""

import glob
import json
import os
import statistics as st

R = "runs/hm_clientes/"
TAREFAS = ["churn", "proxima_online", "next_category", "ltv_90d"]
METRICA = {"churn": "auc", "proxima_online": "auc", "next_category": "accuracy", "ltv_90d": "accuracy"}


def ler(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def resumo(valores):
    valores = [v for v in valores if v is not None]
    if not valores:
        return None
    return {"media": st.mean(valores), "desvio": st.pstdev(valores) if len(valores) > 1 else 0.0, "n": len(valores)}


def gbdt(saida):
    r = ler(f"{R}gbdt/{saida}/relatorio.json")
    if not r:
        return None
    out = {"val": {}, "test": {}}
    for t in TAREFAS:
        x = r["tarefas"].get(t)
        if not x:
            continue
        m = METRICA[t]
        out["val"][t] = resumo([s["val"][m] for s in x["por_semente"]])
        out["test"][t] = resumo([s["test"][m] for s in x["por_semente"] if "test" in s])
        out.setdefault("participacao_embedding", {})[t] = x.get("participacao_importancia_embedding")
    return out


def fusao(padrao):
    rels = [ler(p) for p in sorted(glob.glob(f"{R}{padrao}/tarefa13_relatorio.json"))]
    rels = [r for r in rels if r]
    if not rels:
        return None
    out = {"val": {}, "test": {}}
    for t in TAREFAS:
        m = METRICA[t]
        out["val"][t] = resumo([r["metricas_validacao"][t][m] for r in rels if t in r.get("metricas_validacao", {})])
        out["test"][t] = resumo([r["metricas_teste"][t][m] for r in rels if t in (r.get("metricas_teste") or {})])
    return out


def pretreino(caminho):
    p = ler(caminho)
    if not p:
        return None
    return {k: p.get(k) for k in ["n_params_trainable", "n_epochs_executadas", "melhor_epoca", "loss_val_completa",
                                  "perplexidade_val_completa", "loss_teste", "perplexidade_teste",
                                  "tempo_total_segundos", "tokens_por_segundo_medio", "peak_gpu_mem_gb"]}


def main():
    rot = ler(f"{R}rotulos_downstream/relatorio.json")
    base = {"val": {}, "test": {}}
    for t in TAREFAS:
        x = rot[t]
        chave = "baseline_auc_por_split" if METRICA[t] == "auc" else "baseline_accuracy_por_split"
        base["val"][t] = {"media": x[chave]["val"], "desvio": 0.0, "n": 1}
        base["test"][t] = {"media": x[chave]["test"], "desvio": 0.0, "n": 1}

    modelos = {
        "baseline_causal": base,
        # H5 — régua tradicional
        "gbdt_trad_lgbm": gbdt("trad_lgbm"), "gbdt_trad_xgb": gbdt("trad_xgb"),
        "gbdt_trad_lgbm_orcamento_h7": gbdt("trad_lgbm_orcamento_h7"),
        # H6 — só embedding
        "gbdt_emb_pca_lgbm": gbdt("emb_pca_lgbm"), "gbdt_emb_bruto_lgbm": gbdt("emb_bruto_lgbm"),
        "gbdt_final_emb_pca_lgbm": gbdt("final_emb_pca_lgbm"),
        # H7 — híbrido
        "gbdt_hibrido_lgbm": gbdt("hibrido_lgbm"), "gbdt_hibrido_xgb": gbdt("hibrido_xgb"),
        "gbdt_final_hibrido_lgbm": gbdt("final_hibrido_lgbm"),
        "gbdt_hibrido_lgbm_sem_cadastro": gbdt("hibrido_lgbm_sem_cadastro"),
        # H4/H9 — fusão (validação)
        "fusao_d_completo": fusao("fusao_d_completo_s*"), "fusao_a_emb": fusao("fusao_a_emb_s*"),
        "fusao_b_emb_cad": fusao("fusao_b_emb_cad_s*"), "fusao_c_emb_caus": fusao("fusao_c_emb_caus_s*"),
        "fusao_e_cad_caus": fusao("fusao_e_cad_caus_s*"),
        "fusao_final": fusao("final_s*/fusao"),
        # H10 — teste (configurações congeladas)
        "h10_gbdt_trad_lgbm": gbdt("h10_trad_lgbm"), "h10_gbdt_trad_xgb": gbdt("h10_trad_xgb"),
        "h10_gbdt_emb_pca_lgbm": gbdt("h10_emb_pca_lgbm"), "h10_gbdt_hibrido_lgbm": gbdt("h10_hibrido_lgbm"),
        "h10_gbdt_hibrido_xgb": gbdt("h10_hibrido_xgb"), "h10_gbdt_final_hibrido_lgbm": gbdt("h10_final_hibrido_lgbm"),
        "h10_gbdt_hibrido_lgbm_sem_cadastro": gbdt("h10_hibrido_lgbm_sem_cadastro"),
        "h10_fusao_d_completo": fusao("h10_fusao_d_completo_s*"), "h10_fusao_c_emb_caus": fusao("h10_fusao_c_emb_caus_s*"),
        "h10_fusao_e_cad_caus": fusao("h10_fusao_e_cad_caus_s*"), "h10_fusao_final": fusao("h10_final_fusao_s*"),
    }
    ablacoes = {}
    transcritos = ler(f"{R}ablacoes/resultados_transcritos.json")["ablacoes"]
    for nome, x in transcritos.items():
        ablacoes[nome] = {"params_M": x["params_M"], "loss_val": x["loss_val"], "tempo_s": x.get("tempo_s"),
                          "gbdt_churn": x["gbdt"]["churn_auc"], "gbdt_next_category": x["gbdt"]["next_category_acc"],
                          **{f"fusao_{k}": v for k, v in x["fusao"].items()}, "fonte": "transcrito"}
    for p in sorted(glob.glob(f"{R}ablacoes/*.json")):
        nome = os.path.basename(p)[:-5]
        if nome == "resultados_transcritos" or nome in ablacoes:
            continue
        x = ler(p)
        ablacoes[nome] = {"params_M": x["n_params_treinaveis"] / 1e6, "loss_val": x["loss_val_completa"],
                          "tempo_s": x["tempo_pretreino_s"],
                          "gbdt_churn": x["sonda_gbdt_val"]["churn"]["auc"]["media"],
                          "gbdt_next_category": x["sonda_gbdt_val"]["next_category"]["accuracy"]["media"],
                          **{f"fusao_{k}": v["modelo"] for k, v in x["sonda_fusao_val"].items()}, "fonte": "json"}

    out = {
        "pretreino": {"h3_referencia": pretreino(f"{R}pretreino/pretreino_relatorio.json"),
                      **{f"final_s{s}": pretreino(f"{R}final_s{s}/pretreino/pretreino_relatorio.json") for s in range(3)}},
        "ablacoes": ablacoes,
        "modelos": {k: v for k, v in modelos.items() if v},
        "bootstrap_teste": ler(f"{R}h10_bootstrap.json"),
        "bootstrap_validacao_h9": ler(f"{R}h9_bootstrap.json"),
    }
    with open(f"{R}resultados_consolidados.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    def fmt(x):
        return "   —   " if not x else f"{x['media']:.4f}"
    for split in ("val", "test"):
        print(f"\n## {split}")
        print(f"{'modelo':38s} " + " ".join(f"{t:>14s}" for t in TAREFAS))
        for k, v in out["modelos"].items():
            if any(v[split].get(t) for t in TAREFAS):
                print(f"{k:38s} " + " ".join(f"{fmt(v[split].get(t)):>14s}" for t in TAREFAS))


if __name__ == "__main__":
    main()
