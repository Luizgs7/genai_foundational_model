"""Gera as previsões da fusão DCNv2 (uma por tarefa downstream ativa) pra
uma entidade específica — insumo do relatório visual, equivalente da fusão
ao que gerar_predicoes_entidade.py já faz pro NTP do backbone. Reaproveita
o pré-processamento e a arquitetura de tarefa13_treinar_fusao.py (mesmas
estatísticas de padronização, recalculadas do split de treino — o script
de treino não persiste um "scaler" em disco, mas o cálculo é determinístico
a partir dos mesmos dados) e só carrega os pesos já treinados, sem re-treinar.

Uso: python3 pipeline/common/gerar_predicoes_fusao_entidade.py <config.yaml> <entidade_id>
"""

import json
import os
import sys

import pandas as pd
import torch

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir, tarefas_ativas  # noqa: E402
from tarefa13_treinar_fusao import (  # noqa: E402
    CROSS_RANK, DEEP_HIDDEN, N_CROSS, DCNv2Fusao, carregar_dados, inferir_tipo_tarefa,
)


def preparar_features(df, cat_estaticos, num_estaticos):
    cardinalidades = []
    for campo in cat_estaticos:
        valores_treino = sorted(df.loc[df["split"] == "train", campo].dropna().unique(), key=str)
        mapa = {v: i for i, v in enumerate(valores_treino)}
        df[f"{campo}_id"] = df[campo].map(mapa).fillna(len(mapa)).astype(int)
        cardinalidades.append(len(mapa))

    for campo in num_estaticos:
        media = df.loc[df["split"] == "train", campo].mean()
        desvio = df.loc[df["split"] == "train", campo].std() or 1.0
        df[f"{campo}_norm"] = (df[campo] - media) / desvio

    return cardinalidades


def main(config_path, entidade_id):
    config = carregar_config(config_path)
    emb, df, cat_estaticos, num_estaticos = carregar_dados(config)
    cardinalidades = preparar_features(df, cat_estaticos, num_estaticos)

    tarefas_info = []
    for tarefa_cfg in tarefas_ativas(config):
        nome = tarefa_cfg.get("nome", tarefa_cfg["tipo"])
        tipo, mapa_classes = inferir_tipo_tarefa(df[f"{nome}_rotulo"])
        n_classes = len(mapa_classes) if mapa_classes else 1
        tarefas_info.append((nome, tipo, n_classes, mapa_classes))
    tarefas_modelo = [(nome, tipo, n_classes) for nome, tipo, n_classes, _ in tarefas_info]

    model = DCNv2Fusao(emb.shape[1], cardinalidades, len(num_estaticos), N_CROSS, CROSS_RANK, DEEP_HIDDEN, tarefas_modelo)
    fusao_dir = run_dir(config, "fusao")
    model.load_state_dict(torch.load(os.path.join(fusao_dir, "tarefa13_fusao.pt"), map_location="cpu"))
    model.eval()

    sub = df[df["entidade_id"] == entidade_id].sort_values("data_evento").reset_index(drop=True)
    if len(sub) == 0:
        raise ValueError(f"entidade {entidade_id!r} não encontrada nos dados de fusão")

    emb_t = torch.tensor(emb[sub["row_idx"].values], dtype=torch.float32)
    cat_ids = [torch.tensor(sub[f"{c}_id"].values, dtype=torch.long) for c in cat_estaticos]
    num_vals = (
        torch.tensor(sub[[f"{c}_norm" for c in num_estaticos]].values, dtype=torch.float32)
        if num_estaticos else None
    )

    with torch.no_grad():
        saidas = model(emb_t, cat_ids, num_vals)

    # pos_weight (ver tarefa13_treinar_fusao.py) melhora o ranking (AUC) das
    # tarefas binárias às custas da calibração -- o sigmoid deixa de ser uma
    # probabilidade de verdade. Pra não exibir um número enganoso no
    # relatório, calcula também o percentil de cada score dentro da
    # distribuição do split de teste inteiro (robusto a essa descalibração,
    # é literalmente o que a AUC mede).
    sub_teste = df[df["split"] == "test"].reset_index(drop=True)
    emb_teste = torch.tensor(emb[sub_teste["row_idx"].values], dtype=torch.float32)
    cat_ids_teste = [torch.tensor(sub_teste[f"{c}_id"].values, dtype=torch.long) for c in cat_estaticos]
    num_vals_teste = (
        torch.tensor(sub_teste[[f"{c}_norm" for c in num_estaticos]].values, dtype=torch.float32)
        if num_estaticos else None
    )
    with torch.no_grad():
        saidas_teste = model(emb_teste, cat_ids_teste, num_vals_teste)
    scores_teste = {
        nome: torch.sigmoid(saidas_teste[nome].squeeze(-1)).numpy()
        for nome, tipo, _n in tarefas_modelo if tipo == "binaria"
    }

    registros = []
    for i in range(len(sub)):
        registro = {
            "evento_id": int(sub.loc[i, "evento_id"]),
            "data_evento": str(sub.loc[i, "data_evento"].date()),
            "split": sub.loc[i, "split"],
        }
        for nome, tipo, _n, mapa_classes in tarefas_info:
            rotulo = sub.loc[i, f"{nome}_rotulo"]
            if tipo == "binaria":
                p = torch.sigmoid(saidas[nome][i]).item()
                percentil = float((scores_teste[nome] <= p).mean() * 100)
                registro[nome] = {
                    "p_modelo": round(p, 4), "percentil_teste": round(percentil, 1),
                    "rotulo_real": None if pd.isna(rotulo) else float(rotulo),
                }
            else:
                probs = torch.softmax(saidas[nome][i], dim=-1)
                pred_idx = int(probs.argmax().item())
                classes_inv = {v: k for k, v in mapa_classes.items()}
                registro[nome] = {
                    "classe_prevista": classes_inv[pred_idx],
                    "prob_classe_prevista": round(probs[pred_idx].item(), 4),
                    "rotulo_real": None if pd.isna(rotulo) else rotulo,
                }
        registros.append(registro)
        print(registro)

    out_dir = run_dir(config, "relatorios")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"predicoes_fusao_{entidade_id[:16]}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"entidade_id": entidade_id, "previsoes": registros}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nGravado {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
