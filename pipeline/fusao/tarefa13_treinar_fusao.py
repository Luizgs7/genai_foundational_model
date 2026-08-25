"""Tarefa 13 (Estágio 4/5, ARQUITETURA.md) — treina a fusão DCNv2 + cabeças
de tarefa (churn, próxima categoria, próximo valor) sobre o embedding
sequencial extraído do backbone congelado (tarefa13_extrair_embeddings.py).

Roda 100% em CPU — o backbone (caro, requer GPU) já foi congelado e usado
só pra gerar os embeddings de pipeline/fusao/embeddings_eventos.npy; esta
etapa só treina a DCNv2 + heads (poucos milhares de parâmetros) em cima
deles, ver ARQUITETURA.md, Estágio 4, opção 1 ("backbone congelado").

Entradas dinâmicas: embedding sequencial (128-d, backbone Tarefa 12).
Entradas estáticas: UF (embedding aprendido), sexo (binário), idade em
anos na data da transação-âncora (padronizada com média/desvio do TREINO).

Dimensionamento deliberadamente pequeno (ver Tarefa 12 / LIMITACOES.md,
item 4): a DCNv2 + heads tem ~7 mil parâmetros treináveis para ~73 mil
exemplos de treino — razão parâmetros/exemplos «1, ao contrário do
backbone original que tinha 16x mais parâmetros que tokens.

Observabilidade: pipeline/fusao/tarefa13_config.json,
tarefa13_metricas_epoca.csv, tarefa13_relatorio.json (com comparação
direta contra os baselines causais da Tarefa 8), tarefa13_loss_curve.png,
checkpoint tarefa13_fusao.pt.
"""

import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "pipeline/rotulos_downstream")
from tarefa8_rotulos import accuracy, auc_roc, f1_macro  # noqa: E402

EMB_NPY = "pipeline/fusao/embeddings_eventos.npy"
INDEX_CSV = "pipeline/fusao/embeddings_eventos_index.csv"
ROTULOS_CSV = "pipeline/rotulos_downstream/tarefa8_rotulos.csv"
BASE_CSV = "base_sintetica_embeddings_100k_v2.csv"
VOCAB_JSON = "pipeline/vocabulario/vocab.json"
BASELINE_JSON = "pipeline/rotulos_downstream/tarefa8_relatorio.json"
OUT_DIR = "pipeline/fusao"

EMB_DIM = 128
UF_EMB_DIM = 4
N_CROSS = 2
CROSS_RANK = 8
DEEP_HIDDEN = 16
N_CATEGORIAS = 4
N_VALOR_BUCKETS = 10

BATCH_SIZE = 256
N_EPOCHS = 40
LR = 2e-3
WEIGHT_DECAY = 1e-4
SEED = 0


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def carregar_dados():
    emb = np.load(EMB_NPY)
    index = pd.read_csv(INDEX_CSV)

    rotulos = pd.read_csv(ROTULOS_CSV, parse_dates=["data_compra"])
    df = index.merge(
        rotulos[["transacao_id", "data_compra", "churn_label", "proxima_categoria", "proximo_valor_bucket"]],
        on="transacao_id", how="left", validate="one_to_one",
    )

    estaticos = pd.read_csv(BASE_CSV, encoding="utf-8-sig", usecols=["cpf", "sexo", "data_nascimento", "uf"])
    estaticos["data_nascimento"] = pd.to_datetime(estaticos["data_nascimento"])
    estaticos = estaticos.drop_duplicates(subset="cpf")
    df = df.merge(estaticos, on="cpf", how="left", validate="many_to_one")

    df["idade"] = (df["data_compra"] - df["data_nascimento"]).dt.days / 365.25
    df["sexo_bin"] = (df["sexo"] == "F").astype(float)

    ufs = sorted(df["uf"].unique())
    uf_to_id = {u: i for i, u in enumerate(ufs)}
    df["uf_id"] = df["uf"].map(uf_to_id)

    vocab = json.load(open(VOCAB_JSON, encoding="utf-8"))
    categorias = sorted(k[len("CATEGORIA_"):] for k in vocab if k.startswith("CATEGORIA_"))
    cat_to_id = {c: i for i, c in enumerate(categorias)}
    df["proxima_categoria_id"] = df["proxima_categoria"].map(cat_to_id)

    # padroniza idade só com estatística do TREINO (mesmo princípio dos
    # buckets de quantil da Tarefa 4 — nunca calibrar com dado de val/teste)
    idade_media = df.loc[df["split"] == "train", "idade"].mean()
    idade_std = df.loc[df["split"] == "train", "idade"].std()
    df["idade_norm"] = (df["idade"] - idade_media) / idade_std

    return emb, df, uf_to_id, cat_to_id, {"idade_media": idade_media, "idade_std": idade_std}


class CrossLayerV2(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.v = nn.Linear(dim, rank, bias=False)
        self.u = nn.Linear(rank, dim, bias=True)

    def forward(self, x0, xl):
        return x0 * self.u(self.v(xl)) + xl


class DCNv2Fusao(nn.Module):
    def __init__(self, emb_dim, n_uf, uf_emb_dim, n_cross, rank, deep_hidden):
        super().__init__()
        self.uf_emb = nn.Embedding(n_uf, uf_emb_dim)
        x0_dim = emb_dim + uf_emb_dim + 1 + 1  # + sexo + idade
        self.x0_dim = x0_dim
        self.cross_layers = nn.ModuleList([CrossLayerV2(x0_dim, rank) for _ in range(n_cross)])
        self.deep = nn.Sequential(nn.Linear(x0_dim, deep_hidden), nn.ReLU())
        combined_dim = x0_dim + deep_hidden
        self.head_churn = nn.Linear(combined_dim, 1)
        self.head_categoria = nn.Linear(combined_dim, N_CATEGORIAS)
        self.head_valor = nn.Linear(combined_dim, N_VALOR_BUCKETS)

    def forward(self, emb_seq, uf_id, sexo, idade):
        uf_vec = self.uf_emb(uf_id)
        x0 = torch.cat([emb_seq, uf_vec, sexo.unsqueeze(-1), idade.unsqueeze(-1)], dim=-1)
        xl = x0
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        deep_out = self.deep(x0)
        combined = torch.cat([xl, deep_out], dim=-1)
        return self.head_churn(combined).squeeze(-1), self.head_categoria(combined), self.head_valor(combined)


def montar_tensores(emb, df, split):
    sub = df[df["split"] == split].reset_index(drop=True)
    emb_t = torch.tensor(emb[sub["row_idx"].values], dtype=torch.float32)
    uf_t = torch.tensor(sub["uf_id"].values, dtype=torch.long)
    sexo_t = torch.tensor(sub["sexo_bin"].values, dtype=torch.float32)
    idade_t = torch.tensor(sub["idade_norm"].values, dtype=torch.float32)
    churn_t = torch.tensor(sub["churn_label"].values, dtype=torch.float32)  # NaN preservado
    cat_t = torch.tensor(sub["proxima_categoria_id"].fillna(-1).values, dtype=torch.long)
    valor_raw = sub["proximo_valor_bucket"]
    valor_t = torch.tensor(valor_raw.fillna(-1).values, dtype=torch.long)
    return sub, emb_t, uf_t, sexo_t, idade_t, churn_t, cat_t, valor_t


def perda_multitarefa(out_churn, out_cat, out_valor, churn_t, cat_t, valor_t):
    mask_churn = ~torch.isnan(churn_t)
    loss_churn = (
        F.binary_cross_entropy_with_logits(out_churn[mask_churn], churn_t[mask_churn])
        if mask_churn.any() else torch.tensor(0.0)
    )
    mask_cat = cat_t >= 0
    loss_cat = F.cross_entropy(out_cat[mask_cat], cat_t[mask_cat]) if mask_cat.any() else torch.tensor(0.0)
    mask_valor = valor_t >= 0
    loss_valor = F.cross_entropy(out_valor[mask_valor], valor_t[mask_valor]) if mask_valor.any() else torch.tensor(0.0)
    total = loss_churn + loss_cat + loss_valor
    return total, loss_churn.item(), loss_cat.item(), loss_valor.item()


@torch.no_grad()
def avaliar(model, emb_t, uf_t, sexo_t, idade_t, churn_t, cat_t, valor_t):
    model.eval()
    out_churn, out_cat, out_valor = model(emb_t, uf_t, sexo_t, idade_t)
    model.train()

    p_churn = torch.sigmoid(out_churn).numpy()
    y_churn = churn_t.numpy()
    auc = auc_roc(pd.Series(y_churn), pd.Series(p_churn))

    pred_cat = out_cat.argmax(dim=-1).numpy()
    y_cat = cat_t.numpy()
    y_cat_nan = np.where(y_cat < 0, np.nan, y_cat)
    pred_cat_masked = np.where(y_cat < 0, np.nan, pred_cat)
    acc_cat = accuracy(pd.Series(y_cat_nan), pd.Series(pred_cat_masked))
    f1_cat = f1_macro(pd.Series(y_cat_nan), pd.Series(pred_cat_masked))

    pred_valor = out_valor.argmax(dim=-1).numpy()
    y_valor = valor_t.numpy()
    y_valor_nan = np.where(y_valor < 0, np.nan, y_valor)
    pred_valor_masked = np.where(y_valor < 0, np.nan, pred_valor)
    acc_valor = accuracy(pd.Series(y_valor_nan), pd.Series(pred_valor_masked))

    return {
        "churn_auc": auc,
        "categoria_accuracy": acc_cat,
        "categoria_f1_macro": f1_cat,
        "valor_accuracy": acc_valor,
    }


def main():
    set_seed(SEED)
    t0 = time.time()

    emb, df, uf_to_id, cat_to_id, idade_stats = carregar_dados()

    sub_train, emb_tr, uf_tr, sexo_tr, idade_tr, churn_tr, cat_tr, valor_tr = montar_tensores(emb, df, "train")
    sub_val, emb_val, uf_val, sexo_val, idade_val, churn_val, cat_val, valor_val = montar_tensores(emb, df, "val")
    sub_test, emb_test, uf_test, sexo_test, idade_test, churn_test, cat_test, valor_test = montar_tensores(emb, df, "test")

    n_train = len(sub_train)
    print(f"train={n_train}  val={len(sub_val)}  test={len(sub_test)}")

    model = DCNv2Fusao(EMB_DIM, len(uf_to_id), UF_EMB_DIM, N_CROSS, CROSS_RANK, DEEP_HIDDEN)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DCNv2 + heads: {n_params} parametros treinaveis (razao exemplos/parametro = {n_train/n_params:.2f})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "tarefa13_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "emb_dim": EMB_DIM, "uf_emb_dim": UF_EMB_DIM, "n_cross": N_CROSS, "cross_rank": CROSS_RANK,
            "deep_hidden": DEEP_HIDDEN, "batch_size": BATCH_SIZE, "n_epochs": N_EPOCHS, "lr": LR,
            "weight_decay": WEIGHT_DECAY, "seed": SEED, "n_params": n_params,
            "n_train": n_train, "n_val": len(sub_val), "n_test": len(sub_test),
            "uf_to_id": uf_to_id, "categoria_to_id": cat_to_id, "idade_stats": idade_stats,
            "backbone_congelado": True, "backbone_checkpoint": "pipeline/pretreino/checkpoints/melhor.pt",
        }, f, ensure_ascii=False, indent=2, default=float)

    epoca_csv = open(os.path.join(OUT_DIR, "tarefa13_metricas_epoca.csv"), "w", encoding="utf-8")
    epoca_csv.write("epoch,loss_treino,loss_churn,loss_categoria,loss_valor,val_churn_auc,val_categoria_acc,val_categoria_f1,val_valor_acc\n")

    melhor_auc_val = -1.0
    melhor_epoca = -1
    historico = []

    for epoca in range(1, N_EPOCHS + 1):
        ordem = np.random.permutation(n_train)
        perdas = []
        for i in range(0, n_train, BATCH_SIZE):
            idx = ordem[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            out_churn, out_cat, out_valor = model(emb_tr[idx], uf_tr[idx], sexo_tr[idx], idade_tr[idx])
            loss, lc, lcat, lval = perda_multitarefa(out_churn, out_cat, out_valor, churn_tr[idx], cat_tr[idx], valor_tr[idx])
            loss.backward()
            optimizer.step()
            perdas.append((loss.item(), lc, lcat, lval))

        loss_media, lc_media, lcat_media, lval_media = np.mean(perdas, axis=0)
        val_metrics = avaliar(model, emb_val, uf_val, sexo_val, idade_val, churn_val, cat_val, valor_val)

        epoca_csv.write(
            f"{epoca},{loss_media:.6f},{lc_media:.6f},{lcat_media:.6f},{lval_media:.6f},"
            f"{val_metrics['churn_auc']:.6f},{val_metrics['categoria_accuracy']:.6f},"
            f"{val_metrics['categoria_f1_macro']:.6f},{val_metrics['valor_accuracy']:.6f}\n"
        )
        epoca_csv.flush()
        historico.append({"epoch": epoca, "loss_treino": loss_media, **val_metrics})

        if val_metrics["churn_auc"] > melhor_auc_val:
            melhor_auc_val = val_metrics["churn_auc"]
            melhor_epoca = epoca
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "tarefa13_fusao.pt"))

        if epoca == 1 or epoca % 5 == 0 or epoca == N_EPOCHS:
            print(f"epoca {epoca:>2}/{N_EPOCHS}  loss={loss_media:.4f}  "
                  f"val_churn_auc={val_metrics['churn_auc']:.4f}  "
                  f"val_cat_acc={val_metrics['categoria_accuracy']:.4f}  "
                  f"val_valor_acc={val_metrics['valor_accuracy']:.4f}")

    epoca_csv.close()

    print(f"\nMelhor epoca (por AUC de churn em val): {melhor_epoca} (auc={melhor_auc_val:.4f})")
    model.load_state_dict(torch.load(os.path.join(OUT_DIR, "tarefa13_fusao.pt")))
    test_metrics = avaliar(model, emb_test, uf_test, sexo_test, idade_test, churn_test, cat_test, valor_test)
    print(f"Metricas de TESTE (checkpoint da melhor epoca, avaliacao unica): {test_metrics}")

    baseline = json.load(open(BASELINE_JSON, encoding="utf-8"))
    comparacao = {
        "churn_auc": {
            "modelo_teste": test_metrics["churn_auc"],
            "baseline_rfm_teste": baseline["churn"]["baseline"]["por_split"]["test"],
            "delta": test_metrics["churn_auc"] - baseline["churn"]["baseline"]["por_split"]["test"],
        },
        "categoria_accuracy": {
            "modelo_teste": test_metrics["categoria_accuracy"],
            "baseline_moda_teste": baseline["proxima_categoria"]["baseline"]["accuracy"]["test"],
            "delta": test_metrics["categoria_accuracy"] - baseline["proxima_categoria"]["baseline"]["accuracy"]["test"],
        },
        "categoria_f1_macro": {
            "modelo_teste": test_metrics["categoria_f1_macro"],
            "baseline_moda_teste": baseline["proxima_categoria"]["baseline"]["f1_macro"]["test"],
            "delta": test_metrics["categoria_f1_macro"] - baseline["proxima_categoria"]["baseline"]["f1_macro"]["test"],
        },
        "valor_accuracy": {
            "modelo_teste": test_metrics["valor_accuracy"],
            "baseline_moda_teste": baseline["proximo_valor_bucket"]["baseline"]["accuracy"]["test"],
            "delta": test_metrics["valor_accuracy"] - baseline["proximo_valor_bucket"]["baseline"]["accuracy"]["test"],
        },
    }
    print("\nComparacao contra baselines causais (Tarefa 8):")
    for k, v in comparacao.items():
        print(f"  {k}: modelo={v['modelo_teste']:.4f}  baseline={v[[c for c in v if c.startswith('baseline')][0]]:.4f}  delta={v['delta']:+.4f}")

    tempo_total = time.time() - t0
    relatorio = {
        "n_params": n_params,
        "n_train": n_train, "n_val": len(sub_val), "n_test": len(sub_test),
        "razao_exemplos_por_parametro_treino": n_train / n_params,
        "melhor_epoca": melhor_epoca,
        "melhor_churn_auc_val": melhor_auc_val,
        "metricas_teste": test_metrics,
        "comparacao_com_baseline_tarefa8": comparacao,
        "tempo_total_segundos": tempo_total,
    }
    with open(os.path.join(OUT_DIR, "tarefa13_relatorio.json"), "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in historico]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].plot(epochs, [h["loss_treino"] for h in historico], label="loss treino (soma 3 tarefas)")
        axes[0].axvline(melhor_epoca, color="gray", linestyle="--", alpha=0.6, label=f"melhor epoca ({melhor_epoca})")
        axes[0].set_xlabel("Epoca"); axes[0].set_ylabel("Loss"); axes[0].legend()
        axes[0].set_title("Loss de treino — DCNv2 + heads")

        axes[1].plot(epochs, [h["churn_auc"] for h in historico], label="AUC churn (val)")
        axes[1].plot(epochs, [h["categoria_accuracy"] for h in historico], label="accuracy categoria (val)")
        axes[1].plot(epochs, [h["valor_accuracy"] for h in historico], label="accuracy valor (val)")
        axes[1].axhline(baseline["churn"]["baseline"]["por_split"]["val"], color="C0", linestyle=":", alpha=0.7)
        axes[1].axhline(baseline["proxima_categoria"]["baseline"]["accuracy"]["val"], color="C1", linestyle=":", alpha=0.7)
        axes[1].axvline(melhor_epoca, color="gray", linestyle="--", alpha=0.6)
        axes[1].set_xlabel("Epoca"); axes[1].set_ylabel("Metrica (val)"); axes[1].legend()
        axes[1].set_title("Metricas de validacao vs. baselines (linha pontilhada)")

        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "tarefa13_loss_curve.png"), dpi=150)
        print(f"Gravado {OUT_DIR}/tarefa13_loss_curve.png")
    except Exception as e:
        print(f"Aviso: nao consegui gerar o grafico ({e})")

    print(f"\nGravado {OUT_DIR}/tarefa13_config.json, tarefa13_metricas_epoca.csv, "
          f"tarefa13_relatorio.json, tarefa13_fusao.pt")
    print(f"Tempo total: {tempo_total:.1f}s")


if __name__ == "__main__":
    main()
