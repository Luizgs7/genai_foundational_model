"""Motor genérico — Treino da fusão DCNv2 + cabeças de tarefa (Tarefa 13 generalizada).

Versão schema-agnóstica de `pipeline/fusao/tarefa13_treinar_fusao.py`
(branch `luiz_g`). Mudança estrutural: o número e tipo de cabeças de saída
é construído **dinamicamente** a partir de `tarefas_downstream` do config
— não mais 3 cabeças fixas (churn/categoria/valor). Cada tarefa ativa vira
uma cabeça binária (AUC-ROC) ou multiclasse (accuracy/F1 macro), inferido
automaticamente da forma do rótulo em `rotulos.csv` (Tarefa 8 generalizada).

Roda 100% em CPU — o backbone já foi congelado e usado só pra gerar os
embeddings (Tarefa 13a); esta etapa só treina a DCNv2 + heads (poucos
milhares de parâmetros) em cima deles.

Ablações (H9 do experimento H&M): `--sem_embedding`, `--sem_cadastro`
(campos estáticos da entidade, incl. idade) e `--sem_causais` (estatísticas
causais/baselines expostas como feature) removem cada grupo de entrada.
`--seed` e `--saida` permitem repetir com várias sementes sem sobrescrever.

Uso: python3 pipeline/common/tarefa13_treinar_fusao.py <config.yaml>
         [--emb <subpasta dos embeddings>] [--saida <subpasta>] [--seed N]
         [--sem_embedding] [--sem_cadastro] [--sem_causais] [--avaliar_teste]

O teste só é avaliado com `--avaliar_teste` (protocolo do experimento H&M:
escolhas na validação, teste uma única vez). Sem a flag, o relatório traz
só métricas de validação. Os runs legados (synthetic_agro/olist) foram
gerados no comportamento antigo, equivalente a `--avaliar_teste`.

Gera runs/<nome>/<saida>/{tarefa13_config.json, tarefa13_metricas_epoca.csv,
tarefa13_relatorio.json, tarefa13_loss_curve.png, tarefa13_fusao.pt}.
"""

import argparse
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

sys.path.insert(0, "pipeline/common")
from schema import caminho_tabela, carregar_config, ler_fonte, ler_tabela, run_dir, tarefas_ativas  # noqa: E402
from tarefa8_rotulos import accuracy, auc_roc, f1_macro  # noqa: E402

EMB_DIM_PADRAO = 128
CAT_ESTATICO_EMB_DIM = 4
N_CROSS = 2
CROSS_RANK = 8
DEEP_HIDDEN = 16

BATCH_SIZE = 256
N_EPOCHS = 40
LR = 2e-3
WEIGHT_DECAY = 1e-4
SEED = 0
DESCONHECIDO = "__DESCONHECIDO__"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def inferir_tipo_tarefa(serie_rotulo):
    valores = serie_rotulo.dropna()
    if set(valores.unique()) <= {0.0, 1.0}:
        return "binaria", None
    classes = sorted(valores.unique(), key=str)
    return "multiclasse", {c: i for i, c in enumerate(classes)}


class CrossLayerV2(nn.Module):
    def __init__(self, dim, rank):
        super().__init__()
        self.v = nn.Linear(dim, rank, bias=False)
        self.u = nn.Linear(rank, dim, bias=True)

    def forward(self, x0, xl):
        return x0 * self.u(self.v(xl)) + xl


class DCNv2Fusao(nn.Module):
    def __init__(self, emb_dim, cardinalidades_categoricas, n_numericos, n_cross, rank, deep_hidden, tarefas):
        super().__init__()
        self.embs_categoricos = nn.ModuleList([nn.Embedding(card + 1, CAT_ESTATICO_EMB_DIM) for card in cardinalidades_categoricas])
        x0_dim = emb_dim + len(cardinalidades_categoricas) * CAT_ESTATICO_EMB_DIM + n_numericos
        self.cross_layers = nn.ModuleList([CrossLayerV2(x0_dim, rank) for _ in range(n_cross)])
        self.deep = nn.Sequential(nn.Linear(x0_dim, deep_hidden), nn.ReLU())
        combined_dim = x0_dim + deep_hidden
        self.heads = nn.ModuleDict()
        for nome, tipo, n_classes in tarefas:
            self.heads[nome] = nn.Linear(combined_dim, 1 if tipo == "binaria" else n_classes)

    def forward(self, emb_seq, cat_ids_list, num_vals):
        partes = [emb_seq]
        for emb_layer, ids in zip(self.embs_categoricos, cat_ids_list):
            partes.append(emb_layer(ids))
        if num_vals is not None:
            partes.append(num_vals)
        x0 = torch.cat(partes, dim=-1)
        xl = x0
        for layer in self.cross_layers:
            xl = layer(x0, xl)
        deep_out = self.deep(x0)
        combined = torch.cat([xl, deep_out], dim=-1)
        return {nome: head(combined) for nome, head in self.heads.items()}


def carregar_dados(config, emb_subpasta="fusao", sem_cadastro=False, sem_causais=False):
    emb = np.load(os.path.join(run_dir(config, emb_subpasta), "embeddings_eventos.npy"), mmap_mode="r")
    index = ler_tabela(caminho_tabela(config, emb_subpasta, "embeddings_eventos_index"))

    rotulos = ler_tabela(caminho_tabela(config, "rotulos_downstream", "rotulos"))

    df = index.merge(rotulos, on=["evento_id", "entidade_id"], how="inner", validate="one_to_one")

    cat_estaticos = [] if sem_cadastro else list(config.get("campos_estaticos_categoricos", []))
    num_estaticos = [] if sem_cadastro else list(config.get("campos_estaticos_numericos", []))
    campo_nascimento = None if sem_cadastro else config.get("campo_data_nascimento")

    colunas_canonicas = {"evento_id", "entidade_id", "data_evento"} | set(cat_estaticos) | set(num_estaticos)
    if campo_nascimento:
        colunas_canonicas.add(campo_nascimento)
    canonico = ler_fonte(config, sorted(colunas_canonicas))
    canonico["data_evento"] = pd.to_datetime(canonico["data_evento"])
    if campo_nascimento:
        canonico[campo_nascimento] = pd.to_datetime(canonico[campo_nascimento])

    df = df.merge(canonico[["evento_id", "data_evento"]], on="evento_id", how="left", validate="one_to_one")

    estaticos_por_entidade = canonico.drop_duplicates(subset="entidade_id").set_index("entidade_id")
    for campo in cat_estaticos:
        df[campo] = df["entidade_id"].map(estaticos_por_entidade[campo])
    for campo in num_estaticos:
        df[campo] = df["entidade_id"].map(estaticos_por_entidade[campo])
    if campo_nascimento:
        nascimento = df["entidade_id"].map(estaticos_por_entidade[campo_nascimento])
        df["_idade"] = (df["data_evento"] - nascimento).dt.days / 365.25
        num_estaticos = num_estaticos + ["_idade"]

    # Features causais genéricas: expõe pro DCNv2, como número, o mesmo
    # material que os baselines já usam (ex. média histórica causal do gap
    # entre eventos) — sem isso a cabeça só recebe o embedding congelado e
    # precisa redescobrir indiretamente um sinal que já está calculado em
    # tarefa8_rotulos.py. Só entram colunas numéricas (baseline categórico/
    # moda de texto, como next_category, fica de fora).
    num_causais = [c for c in ["n_eventos_anteriores"] if c in df.columns and not sem_causais]
    for tarefa_cfg in ([] if sem_causais else tarefas_ativas(config)):
        nome = tarefa_cfg.get("nome", tarefa_cfg["tipo"])
        col = f"{nome}_baseline"
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            num_causais.append(col)
    num_estaticos = num_estaticos + num_causais

    return emb, df, cat_estaticos, num_estaticos


def main(config_path, emb_subpasta="fusao", saida="fusao", seed=SEED, sem_embedding=False,
         sem_cadastro=False, sem_causais=False, avaliar_teste=False):
    set_seed(seed)
    t0 = time.time()
    config = carregar_config(config_path)
    emb, df, cat_estaticos, num_estaticos = carregar_dados(config, emb_subpasta, sem_cadastro, sem_causais)
    emb_dim = 0 if sem_embedding else emb.shape[1]

    # --- codificação dos campos estáticos categóricos (id calibrado no treino) ---
    mapas_categoricos = {}
    cardinalidades = []
    for campo in cat_estaticos:
        valores_treino = sorted(df.loc[df["split"] == "train", campo].dropna().unique(), key=str)
        mapa = {v: i for i, v in enumerate(valores_treino)}
        mapas_categoricos[campo] = mapa
        df[f"{campo}_id"] = df[campo].map(mapa).fillna(len(mapa)).astype(int)
        cardinalidades.append(len(mapa))

    # --- padronização dos campos estáticos numéricos (média/desvio do treino) ---
    stats_numericos = {}
    for campo in num_estaticos:
        media = df.loc[df["split"] == "train", campo].mean()
        desvio = df.loc[df["split"] == "train", campo].std() or 1.0
        stats_numericos[campo] = {"media": float(media), "desvio": float(desvio)}
        df[f"{campo}_norm"] = (df[campo] - media) / desvio

    # --- inferir tipo/classes de cada tarefa ativa a partir do rótulo real ---
    tarefas_info = []
    for tarefa_cfg in tarefas_ativas(config):
        nome = tarefa_cfg.get("nome", tarefa_cfg["tipo"])
        tipo, mapa_classes = inferir_tipo_tarefa(df[f"{nome}_rotulo"])
        n_classes = len(mapa_classes) if mapa_classes else 1
        tarefas_info.append((nome, tipo, n_classes, mapa_classes))
        if mapa_classes:
            df[f"{nome}_rotulo_id"] = df[f"{nome}_rotulo"].map(mapa_classes)
            df[f"{nome}_baseline_id"] = df[f"{nome}_baseline"].map(mapa_classes)

    tarefas_modelo = [(nome, tipo, n_classes) for nome, tipo, n_classes, _ in tarefas_info]

    def montar_tensores(split):
        sub = df[df["split"] == split].reset_index(drop=True)
        emb_t = (torch.zeros((len(sub), 0), dtype=torch.float32) if sem_embedding
                 else torch.tensor(np.asarray(emb[sub["row_idx"].values]),
                                   dtype=torch.float32))
        cat_ids = [torch.tensor(sub[f"{c}_id"].values, dtype=torch.long) for c in cat_estaticos]
        num_vals = (
            torch.tensor(sub[[f"{c}_norm" for c in num_estaticos]].values, dtype=torch.float32)
            if num_estaticos else None
        )
        alvos = {}
        for nome, tipo, _n_classes, mapa_classes in tarefas_info:
            col = f"{nome}_rotulo_id" if mapa_classes else f"{nome}_rotulo"
            dtype = torch.long if mapa_classes else torch.float32
            fill = -1 if mapa_classes else float("nan")
            alvos[nome] = torch.tensor(sub[col].fillna(fill).values, dtype=dtype)
        return sub, emb_t, cat_ids, num_vals, alvos

    sub_train, emb_tr, cat_tr, num_tr, alvos_tr = montar_tensores("train")
    sub_val, emb_val, cat_val, num_val, alvos_val = montar_tensores("val")
    sub_test, emb_test, cat_test, num_test, alvos_test = montar_tensores("test")
    n_train = len(sub_train)
    print(f"train={n_train}  val={len(sub_val)}  test={len(sub_test)}")
    print(f"tarefas ativas: {[t[0] for t in tarefas_modelo]}")

    # pos_weight por tarefa binária (razão neg/pos do TREINO) -- sem isso a
    # BCE trata um rótulo raro (ex. churn a ~5%) igual a um balanceado, e o
    # gradiente é dominado pelos negativos.
    pos_weights = {}
    for nome, tipo, _n in tarefas_modelo:
        if tipo != "binaria":
            continue
        y_validos = alvos_tr[nome][~torch.isnan(alvos_tr[nome])]
        n_pos = (y_validos == 1).sum().item()
        n_neg = (y_validos == 0).sum().item()
        pos_weights[nome] = torch.tensor(n_neg / max(n_pos, 1), dtype=torch.float32)
        print(f"  pos_weight[{nome}] = {pos_weights[nome].item():.2f}  (n_pos={n_pos} n_neg={n_neg})")

    model = DCNv2Fusao(emb_dim, cardinalidades, len(num_estaticos), N_CROSS, CROSS_RANK, DEEP_HIDDEN, tarefas_modelo)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DCNv2 + heads: {n_params} parâmetros treináveis (razão exemplos/parâmetro = {n_train/n_params:.2f})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def perda_multitarefa(saidas, alvos, idx):
        total = 0.0
        partes = {}
        for nome, tipo, _n in tarefas_modelo:
            y = alvos[nome][idx]
            if tipo == "binaria":
                mask = ~torch.isnan(y)
                l = F.binary_cross_entropy_with_logits(
                    saidas[nome].squeeze(-1)[mask], y[mask], pos_weight=pos_weights[nome]
                ) if mask.any() else torch.tensor(0.0)
            else:
                mask = y >= 0
                l = F.cross_entropy(saidas[nome][mask], y[mask]) if mask.any() else torch.tensor(0.0)
            partes[nome] = l.item()
            total = total + l
        return total, partes

    @torch.no_grad()
    def avaliar(emb_t, cat_ids, num_vals, alvos):
        model.eval()
        saidas = model(emb_t, cat_ids, num_vals)
        model.train()
        resultado = {}
        for nome, tipo, _n in tarefas_modelo:
            y = alvos[nome].numpy()
            if tipo == "binaria":
                p = torch.sigmoid(saidas[nome].squeeze(-1)).numpy()
                resultado[nome] = {"auc": auc_roc(pd.Series(y), pd.Series(p))}
            else:
                pred = saidas[nome].argmax(dim=-1).numpy()
                y_nan = np.where(y < 0, np.nan, y)
                pred_nan = np.where(y < 0, np.nan, pred)
                resultado[nome] = {
                    "accuracy": accuracy(pd.Series(y_nan), pd.Series(pred_nan)),
                    "f1_macro": f1_macro(pd.Series(y_nan), pd.Series(pred_nan)),
                }
        return resultado

    out_dir = run_dir(config, saida)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "tarefa13_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "emb_subpasta": emb_subpasta, "emb_dim": emb_dim, "sem_embedding": sem_embedding,
            "sem_cadastro": sem_cadastro, "sem_causais": sem_causais,
            "n_params": n_params, "n_train": n_train, "n_val": len(sub_val), "n_test": len(sub_test),
            "tarefas": [{"nome": n, "tipo": t, "n_classes": c} for n, t, c in tarefas_modelo],
            "campos_estaticos_categoricos": cat_estaticos, "campos_estaticos_numericos": num_estaticos,
            "pos_weights": {k: v.item() for k, v in pos_weights.items()},
            "batch_size": BATCH_SIZE, "n_epochs": N_EPOCHS, "lr": LR, "seed": seed,
        }, f, ensure_ascii=False, indent=2, default=str)

    epoca_csv = open(os.path.join(out_dir, "tarefa13_metricas_epoca.csv"), "w", encoding="utf-8")
    colunas_metricas = []
    for nome, tipo, _n in tarefas_modelo:
        colunas_metricas += [f"{nome}_auc"] if tipo == "binaria" else [f"{nome}_acc", f"{nome}_f1"]
    epoca_csv.write("epoch,loss_treino," + ",".join(colunas_metricas) + "\n")

    metrica_chave = tarefas_modelo[0][0]  # 1ª tarefa ativa decide o "melhor checkpoint" (convenção: churn primeiro)
    melhor_score, melhor_epoca = -1.0, -1
    historico = []

    for epoca in range(1, N_EPOCHS + 1):
        ordem = np.random.permutation(n_train)
        perdas = []
        for i in range(0, n_train, BATCH_SIZE):
            idx = ordem[i:i + BATCH_SIZE]
            cat_batch = [c[idx] for c in cat_tr]
            num_batch = num_tr[idx] if num_tr is not None else None
            optimizer.zero_grad()
            saidas = model(emb_tr[idx], cat_batch, num_batch)
            loss, _partes = perda_multitarefa(saidas, alvos_tr, idx)
            loss.backward()
            optimizer.step()
            perdas.append(loss.item())

        loss_media = float(np.mean(perdas))
        val_metrics = avaliar(emb_val, cat_val, num_val, alvos_val)
        linha_csv = [str(epoca), f"{loss_media:.6f}"]
        for nome, tipo, _n in tarefas_modelo:
            if tipo == "binaria":
                linha_csv.append(f"{val_metrics[nome]['auc']:.6f}")
            else:
                linha_csv += [f"{val_metrics[nome]['accuracy']:.6f}", f"{val_metrics[nome]['f1_macro']:.6f}"]
        epoca_csv.write(",".join(linha_csv) + "\n")
        epoca_csv.flush()
        historico.append({"epoch": epoca, "loss_treino": loss_media, "val_metrics": val_metrics})

        score_chave = val_metrics[metrica_chave].get("auc", val_metrics[metrica_chave].get("accuracy"))
        if score_chave is not None and not math.isnan(score_chave) and score_chave > melhor_score:
            melhor_score, melhor_epoca = score_chave, epoca
            torch.save(model.state_dict(), os.path.join(out_dir, "tarefa13_fusao.pt"))

        if epoca == 1 or epoca % 5 == 0 or epoca == N_EPOCHS:
            print(f"epoca {epoca:>2}/{N_EPOCHS}  loss={loss_media:.4f}  "
                  f"{metrica_chave}={score_chave:.4f}" if score_chave is not None else "")

    epoca_csv.close()
    print(f"\nMelhor época (por {metrica_chave} em val): {melhor_epoca} (score={melhor_score:.4f})")
    model.load_state_dict(torch.load(os.path.join(out_dir, "tarefa13_fusao.pt")))

    @torch.no_grad()
    def scores(emb_t, cat_ids, num_vals):
        model.eval()
        saidas = model(emb_t, cat_ids, num_vals)
        out = {}
        for nome, tipo, _n in tarefas_modelo:
            out[nome] = (torch.sigmoid(saidas[nome].squeeze(-1)).numpy() if tipo == "binaria"
                         else torch.softmax(saidas[nome], dim=-1).numpy())
        return out

    def comparar(sub_s, metricas):
        comparacao = {}
        for nome, tipo, _n, _mapa in tarefas_info:
            if tipo == "binaria":
                base = auc_roc(sub_s[f"{nome}_rotulo"], sub_s[f"{nome}_baseline"])
                comparacao[nome] = {"metrica": "auc", "modelo": metricas[nome]["auc"], "baseline": base,
                                    "delta": metricas[nome]["auc"] - base}
            else:
                base = accuracy(sub_s[f"{nome}_rotulo_id"], sub_s[f"{nome}_baseline_id"])
                comparacao[nome] = {"metrica": "accuracy", "modelo": metricas[nome]["accuracy"], "baseline": base,
                                    "delta": metricas[nome]["accuracy"] - base}
        return comparacao

    def gravar_predicoes(sub_s, sc, split):
        cols = {"evento_id": sub_s["evento_id"].values, "entidade_id": sub_s["entidade_id"].values}
        for nome, tipo, _n, mapa in tarefas_info:
            if tipo == "binaria":
                cols[f"{nome}_score"] = sc[nome]
            else:
                inv = {i: c for c, i in mapa.items()}
                cols[f"{nome}_pred"] = [inv[i] for i in sc[nome].argmax(axis=1)]
                for c, i in mapa.items():
                    cols[f"{nome}_p_{c}"] = sc[nome][:, i]
        pd.DataFrame(cols).to_parquet(os.path.join(out_dir, f"predicoes_{split}.parquet"), index=False)

    val_metrics = avaliar(emb_val, cat_val, num_val, alvos_val)
    comparacao_val = comparar(sub_val, val_metrics)
    gravar_predicoes(sub_val, scores(emb_val, cat_val, num_val), "val")
    print("Validação (melhor checkpoint):")
    for nome, c in comparacao_val.items():
        print(f"  {nome}: modelo={c['modelo']:.4f}  baseline={c['baseline']:.4f}  delta={c['delta']:+.4f}")

    test_metrics, comparacao = None, None
    if avaliar_teste:
        test_metrics = avaliar(emb_test, cat_test, num_test, alvos_test)
        comparacao = comparar(sub_test, test_metrics)
        gravar_predicoes(sub_test, scores(emb_test, cat_test, num_test), "test")
        print("TESTE (avaliação única):")
        for nome, c in comparacao.items():
            print(f"  {nome}: modelo={c['modelo']:.4f}  baseline={c['baseline']:.4f}  delta={c['delta']:+.4f}")

    tempo_total = time.time() - t0
    relatorio = {
        "n_params": n_params, "n_train": n_train, "n_val": len(sub_val), "n_test": len(sub_test),
        "melhor_epoca": melhor_epoca, "metricas_validacao": val_metrics,
        "comparacao_com_baseline_validacao": comparacao_val,
        "metricas_teste": test_metrics, "comparacao_com_baseline": comparacao,
        "tempo_total_segundos": tempo_total,
    }
    with open(os.path.join(out_dir, "tarefa13_relatorio.json"), "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2, default=str)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in historico]
        plt.figure(figsize=(9, 5))
        plt.plot(epochs, [h["loss_treino"] for h in historico], label="loss treino (soma das tarefas)")
        plt.axvline(melhor_epoca, color="gray", linestyle="--", alpha=0.6, label=f"melhor época ({melhor_epoca})")
        plt.xlabel("Época"); plt.ylabel("Loss"); plt.legend()
        plt.title(f"Fusão DCNv2 — {config['nome']}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "tarefa13_loss_curve.png"), dpi=150)
    except Exception as e:
        print(f"Aviso: não consegui gerar o gráfico ({e})")

    print(f"\nGravado {out_dir}/{{tarefa13_config.json, tarefa13_metricas_epoca.csv, tarefa13_relatorio.json, tarefa13_fusao.pt}}")
    print(f"Tempo total: {tempo_total:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--emb", default="fusao", help="subpasta com embeddings_eventos.npy")
    ap.add_argument("--saida", default="fusao")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--sem_embedding", action="store_true")
    ap.add_argument("--sem_cadastro", action="store_true")
    ap.add_argument("--sem_causais", action="store_true")
    ap.add_argument("--avaliar_teste", action="store_true")
    a = ap.parse_args()
    main(a.config, a.emb, a.saida, a.seed, a.sem_embedding, a.sem_cadastro, a.sem_causais, a.avaliar_teste)
