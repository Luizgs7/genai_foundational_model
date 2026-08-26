"""Motor genérico — Pré-treino do backbone (Tarefa 11/12 generalizada).

Versão schema-agnóstica de `pipeline/pretreino/tarefa_pretreino.py` (branch
`luiz_g`) — a lógica de treino/avaliação já era 100% agnóstica a schema
(opera só sobre ids de token); a única mudança aqui é ler hiperparâmetros e
caminhos do config em vez de constantes fixas pra base sintética.

Requer GPU (attn_implementation=flash_attention_2). Roda no Colab.

Uso: python3 pipeline/common/tarefa_pretreino.py <config.yaml>

Gera runs/<nome>/pretreino/{config.json, metricas_passo.csv,
metricas_epoca.csv, relatorio.json, loss_curve.png, checkpoints/{melhor,final}.pt}.
"""

import csv
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import GPT2Config, GPT2LMHeadModel

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402

BATCH_SIZE = 32
N_EPOCHS = 30
LR = 3e-4
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
SEED = 0
LOG_A_CADA_N_PASSOS = 20


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def carregar_dados(config):
    with open(f"{run_dir(config, 'vocabulario')}/vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    pad_id = vocab["PAD"]
    with open(f"{run_dir(config, 'serializacao')}/sequencias.json", encoding="utf-8") as f:
        seqs = json.load(f)

    train_seqs, val_exemplos, test_exemplos = [], [], []
    n_test_frios = 0

    for _entidade_id, s in seqs.items():
        st, stv, sf = s["seq_train"], s["seq_train_val"], s["seq_full"]
        if st:
            train_seqs.append(st)

        boundary_train = (len(st) - 1) if st else 0
        if stv and len(stv) > (len(st) if st else 1):
            val_exemplos.append((stv, boundary_train))

        boundary_train_val = (len(stv) - 1) if stv else boundary_train
        if len(sf) > (len(stv) if stv else (len(st) if st else 1)):
            test_exemplos.append((sf, boundary_train_val))
            if not st:
                n_test_frios += 1

    return vocab, pad_id, train_seqs, val_exemplos, test_exemplos, n_test_frios


def rotular_alvo(seq, boundary, n_positions):
    input_ids = seq[:-1]
    labels = seq[1:]
    L = len(seq)
    mask = [boundary <= (i + 1) <= L - 2 for i in range(L - 1)]
    if len(input_ids) > n_positions:
        input_ids = input_ids[-n_positions:]
        labels = labels[-n_positions:]
        mask = mask[-n_positions:]
    return input_ids, labels, mask


def montar_batch_treino(seqs, pad_id, n_positions, rng):
    escolhidos = [seqs[i] for i in rng]
    truncados = [s[-n_positions:] if len(s) > n_positions else s for s in escolhidos]
    max_len = max(len(s) for s in truncados)
    B = len(truncados)
    input_ids = np.full((B, max_len), pad_id, dtype=np.int64)
    attention_mask = np.zeros((B, max_len), dtype=np.int64)
    labels = np.full((B, max_len), -100, dtype=np.int64)
    for i, s in enumerate(truncados):
        L = len(s)
        input_ids[i, :L] = s
        attention_mask[i, :L] = 1
        # labels é cópia alinhada de input_ids (não pré-deslocada) — o
        # GPT2LMHeadModel desloca internamente. Ver ROADMAP.md, Tarefa 11,
        # pro bug real que isso corrige (duplo deslocamento).
        labels[i, :L] = s
        labels[i, L - 1] = -100
    return input_ids, attention_mask, labels


def montar_batch_avaliacao(exemplos, pad_id, n_positions):
    processados = [rotular_alvo(seq, boundary, n_positions) for seq, boundary in exemplos]
    max_len = max(len(p[0]) for p in processados)
    B = len(processados)
    input_ids = np.full((B, max_len), pad_id, dtype=np.int64)
    attention_mask = np.zeros((B, max_len), dtype=np.int64)
    labels = np.full((B, max_len), -100, dtype=np.int64)
    mascara_alvo = np.zeros((B, max_len), dtype=bool)
    for i, (ids, labs, msk) in enumerate(processados):
        n = len(ids)
        input_ids[i, :n] = ids
        attention_mask[i, :n] = 1
        labels[i, :n] = labs
        mascara_alvo[i, :n] = msk
    return input_ids, attention_mask, labels, mascara_alvo


def loss_mascarada(logits, labels, mascara_alvo, device):
    B, T, V = logits.shape
    logits_flat = logits.reshape(-1, V)
    labels_flat = torch.from_numpy(labels).to(device).reshape(-1)
    mascara_flat = torch.from_numpy(mascara_alvo).to(device).reshape(-1)
    labels_seguro = torch.where(mascara_flat, labels_flat, torch.zeros_like(labels_flat))
    perdas = F.cross_entropy(logits_flat.float(), labels_seguro, reduction="none")
    perdas = perdas * mascara_flat
    n_alvos = mascara_flat.sum().clamp(min=1)
    return perdas.sum() / n_alvos, int(n_alvos.item())


@torch.no_grad()
def avaliar(model, exemplos, pad_id, n_positions, device, batch_size=32):
    model.eval()
    perda_total, n_total = 0.0, 0
    for i in range(0, len(exemplos), batch_size):
        lote = exemplos[i : i + batch_size]
        input_ids, attention_mask, labels, mascara_alvo = montar_batch_avaliacao(lote, pad_id, n_positions)
        out = model(
            input_ids=torch.from_numpy(input_ids).to(device),
            attention_mask=torch.from_numpy(attention_mask).to(device),
        )
        perda, n_alvos = loss_mascarada(out.logits, labels, mascara_alvo, device)
        perda_total += perda.item() * n_alvos
        n_total += n_alvos
    model.train()
    if n_total == 0:
        return float("nan"), float("nan"), 0
    media = perda_total / n_total
    return media, math.exp(media), n_total


def lr_lambda(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progresso = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(progresso, 1.0)))


def main(config_path):
    config = carregar_config(config_path)
    arch = config.get("arquitetura", {})
    n_embd, n_layer, n_head = arch.get("n_embd", 128), arch.get("n_layer", 4), arch.get("n_head", 4)
    n_positions = arch.get("n_positions", 512)
    attn_implementation = arch.get("attn_implementation", "flash_attention_2")

    set_seed(SEED)
    out_dir = run_dir(config, "pretreino")
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    device = "cuda"

    vocab, pad_id, train_seqs, val_exemplos, test_exemplos, n_test_frios = carregar_dados(config)
    vocab_size = len(vocab)
    total_tokens_treino = sum(len(s) - 1 for s in train_seqs)

    print(f"vocab_size={vocab_size}  entidades_treino={len(train_seqs)}  tokens_treino={total_tokens_treino}")
    print(f"entidades_val_avaliaveis={len(val_exemplos)}  entidades_teste_avaliaveis={len(test_exemplos)} "
          f"(dos quais {n_test_frios} sem nenhum histórico de treino)")

    gpt2_config = GPT2Config(
        vocab_size=vocab_size, n_positions=n_positions, n_embd=n_embd, n_layer=n_layer, n_head=n_head,
        bos_token_id=vocab["BOS"], eos_token_id=vocab["EOS"], attn_implementation=attn_implementation,
    )
    model = GPT2LMHeadModel(gpt2_config).to(dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        model.transformer.wpe.weight.zero_()
    model.transformer.wpe.weight.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modelo: {n_params/1e6:.2f}M parâmetros ({n_params_trainable/1e6:.2f}M treináveis, NoPE congelado)")
    print(f"Razão tokens_treino/parametros_treináveis: {total_tokens_treino/n_params_trainable:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    passos_por_epoca = math.ceil(len(train_seqs) / BATCH_SIZE)
    total_steps = passos_por_epoca * N_EPOCHS
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: lr_lambda(s, WARMUP_STEPS, total_steps))

    with open(os.path.join(out_dir, "pretreino_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "vocab_size": vocab_size, "n_positions": n_positions, "n_embd": n_embd, "n_layer": n_layer,
            "n_head": n_head, "batch_size": BATCH_SIZE, "n_epochs": N_EPOCHS, "lr": LR,
            "n_params": n_params, "n_params_trainable": n_params_trainable,
            "n_entidades_treino": len(train_seqs), "total_tokens_treino": total_tokens_treino,
            "n_entidades_val": len(val_exemplos), "n_entidades_teste": len(test_exemplos),
        }, f, ensure_ascii=False, indent=2)

    passo_csv = open(os.path.join(out_dir, "pretreino_metricas_passo.csv"), "w", newline="", encoding="utf-8")
    passo_writer = csv.writer(passo_csv)
    passo_writer.writerow(["step", "epoch", "loss_treino", "lr", "grad_norm", "tokens_por_seg", "gpu_mem_gb", "elapsed_s"])

    epoca_csv = open(os.path.join(out_dir, "pretreino_metricas_epoca.csv"), "w", newline="", encoding="utf-8")
    epoca_writer = csv.writer(epoca_csv)
    epoca_writer.writerow(["epoch", "loss_treino_media", "loss_val", "perplexidade_val", "n_tokens_val", "elapsed_s"])

    melhor_loss_val, melhor_epoca = float("inf"), -1
    historico_epocas = []
    t0_global = time.time()
    step = 0

    for epoca in range(1, N_EPOCHS + 1):
        ordem = np.random.permutation(len(train_seqs))
        perdas_epoca = []
        t0_epoca = time.time()
        for i in range(0, len(ordem), BATCH_SIZE):
            lote_idx = ordem[i : i + BATCH_SIZE]
            input_ids, attention_mask, labels = montar_batch_treino(train_seqs, pad_id, n_positions, lote_idx)
            t0_passo = time.time()

            optimizer.zero_grad()
            out = model(
                input_ids=torch.from_numpy(input_ids).to(device),
                attention_mask=torch.from_numpy(attention_mask).to(device),
                labels=torch.from_numpy(labels).to(device),
            )
            out.loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            dt = time.time() - t0_passo
            n_tokens_passo = int(attention_mask.sum())
            perdas_epoca.append(out.loss.item())
            step += 1

            if step % LOG_A_CADA_N_PASSOS == 0 or step == 1:
                mem_gb = torch.cuda.max_memory_allocated() / 1e9
                passo_writer.writerow([
                    step, epoca, f"{out.loss.item():.6f}", f"{scheduler.get_last_lr()[0]:.6e}",
                    f"{grad_norm.item():.4f}", f"{n_tokens_passo/max(dt,1e-6):.1f}", f"{mem_gb:.3f}",
                    f"{time.time()-t0_global:.1f}",
                ])
                passo_csv.flush()

        loss_treino_media = float(np.mean(perdas_epoca))
        loss_val, ppl_val, n_tokens_val = avaliar(model, val_exemplos, pad_id, n_positions, device)
        elapsed_epoca = time.time() - t0_epoca

        epoca_writer.writerow([epoca, f"{loss_treino_media:.6f}", f"{loss_val:.6f}", f"{ppl_val:.4f}", n_tokens_val, f"{elapsed_epoca:.1f}"])
        epoca_csv.flush()
        historico_epocas.append({"epoch": epoca, "loss_treino": loss_treino_media, "loss_val": loss_val,
                                  "perplexidade_val": ppl_val, "elapsed_s": elapsed_epoca})
        print(f"epoca {epoca:>3}/{N_EPOCHS}  loss_treino={loss_treino_media:.4f}  loss_val={loss_val:.4f}  "
              f"ppl_val={ppl_val:.3f}  ({elapsed_epoca:.1f}s)")

        torch.save(model.state_dict(), os.path.join(ckpt_dir, "final.pt"))
        if loss_val < melhor_loss_val:
            melhor_loss_val, melhor_epoca = loss_val, epoca
            torch.save(model.state_dict(), os.path.join(ckpt_dir, "melhor.pt"))

    passo_csv.close()
    epoca_csv.close()

    print(f"\nMelhor época (por loss de val): {melhor_epoca} (loss_val={melhor_loss_val:.4f})")
    model.load_state_dict(torch.load(os.path.join(ckpt_dir, "melhor.pt")))
    loss_teste, ppl_teste, n_tokens_teste = avaliar(model, test_exemplos, pad_id, n_positions, device)
    print(f"Loss de TESTE (melhor checkpoint, avaliação única): {loss_teste:.4f}  ppl={ppl_teste:.3f}")

    tempo_total = time.time() - t0_global
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9

    relatorio = {
        "n_params": n_params, "n_params_trainable": n_params_trainable,
        "total_tokens_treino": total_tokens_treino,
        "razao_tokens_por_parametro_treinavel": total_tokens_treino / n_params_trainable,
        "n_epochs_executadas": N_EPOCHS, "melhor_epoca": melhor_epoca, "melhor_loss_val": melhor_loss_val,
        "melhor_perplexidade_val": math.exp(melhor_loss_val),
        "loss_val_ultima_epoca": historico_epocas[-1]["loss_val"],
        "loss_treino_ultima_epoca": historico_epocas[-1]["loss_treino"],
        "loss_teste": loss_teste, "perplexidade_teste": ppl_teste, "n_tokens_teste_avaliados": n_tokens_teste,
        "n_entidades_teste_frias_incluidas": n_test_frios, "tempo_total_segundos": tempo_total,
        "peak_gpu_mem_gb": peak_mem_gb, "gpu": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
    }
    with open(os.path.join(out_dir, "pretreino_relatorio.json"), "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in historico_epocas]
        plt.figure(figsize=(9, 5))
        plt.plot(epochs, [h["loss_treino"] for h in historico_epocas], label="loss treino")
        plt.plot(epochs, [h["loss_val"] for h in historico_epocas], label="loss validação")
        plt.axvline(melhor_epoca, color="gray", linestyle="--", alpha=0.6, label=f"melhor época ({melhor_epoca})")
        plt.xlabel("Época"); plt.ylabel("Loss (cross-entropy, NTP)")
        plt.title(f"Pré-treino do backbone — {config['nome']}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pretreino_loss_curve.png"), dpi=150)
        print(f"Gravado {out_dir}/pretreino_loss_curve.png")
    except Exception as e:
        print(f"Aviso: não consegui gerar o gráfico ({e})")

    print(f"\nGravado {out_dir}/{{pretreino_config.json, pretreino_metricas_*.csv, pretreino_relatorio.json}}")
    print(f"Gravado {ckpt_dir}/{{melhor.pt, final.pt}}")
    print(f"Tempo total: {tempo_total/60:.1f} min | pico de VRAM: {peak_mem_gb:.2f}GB")


if __name__ == "__main__":
    main(sys.argv[1])
