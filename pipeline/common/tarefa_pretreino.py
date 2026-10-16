"""Motor genérico — Pré-treino do backbone (Tarefa 11/12 generalizada).

Objetivo de treino: prever o próximo token (NTP) sobre `seq_train` de cada
entidade. Avaliação temporal correta: a loss de validação é medida só nos
tokens do período de validação (contexto = histórico de treino + val), a
de teste só nos tokens do período de teste, uma única vez, no melhor
checkpoint por validação.

Escala (experimento H&M): lê as sequências via memmap (`sequencias_io.py`),
monta lotes por orçamento de tokens com sequências de tamanho parecido
(`pretreino.tokens_por_lote`), treina com pesos em fp32 + autocast bf16,
grava checkpoint periódico com estado do otimizador e retoma com
`--retomar` (a sessão do Colab cai). Hiperparâmetros vêm da seção
`pretreino:` do YAML (padrões em `schema.config_pretreino`, iguais às
constantes dos runs anteriores). Arquitetura via `modelo_decoder.py`
(`arquitetura.tipo`: gpt2_hf | decoder).

Detalhe herdado da Tarefa 11 (bug real corrigido lá): o alvo da posição t é
o token t+1 — o deslocamento é feito UMA vez, aqui, ao calcular a loss. O
EOS nunca é alvo (marca posição, não é evento a prever).

Uso: python3 pipeline/common/tarefa_pretreino.py <config.yaml> [--retomar] [--saida <subpasta>]

Gera runs/<nome>/<saida>/{pretreino_config.json, pretreino_metricas_passo.csv,
pretreino_metricas_epoca.csv, pretreino_relatorio.json, pretreino_loss_curve.png,
checkpoints/{melhor,final,ultimo}.pt}  (saida padrão: pretreino).
"""

import argparse
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

sys.path.insert(0, "pipeline/common")
from modelo_decoder import arquitetura_efetiva, construir_modelo  # noqa: E402
from schema import carregar_config, config_pretreino, run_dir  # noqa: E402
from sequencias_io import carregar_sequencias  # noqa: E402


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------- exemplos

def janelas_de_treino(seqs, n_positions, modo):
    """Lista de janelas de treino (idx_entidade, inicio, fim) dentro de
    seq_train. `ultima` = legado (só os últimos n_positions tokens);
    `todas` = a sequência inteira em pedaços de n_positions a partir do fim."""
    ent, ini, fim = [], [], []
    for i in range(len(seqs)):
        lt = seqs.len_train[i]
        if lt == 0:
            continue
        L = lt + 2  # BOS + blocos + EOS
        if modo == "ultima" or L <= n_positions:
            ent.append(i); ini.append(max(0, L - n_positions)); fim.append(L)
        else:
            f = L
            while f > 0:
                ent.append(i); ini.append(max(0, f - n_positions)); fim.append(f)
                f -= n_positions
    return np.array(ent), np.array(ini), np.array(fim)


def exemplos_de_avaliacao(seqs, split):
    """(idx_entidade, boundary) — mesma regra do legado: val usa seq_train_val
    com fronteira no fim do treino; teste usa seq_full com fronteira no fim
    de treino+val (entidades 'frias', sem treino, entram também)."""
    ex, n_frios = [], 0
    for i in range(len(seqs)):
        lt, ltv, lf = seqs.len_train[i], seqs.len_train_val[i], seqs.len_full[i]
        len_st = lt + 2 if lt else 0
        len_stv = ltv + 2 if ltv else 0
        boundary_train = (len_st - 1) if lt else 0
        if split == "val":
            if ltv and len_stv > (len_st if lt else 1):
                ex.append((i, boundary_train))
        else:
            boundary_tv = (len_stv - 1) if ltv else boundary_train
            if lf > (len_stv if ltv else (len_st if lt else 1)):
                ex.append((i, boundary_tv))
                n_frios += int(lt == 0)
    return ex, n_frios


def seq_avaliacao(seqs, i, split):
    return seqs.train_val(i) if split == "val" else seqs.full(i)


# ---------------------------------------------------------------- lotes

def plano_de_lotes(comprimentos, cfg, rng):
    """Ordem dos lotes de uma época. Sem `tokens_por_lote`: legado (lotes de
    `batch_size` na ordem de uma permutação global). Com orçamento: embaralha,
    ordena por tamanho dentro de blocos de ~100 lotes, corta por orçamento
    (n_seqs * maior_tamanho <= tokens_por_lote) e embaralha a ordem dos lotes."""
    n = len(comprimentos)
    if not cfg["tokens_por_lote"]:
        ordem = np.random.permutation(n)  # RNG global — mesma sequência do legado
        return [ordem[i:i + cfg["batch_size"]] for i in range(0, n, cfg["batch_size"])]
    orcamento = cfg["tokens_por_lote"]
    ordem = rng.permutation(n)
    media = max(1, int(np.mean(comprimentos)))
    bloco = max(1, 100 * orcamento // media)
    lotes = []
    for b in range(0, n, bloco):
        idx = ordem[b:b + bloco]
        idx = idx[np.argsort(comprimentos[idx], kind="stable")]
        atual, maior = [], 0
        for j in idx:
            novo_maior = max(maior, comprimentos[j])
            if atual and (len(atual) + 1) * novo_maior > orcamento:
                lotes.append(np.array(atual))
                atual, novo_maior = [], comprimentos[j]
            atual.append(j)
            maior = novo_maior
        if atual:
            lotes.append(np.array(atual))
    return [lotes[k] for k in rng.permutation(len(lotes))]


def montar_lote_treino(seqs, ent, ini, fim, idx, pad_id):
    janelas = []
    for k in idx:
        st = seqs.train(ent[k])
        janelas.append(st[ini[k]:fim[k]])
    T = max(len(s) for s in janelas)
    input_ids = np.full((len(janelas), T), pad_id, dtype=np.int64)
    alvo_valido = np.zeros((len(janelas), T), dtype=bool)
    for r, s in enumerate(janelas):
        input_ids[r, :len(s)] = s
        alvo_valido[r, 1:len(s) - 1] = True   # alvos: tokens 1..L-2 (nunca o 1º, nunca o EOS)
    return input_ids, alvo_valido


def montar_lote_avaliacao(seqs, exemplos, split, n_positions, pad_id):
    """Mesma regra do legado `rotular_alvo`: entrada = seq[:-1], alvo de i =
    seq[i+1], conta só onde boundary <= i+1 <= L-2; trunca pros últimos
    n_positions. Aqui no formato (input_ids, alvo_valido) com alvo implícito
    = input deslocado: guardamos a seq até L-1 e marcamos o alvo na posição
    do PRÓPRIO token alvo (i+1)."""
    seqs_l, mascaras = [], []
    for i, boundary in exemplos:
        s = seq_avaliacao(seqs, i, split)
        L = len(s)
        pos = np.arange(L)
        alvo = (pos >= max(boundary, 1)) & (pos <= L - 2)
        if L - 1 > n_positions:           # entrada (L-1 tokens) maior que a janela
            corte = (L - 1) - n_positions
            s, alvo = s[corte:], alvo[corte:]
            alvo[0] = False               # o 1º token da janela não tem contexto -> nunca é alvo
        seqs_l.append(s[:-1] if len(s) > 1 else s)
        mascaras.append(alvo[:len(s) - 1] if len(s) > 1 else alvo)
    # o alvo de posição p (p>=1) é o token p; a loss usa logits de p-1
    T = max(len(s) for s in seqs_l)
    input_ids = np.full((len(seqs_l), T), pad_id, dtype=np.int64)
    alvo_valido = np.zeros((len(seqs_l), T), dtype=bool)
    for r, (s, m) in enumerate(zip(seqs_l, mascaras)):
        input_ids[r, :len(s)] = s
        alvo_valido[r, :len(m)] = m
    # o último token de entrada (= seq[L-2]) também é alvo potencial do logit L-3;
    # o alvo seq[L-1] (EOS) nunca entra. Para incluir o alvo seq[L-2] precisamos
    # dele na entrada — já está (entrada = seq[:-1]).
    return input_ids, alvo_valido


def perda_ntp(logits, input_ids, alvo_valido):
    """CE do logit da posição p-1 contra o token p, nas posições p marcadas."""
    alvos = input_ids[:, 1:]
    mascara = alvo_valido[:, 1:]
    lg = logits[:, :-1].float()
    perdas = F.cross_entropy(lg.reshape(-1, lg.shape[-1]), alvos.reshape(-1), reduction="none")
    perdas = perdas * mascara.reshape(-1)
    n = mascara.sum()
    return perdas.sum(), n


# ---------------------------------------------------------------- avaliação

@torch.no_grad()
def avaliar(model, seqs, exemplos, split, n_positions, pad_id, device, autocast, tokens_por_lote=65536):
    model.eval()
    if not exemplos:
        model.train()
        return float("nan"), float("nan"), 0
    comp = np.array([min(seqs.len_full[i], n_positions + 1) for i, _ in exemplos])
    ordem = np.argsort(comp, kind="stable")
    perda_total, n_total = 0.0, 0
    atual, maior = [], 0

    def rodar(lote):
        nonlocal perda_total, n_total
        ids, alvo = montar_lote_avaliacao(seqs, [exemplos[k] for k in lote], split, n_positions, pad_id)
        ids_t = torch.from_numpy(ids).to(device)
        alvo_t = torch.from_numpy(alvo).to(device)
        with autocast():
            logits = model(ids_t, (ids_t != pad_id).long())
        s, n = perda_ntp(logits, ids_t, alvo_t)
        perda_total += s.item()
        n_total += int(n.item())

    for k in ordem:
        novo_maior = max(maior, comp[k])
        if atual and (len(atual) + 1) * novo_maior > tokens_por_lote:
            rodar(atual)
            atual, novo_maior = [], comp[k]
        atual.append(k)
        maior = novo_maior
    if atual:
        rodar(atual)
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


# ---------------------------------------------------------------- main

def main(config_path, retomar=False, saida="pretreino"):
    config = carregar_config(config_path)
    arch = arquitetura_efetiva(config.get("arquitetura", {}))
    cfg = config_pretreino(config)
    n_positions = arch["n_positions"]

    set_seed(cfg["seed"])
    out_dir = run_dir(config, saida)
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(f"{run_dir(config, 'vocabulario')}/vocab.json", encoding="utf-8") as f:
        vocab = json.load(f)
    pad_id = vocab["PAD"]
    seqs = carregar_sequencias(config)

    ent, ini, fim = janelas_de_treino(seqs, n_positions, cfg["janelas_treino"])
    comprimentos = fim - ini
    val_ex, _ = exemplos_de_avaliacao(seqs, "val")
    test_ex, n_test_frios = exemplos_de_avaliacao(seqs, "test")
    total_tokens_treino = int(np.sum(seqs.len_train[seqs.len_train > 0] + 1))  # tokens-alvo como no legado (len-1)
    n_entidades_treino = int((seqs.len_train > 0).sum())

    print(f"vocab_size={len(vocab)}  entidades_treino={n_entidades_treino}  janelas={len(ent)}  "
          f"tokens_treino={total_tokens_treino:,}")
    print(f"entidades_val_avaliaveis={len(val_ex)}  entidades_teste_avaliaveis={len(test_ex)} "
          f"(dos quais {n_test_frios} sem nenhum histórico de treino)")

    model = construir_modelo(arch, vocab).to(device)
    if cfg["precisao"] == "bf16_puro":
        model = model.to(torch.bfloat16)
        autocast = lambda: torch.autocast(device_type=device, enabled=False)  # noqa: E731
    else:
        autocast = lambda: torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda")  # noqa: E731
    n_params, n_params_trainable = model.n_parametros()
    print(f"Modelo ({arch['tipo']}): {n_params/1e6:.2f}M parâmetros ({n_params_trainable/1e6:.2f}M treináveis)")
    print(f"Razão tokens_treino/parametros_treináveis: {total_tokens_treino/n_params_trainable:.4f}")

    # planos de lote de todas as épocas, determinísticos (permitem retomar no meio)
    rng = np.random.RandomState(cfg["seed"])
    planos = [plano_de_lotes(comprimentos, cfg, rng) for _ in range(cfg["n_epochs"])]
    total_steps = sum(len(p) for p in planos)
    if cfg["max_passos"]:
        total_steps = min(total_steps, cfg["max_passos"])

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda s: lr_lambda(s, cfg["warmup_steps"], total_steps))

    rng_val = np.random.RandomState(cfg["seed"] + 1)
    val_amostra = [val_ex[k] for k in sorted(rng_val.choice(len(val_ex), min(len(val_ex), cfg["val_amostra_entidades"]), replace=False))] if val_ex else []

    estado = {"step": 0, "epoca": 1, "lote": 0, "melhor_loss_val": float("inf"), "melhor_epoca": -1,
              "historico": [], "sem_melhora": 0, "segundos": 0.0, "tokens_vistos": 0}
    ultimo_path = os.path.join(ckpt_dir, "ultimo.pt")
    if retomar and os.path.exists(ultimo_path):
        ck = torch.load(ultimo_path, map_location=device, weights_only=False)
        model.carregar_pesos(ck["modelo"])
        optimizer.load_state_dict(ck["otimizador"])
        scheduler.load_state_dict(ck["scheduler"])
        estado = ck["estado"]
        torch.set_rng_state(ck["rng_torch"])
        if device == "cuda" and ck.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(ck["rng_cuda"])
        print(f"Retomando do passo {estado['step']} (época {estado['epoca']}, lote {estado['lote']})")
    elif retomar:
        print("Nenhum checkpoint 'ultimo.pt' encontrado — começando do zero.")

    modo_arquivo = "a" if retomar and estado["step"] > 0 else "w"
    if modo_arquivo == "a":
        # descarta linhas registradas depois do checkpoint (serão refeitas)
        for nome, coluna, limite in [("pretreino_metricas_passo.csv", "step", estado["step"]),
                                     ("pretreino_metricas_epoca.csv", "epoch", estado["epoca"] - 1)]:
            path = os.path.join(out_dir, nome)
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    linhas = list(csv.reader(f))
                cab = linhas[0]
                k = cab.index(coluna)
                with open(path, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerows([cab] + [r for r in linhas[1:] if int(r[k]) <= limite])
    with open(os.path.join(out_dir, "pretreino_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "arquitetura": arch, "pretreino": cfg, "vocab_size": len(vocab),
            "n_params": n_params, "n_params_trainable": n_params_trainable,
            "n_entidades_treino": n_entidades_treino, "n_janelas_treino": int(len(ent)),
            "total_tokens_treino": total_tokens_treino,
            "n_entidades_val": len(val_ex), "n_entidades_teste": len(test_ex), "total_steps": total_steps,
            # campos de compatibilidade com o formato antigo
            "n_positions": n_positions, "n_embd": arch["n_embd"], "n_layer": arch["n_layer"], "n_head": arch["n_head"],
            "batch_size": cfg["batch_size"], "n_epochs": cfg["n_epochs"], "lr": cfg["lr"],
        }, f, ensure_ascii=False, indent=2)

    passo_csv = open(os.path.join(out_dir, "pretreino_metricas_passo.csv"), modo_arquivo, newline="", encoding="utf-8")
    passo_writer = csv.writer(passo_csv)
    epoca_csv = open(os.path.join(out_dir, "pretreino_metricas_epoca.csv"), modo_arquivo, newline="", encoding="utf-8")
    epoca_writer = csv.writer(epoca_csv)
    if modo_arquivo == "w":
        passo_writer.writerow(["step", "epoch", "loss_treino", "lr", "grad_norm", "tokens_por_seg", "gpu_mem_gb",
                               "elapsed_s", "loss_val_amostra"])
        epoca_writer.writerow(["epoch", "loss_treino_media", "loss_val", "perplexidade_val", "n_tokens_val", "elapsed_s"])

    def salvar_ultimo():
        torch.save({
            "modelo": model.pesos(), "otimizador": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "estado": estado, "rng_torch": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state() if device == "cuda" else None,
        }, ultimo_path + ".tmp")
        os.replace(ultimo_path + ".tmp", ultimo_path)

    model.train()
    t0_sessao = time.time()
    segundos_antes = estado["segundos"]
    parar = False
    perdas_epoca = estado.get("perdas_epoca", [])

    for epoca in range(estado["epoca"], cfg["n_epochs"] + 1):
        plano = planos[epoca - 1]
        t0_epoca = time.time()
        for b in range(estado["lote"], len(plano)):
            idx = plano[b]
            ids, alvo = montar_lote_treino(seqs, ent, ini, fim, idx, pad_id)
            ids_t = torch.from_numpy(ids).to(device, non_blocking=True)
            alvo_t = torch.from_numpy(alvo).to(device, non_blocking=True)
            t0_passo = time.time()

            optimizer.zero_grad(set_to_none=True)
            with autocast():
                logits = model(ids_t, (ids_t != pad_id).long())
            soma, n = perda_ntp(logits, ids_t, alvo_t)
            loss = soma / n.clamp(min=1)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()

            n_tok = int((ids_t != pad_id).sum().item())
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0_passo
            perdas_epoca.append(loss.item())
            estado["step"] += 1
            estado["lote"] = b + 1
            estado["tokens_vistos"] += n_tok
            estado["perdas_epoca"] = perdas_epoca

            loss_val_amostra = ""
            if cfg["val_a_cada_passos"] and estado["step"] % cfg["val_a_cada_passos"] == 0:
                lv, _, _ = avaliar(model, seqs, val_amostra, "val", n_positions, pad_id, device, autocast)
                loss_val_amostra = f"{lv:.6f}"
                print(f"  passo {estado['step']}/{total_steps}  loss={loss.item():.4f}  val_amostra={lv:.4f}")
            if estado["step"] % cfg["log_a_cada_n_passos"] == 0 or estado["step"] == 1 or loss_val_amostra:
                mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
                passo_writer.writerow([
                    estado["step"], epoca, f"{loss.item():.6f}", f"{scheduler.get_last_lr()[0]:.6e}",
                    f"{grad_norm.item():.4f}", f"{n_tok/max(dt,1e-6):.1f}", f"{mem_gb:.3f}",
                    f"{segundos_antes + time.time()-t0_sessao:.1f}", loss_val_amostra,
                ])
                passo_csv.flush()
            if cfg["checkpoint_a_cada_passos"] and estado["step"] % cfg["checkpoint_a_cada_passos"] == 0:
                estado["segundos"] = segundos_antes + time.time() - t0_sessao
                salvar_ultimo()
            if cfg["max_passos"] and estado["step"] >= cfg["max_passos"]:
                parar = True
                break

        loss_treino_media = float(np.mean(perdas_epoca)) if perdas_epoca else float("nan")
        loss_val, ppl_val, n_tokens_val = avaliar(model, seqs, val_ex, "val", n_positions, pad_id, device, autocast)
        elapsed_epoca = time.time() - t0_epoca
        epoca_writer.writerow([epoca, f"{loss_treino_media:.6f}", f"{loss_val:.6f}", f"{ppl_val:.4f}", n_tokens_val,
                               f"{elapsed_epoca:.1f}"])
        epoca_csv.flush()
        estado["historico"].append({"epoch": epoca, "loss_treino": loss_treino_media, "loss_val": loss_val,
                                    "perplexidade_val": ppl_val, "elapsed_s": elapsed_epoca})
        print(f"epoca {epoca:>3}/{cfg['n_epochs']}  loss_treino={loss_treino_media:.4f}  loss_val={loss_val:.4f}  "
              f"ppl_val={ppl_val:.3f}  ({elapsed_epoca:.1f}s)")

        torch.save(model.pesos(), os.path.join(ckpt_dir, "final.pt"))
        if loss_val < estado["melhor_loss_val"]:
            estado["melhor_loss_val"], estado["melhor_epoca"] = loss_val, epoca
            estado["sem_melhora"] = 0
            torch.save(model.pesos(), os.path.join(ckpt_dir, "melhor.pt"))
        else:
            estado["sem_melhora"] += 1

        estado["epoca"], estado["lote"] = epoca + 1, 0
        perdas_epoca = []
        estado["perdas_epoca"] = perdas_epoca
        estado["segundos"] = segundos_antes + time.time() - t0_sessao
        salvar_ultimo()
        if parar:
            break
        if cfg["paciencia_epocas"] and estado["sem_melhora"] >= cfg["paciencia_epocas"]:
            print(f"Early stopping: {estado['sem_melhora']} épocas sem melhora na validação")
            break

    passo_csv.close()
    epoca_csv.close()

    print(f"\nMelhor época (por loss de val): {estado['melhor_epoca']} (loss_val={estado['melhor_loss_val']:.4f})")
    model.carregar_pesos(torch.load(os.path.join(ckpt_dir, "melhor.pt"), map_location=device))
    loss_teste, ppl_teste, n_tokens_teste = avaliar(model, seqs, test_ex, "test", n_positions, pad_id, device, autocast)
    print(f"Loss de TESTE (melhor checkpoint, avaliação única): {loss_teste:.4f}  ppl={ppl_teste:.3f}")

    tempo_total = segundos_antes + time.time() - t0_sessao
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
    historico = estado["historico"]
    relatorio = {
        "arquitetura": arch,
        "n_params": n_params, "n_params_trainable": n_params_trainable,
        "total_tokens_treino": total_tokens_treino,
        "razao_tokens_por_parametro_treinavel": total_tokens_treino / n_params_trainable,
        "tokens_vistos": estado["tokens_vistos"], "passos": estado["step"],
        "n_epochs_executadas": len(historico), "melhor_epoca": estado["melhor_epoca"],
        "melhor_loss_val": estado["melhor_loss_val"],
        "melhor_perplexidade_val": math.exp(estado["melhor_loss_val"]),
        "loss_val_ultima_epoca": historico[-1]["loss_val"],
        "loss_treino_ultima_epoca": historico[-1]["loss_treino"],
        "loss_teste": loss_teste, "perplexidade_teste": ppl_teste, "n_tokens_teste_avaliados": n_tokens_teste,
        "n_entidades_teste_frias_incluidas": n_test_frios, "tempo_total_segundos": tempo_total,
        "tokens_por_segundo_medio": estado["tokens_vistos"] / max(tempo_total, 1e-9),
        "peak_gpu_mem_gb": peak_mem_gb,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu", "torch_version": torch.__version__,
    }
    with open(os.path.join(out_dir, "pretreino_relatorio.json"), "w", encoding="utf-8") as f:
        json.dump(relatorio, f, ensure_ascii=False, indent=2)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = [h["epoch"] for h in historico]
        plt.figure(figsize=(9, 5))
        plt.plot(epochs, [h["loss_treino"] for h in historico], label="loss treino", marker="o")
        plt.plot(epochs, [h["loss_val"] for h in historico], label="loss validação", marker="o")
        plt.axvline(estado["melhor_epoca"], color="gray", linestyle="--", alpha=0.6,
                    label=f"melhor época ({estado['melhor_epoca']})")
        plt.xlabel("Época"); plt.ylabel("Loss (cross-entropy, NTP)")
        plt.title(f"Pré-treino do backbone — {config['nome']}")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "pretreino_loss_curve.png"), dpi=150)
    except Exception as e:
        print(f"Aviso: não consegui gerar o gráfico ({e})")

    print(f"\nGravado {out_dir}/{{pretreino_config.json, pretreino_metricas_*.csv, pretreino_relatorio.json}}")
    print(f"Tempo total: {tempo_total/60:.1f} min | pico de VRAM: {peak_mem_gb:.2f}GB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--retomar", action="store_true")
    ap.add_argument("--saida", default="pretreino")
    a = ap.parse_args()
    main(a.config, a.retomar, a.saida)
