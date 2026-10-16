"""Driver das ablações de arquitetura do backbone — H8 do experimento H&M.

Cada ablação muda UM fator da referência e roda, com o mesmo orçamento
(1 época de pré-treino, mesma semente, mesmos dados):
  1. pré-treino           -> runs/<nome>/abl_<ablacao>/pretreino/
  2. extração de embeddings (fp16, todas as âncoras) -> .../embeddings/
  3. sondas downstream fixas (iguais para todas as ablações):
       - LightGBM com hiperparâmetros fixos sobre PCA-32 do embedding
         (churn e next_category, validação)
       - fusão DCNv2 (todas as tarefas, validação)
Grava runs/<nome>/ablacoes/<ablacao>.json com loss/perplexidade de
validação, custo (tokens/s, VRAM, tempo) e as métricas das sondas.

Etapas já concluídas são puladas (dá pra retomar depois de queda do Colab;
o pré-treino retoma do último checkpoint).

Uso: python3 pipeline/common/experimento_ablacao.py <config_base.yaml> <ablacao> [<ablacao> ...]
     python3 pipeline/common/experimento_ablacao.py <config_base.yaml> --listar
"""

import copy
import json
import os
import subprocess
import sys

import yaml

sys.path.insert(0, "pipeline/common")
from schema import carregar_config, run_dir  # noqa: E402

# referência das ablações = decoder próprio na configuração equivalente ao
# GPT-2 + NoPE da H3 (isola um fator por vez com o mesmo código)
REFERENCIA = {"tipo": "decoder", "n_embd": 384, "n_layer": 6, "n_head": 6, "n_kv_heads": 6, "n_positions": 1024,
              "posicao": "nope", "norm": "layernorm", "ffn": "gelu", "dropout": 0.1, "bias": True}

ABLACOES = {
    "ref": {},
    # tamanho (curva de escala)
    "tam_128x4": {"n_embd": 128, "n_layer": 4, "n_head": 4, "n_kv_heads": 4},
    "tam_256x6": {"n_embd": 256, "n_layer": 6, "n_head": 4, "n_kv_heads": 4},
    "tam_512x8": {"n_embd": 512, "n_layer": 8, "n_head": 8, "n_kv_heads": 8},
    # atenção
    "attn_gqa2": {"n_kv_heads": 2},
    "attn_mqa": {"n_kv_heads": 1},
    # FFN
    "ffn_swiglu": {"ffn": "swiglu"},
    # posição
    "pos_rope": {"posicao": "rope"},
    "pos_aprendida": {"posicao": "aprendida"},
    # normalização
    "norm_rms": {"norm": "rmsnorm"},
    # dropout
    "drop_0": {"dropout": 0.0},
    # contexto
    "ctx_512": {"n_positions": 512},
    "ctx_2048": {"n_positions": 2048},
    # combinação "moderna" (estilo Llama) no tamanho de referência
    "moderna": {"posicao": "rope", "norm": "rmsnorm", "ffn": "swiglu", "n_kv_heads": 2, "dropout": 0.0, "bias": False},
}

PRETREINO_ABLACAO = {"n_epochs": 1, "paciencia_epocas": None, "val_a_cada_passos": 1000,
                     "checkpoint_a_cada_passos": 500}


def config_ablacao(config_base_path, nome, extra=None):
    base = yaml.safe_load(open(config_base_path, encoding="utf-8"))
    cfg = copy.deepcopy(base)
    cfg["arquitetura"] = {**REFERENCIA, **ABLACOES.get(nome, {}), **(extra or {})}
    cfg["pretreino"] = {**base.get("pretreino", {}), **PRETREINO_ABLACAO}
    os.makedirs(run_dir(cfg, "ablacoes"), exist_ok=True)
    path = os.path.join(run_dir(cfg, "ablacoes"), f"{nome}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path, cfg


def rodar(cmd):
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main(config_base_path, nomes):
    py = sys.executable
    for nome in nomes:
        path, cfg = config_ablacao(config_base_path, nome)
        sub = f"abl_{nome}"
        pre_dir = os.path.join(run_dir(cfg, sub), "pretreino")
        emb_dir = os.path.join(run_dir(cfg, sub), "embeddings")
        rel_pre = os.path.join(pre_dir, "pretreino_relatorio.json")
        print(f"\n######## ablação {nome}: {cfg['arquitetura']}", flush=True)

        if not os.path.exists(rel_pre):
            rodar([py, "-u", "pipeline/common/tarefa_pretreino.py", path, "--retomar", "--saida", f"{sub}/pretreino"])
        if not os.path.exists(os.path.join(emb_dir, "embeddings_eventos_index.parquet")):
            rodar([py, "-u", "pipeline/common/tarefa13_extrair_embeddings.py", path, "--pretreino", f"{sub}/pretreino",
                   "--saida", f"{sub}/embeddings", "--dtype", "float16"])
        sonda_gbdt = os.path.join(run_dir(cfg, "gbdt"), f"sonda_{nome}", "relatorio.json")
        if not os.path.exists(sonda_gbdt):
            rodar([py, "-u", "pipeline/common/tarefa14_gbdt.py", path, "--features", "emb_pca", "--pca_k", "32",
                   "--modelo", "lgbm", "--n_trials", "0", "--seeds", "0", "--tarefas", "churn,next_category",
                   "--emb", f"{sub}/embeddings", "--saida", f"sonda_{nome}"])
        sonda_fusao = os.path.join(run_dir(cfg, sub), "fusao", "tarefa13_relatorio.json")
        if not os.path.exists(sonda_fusao):
            rodar([py, "-u", "pipeline/common/tarefa13_treinar_fusao.py", path, "--emb", f"{sub}/embeddings",
                   "--saida", f"{sub}/fusao", "--dispositivo", "auto"])

        pre = json.load(open(rel_pre))
        gb = json.load(open(sonda_gbdt))
        fu = json.load(open(sonda_fusao))
        resumo = {
            "ablacao": nome, "arquitetura": cfg["arquitetura"],
            "n_params_treinaveis": pre["n_params_trainable"],
            "loss_val_amostra_epoca": pre["melhor_loss_val"], "loss_val_completa": pre.get("loss_val_completa"),
            "perplexidade_val_completa": pre.get("perplexidade_val_completa"),
            "tokens_por_segundo": pre.get("tokens_por_segundo_medio"), "peak_gpu_mem_gb": pre["peak_gpu_mem_gb"],
            "tempo_pretreino_s": pre["tempo_total_segundos"],
            "sonda_gbdt_val": {t: v["val_resumo"] for t, v in gb["tarefas"].items()},
            "sonda_fusao_val": fu["comparacao_com_baseline_validacao"],
        }
        with open(os.path.join(run_dir(cfg, "ablacoes"), f"{nome}.json"), "w", encoding="utf-8") as f:
            json.dump(resumo, f, ensure_ascii=False, indent=2)
        print(f"\n>>> {nome}: loss_val={resumo['loss_val_completa']:.4f}  "
              + "  ".join(f"gbdt/{t}={list(v.values())[0]['media']:.4f}" for t, v in resumo["sonda_gbdt_val"].items())
              + "  " + "  ".join(f"fusao/{t}={v['modelo']:.4f}" for t, v in resumo["sonda_fusao_val"].items()), flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "--listar":
        for k, v in ABLACOES.items():
            print(f"{k:15s} {v}")
        sys.exit(0)
    main(sys.argv[1], sys.argv[2:])
