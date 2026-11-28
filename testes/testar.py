#!/usr/bin/env python3
"""Testes do motor antes de entregar ou de mudar o código (critério 4 da seção 7 do PLANO_FASE3.md).

1. Regressão: regenera a base do zero (preparar + serializar) numa pasta temporária
   e compara cada artefato com a referência gravada em testes/referencia/<nome>.json.
   Qualquer mudança de código que altere splits, faixas, rótulos, variáveis ou a
   serialização do v2 aparece aqui. Números de ponto flutuante são comparados com 6
   casas decimais.
2. Vazamento: as variáveis tradicionais de âncoras de validação e teste,
   recalculadas só com os eventos até a data de cada uma, têm de sair idênticas.
3. Motor v2: causalidade (mudar o futuro não muda o passado), censura (nenhum alvo
   usa evento depois do corte) e aprendizado mínimo (pipeline/common/v2_testes.py).

Uso:
  python3 testes/testar.py configs/beleza_b2b_mini.yaml                      # roda e compara
  python3 testes/testar.py configs/beleza_b2b_mini.yaml --gravar_referencia  # grava a referência
Feito para bases pequenas (a mini roda em poucos minutos na CPU).
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml

AQUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = [sys.executable, "-u"]
ARTEFATOS = ["splits/splits.parquet", "discretizacao/discretizado.parquet", "rotulos_downstream/rotulos.parquet",
             "features_tradicionais/features.parquet", "v2/*.npy", "v2/*.parquet"]


def impressao(caminho):
    """Assinatura do conteúdo (não dos bytes do arquivo, que mudam com a versão da biblioteca)."""
    h = hashlib.sha256()
    if caminho.endswith(".npy"):
        x = np.load(caminho, allow_pickle=False)
        if x.dtype.kind == "f":
            x = np.round(x.astype(np.float64), 6)
        h.update(str(x.shape).encode()); h.update(np.ascontiguousarray(x).tobytes())
        return {"sha256": h.hexdigest(), "forma": list(x.shape)}
    df = pd.read_parquet(caminho)
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].round(6)
    h.update(",".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return {"sha256": h.hexdigest(), "linhas": int(len(df)), "colunas": list(map(str, df.columns))}


def rodar(*args, ok_codes=(0,)):
    cmd = PY + [str(x) for x in args]
    print("$", " ".join(cmd[1:]), flush=True)
    r = subprocess.run(cmd, cwd=AQUI)
    return r.returncode in ok_codes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--gravar_referencia", action="store_true")
    ap.add_argument("--manter", action="store_true", help="não apaga a pasta temporária")
    a = ap.parse_args()
    with open(os.path.join(AQUI, a.config) if not os.path.isabs(a.config) else a.config) as f:
        cfg = yaml.safe_load(f)
    nome = cfg["nome"]
    tmp_nome = f"{nome}__teste"
    tmp_cfg = os.path.join(AQUI, "runs", f"{tmp_nome}.yaml")
    tmp_run = os.path.join(AQUI, "runs", tmp_nome)
    shutil.rmtree(tmp_run, ignore_errors=True)
    os.makedirs(os.path.dirname(tmp_cfg), exist_ok=True)
    with open(tmp_cfg, "w") as f:
        yaml.safe_dump({**cfg, "nome": tmp_nome}, f, allow_unicode=True, sort_keys=False)
    falhas = []
    try:
        # 1. regressão
        print("\n### 1. Regressão: regenera a base do zero", flush=True)
        for etapa in ("preparar", "serializar"):
            if not rodar(os.path.join(AQUI, "motor.py"), tmp_cfg, etapa):
                falhas.append(f"motor {etapa}")
        atual = {}
        for padrao in ARTEFATOS:
            for p in sorted(glob.glob(os.path.join(tmp_run, padrao))):
                atual[os.path.relpath(p, tmp_run)] = impressao(p)
        ref_path = os.path.join(AQUI, "testes", "referencia", f"{nome}.json")
        if a.gravar_referencia:
            os.makedirs(os.path.dirname(ref_path), exist_ok=True)
            with open(ref_path, "w") as f:
                json.dump(atual, f, indent=1, ensure_ascii=False)
            print(f"  referência gravada: {ref_path} ({len(atual)} artefatos)")
        elif not os.path.exists(ref_path):
            falhas.append("regressão (sem referência: rode com --gravar_referencia numa versão confiável)")
        else:
            with open(ref_path) as f:
                ref = json.load(f)
            for k in sorted(set(ref) | set(atual)):
                if k not in atual:
                    print(f"  [FALHOU] {k}: sumiu"); falhas.append(f"regressão {k}")
                elif k not in ref:
                    print(f"  [FALHOU] {k}: artefato novo, fora da referência"); falhas.append(f"regressão {k}")
                elif ref[k] != atual[k]:
                    print(f"  [FALHOU] {k}: conteúdo mudou"); falhas.append(f"regressão {k}")
            print(f"  {len(atual)} artefatos comparados com {os.path.relpath(ref_path, AQUI)}")
        # 2. vazamento
        print("\n### 2. Vazamento das variáveis tradicionais", flush=True)
        if not rodar(os.path.join(AQUI, "pipeline", "common", "features_tradicionais.py"), tmp_cfg, "--teste_vazamento", 300):
            falhas.append("vazamento")
        print("\n### 2b. Vazamento por perturbação do futuro (baselines dos rótulos e features)", flush=True)
        if not rodar(os.path.join(AQUI, "testes", "teste_futuro.py"), tmp_cfg):
            falhas.append("vazamento (perturbação do futuro)")
        # 3. motor v2
        print("\n### 3. Motor v2: serialização, censura, causalidade, aprendizado", flush=True)
        if not rodar(os.path.join(AQUI, "pipeline", "common", "v2_testes.py"), tmp_cfg, "v2"):
            falhas.append("motor v2")
    finally:
        if not a.manter:
            shutil.rmtree(tmp_run, ignore_errors=True)
            os.remove(tmp_cfg)
    print("\n" + ("TODOS OS TESTES PASSARAM" if not falhas else f"{len(falhas)} FALHA(S): {falhas}"))
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()
