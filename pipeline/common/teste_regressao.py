"""Teste de regressão do motor genérico contra artefatos de referência.

Compara, etapa por etapa, os artefatos de dados de um run recém-gerado
(`runs/<nome>/`) com uma cópia de referência (ex. os artefatos commitados
antes de uma refatoração). Não depende de GPU: cobre splits,
discretização, vocabulário, serialização (token a token) e rótulos.

Critério:
  - splits, buckets, vocab, top_n_mapa, sequências, posições de âncora,
    rótulos e n_eventos_anteriores: idênticos;
  - baselines numéricos: iguais até 1e-9 (relativo) — a média causal
    vetorizada (somas acumuladas) difere da versão com expanding().mean()
    só no último bit de ponto flutuante.

Uso: python3 pipeline/common/teste_regressao.py <config.yaml> <dir_referencia_do_run>
     (ex. <dir_referencia_do_run> = /caminho/ref_runs/synthetic_agro)
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "pipeline/common")
from schema import caminho_tabela, carregar_config, ler_tabela, run_dir, tarefas_ativas  # noqa: E402
from sequencias_io import Sequencias  # noqa: E402

falhas = []


def checar(nome, ok, detalhe=""):
    print(f"  [{'OK' if ok else 'FALHOU'}] {nome}{(' — ' + detalhe) if detalhe else ''}")
    if not ok:
        falhas.append(nome)


def ler_ref(ref_dir, subpasta, nome, colunas=None):
    base = os.path.join(ref_dir, subpasta, nome)
    return ler_tabela(base + ".parquet" if os.path.exists(base + ".parquet") else base + ".csv", colunas)


def iguais_series(a, b):
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
        return bool(np.array_equal(a.to_numpy(dtype=float), b.to_numpy(dtype=float), equal_nan=True))
    na, nb = a.isna(), b.isna()
    return bool((na == nb).all() and (a[~na].astype(str) == b[~nb].astype(str)).all())


def main(config_path, ref_dir):
    config = carregar_config(config_path)
    print(f"Regressão de {config['nome']} contra {ref_dir}\n")

    print("Splits")
    novo = ler_tabela(caminho_tabela(config, "splits", "splits")).sort_values("evento_id").reset_index(drop=True)
    ref = ler_ref(ref_dir, "splits", "splits").sort_values("evento_id").reset_index(drop=True)
    checar("split por evento", iguais_series(novo["split"], ref["split"]))
    checar("elegivel_downstream", iguais_series(novo["elegivel_downstream"].astype(bool), ref["elegivel_downstream"].astype(bool)))

    if config["campos_numericos"]:
        print("Discretização")
        with open(os.path.join(run_dir(config, "discretizacao"), "buckets.json")) as f:
            b_novo = json.load(f)
        with open(os.path.join(ref_dir, "discretizacao", "buckets.json")) as f:
            b_ref = json.load(f)
        checar("buckets.json", b_novo == b_ref)
        novo = ler_tabela(caminho_tabela(config, "discretizacao", "discretizado")).sort_values("evento_id").reset_index(drop=True)
        ref = ler_ref(ref_dir, "discretizacao", "discretizado").sort_values("evento_id").reset_index(drop=True)
        for c in config["campos_numericos"]:
            checar(f"{c['nome']}_bucket", iguais_series(novo[f"{c['nome']}_bucket"], ref[f"{c['nome']}_bucket"]))

    print("Vocabulário")
    for nome in ["vocab.json", "top_n_mapa.json"]:
        with open(os.path.join(run_dir(config, "vocabulario"), nome)) as f:
            a = json.load(f)
        with open(os.path.join(ref_dir, "vocabulario", nome)) as f:
            b = json.load(f)
        checar(nome, a == b)

    print("Serialização")
    seq_novo = Sequencias.carregar(run_dir(config, "serializacao"))
    seq_ref = Sequencias.carregar(os.path.join(ref_dir, "serializacao"))
    checar("ordem das entidades", [str(x) for x in seq_novo.ids] == [str(x) for x in seq_ref.ids])
    diffs = 0
    for i in range(len(seq_ref)):
        j = seq_novo.indice(seq_ref.ids[i])
        if j is None or seq_novo.como_dict(j) != seq_ref.como_dict(i):
            diffs += 1
    checar("seq_full/seq_train/seq_train_val/n_eventos", diffs == 0, f"{diffs} entidades diferentes de {len(seq_ref)}")
    pos_novo = ler_tabela(caminho_tabela(config, "serializacao", "posicoes_evento"))
    pos_ref = ler_ref(ref_dir, "serializacao", "posicoes_evento")
    checar("posicoes_evento (ordem e valores)",
           iguais_series(pos_novo["evento_id"], pos_ref["evento_id"]) and iguais_series(pos_novo["pos_full"], pos_ref["pos_full"]))

    print("Rótulos")
    novo = ler_tabela(caminho_tabela(config, "rotulos_downstream", "rotulos")).sort_values("evento_id").reset_index(drop=True)
    ref = ler_ref(ref_dir, "rotulos_downstream", "rotulos").sort_values("evento_id").reset_index(drop=True)
    checar("n_eventos_anteriores", iguais_series(novo["n_eventos_anteriores"], ref["n_eventos_anteriores"]))
    for t in tarefas_ativas(config):
        nome = t.get("nome", t["tipo"])
        checar(f"{nome}_rotulo", iguais_series(novo[f"{nome}_rotulo"], ref[f"{nome}_rotulo"]))
        a, b = novo[f"{nome}_baseline"], ref[f"{nome}_baseline"]
        a_num, b_num = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
        if a_num.notna().sum() == a.notna().sum() and b_num.notna().sum() == b.notna().sum():
            ok = bool(np.allclose(a_num.to_numpy(float), b_num.to_numpy(float), rtol=1e-9, atol=1e-12, equal_nan=True))
            dif = float(np.nanmax(np.abs(a_num.to_numpy(float) - b_num.to_numpy(float)))) if len(a) else 0.0
            checar(f"{nome}_baseline (numérico, rtol=1e-9)", ok, f"máx |diff|={dif:.2e}")
        else:
            checar(f"{nome}_baseline (categórico)", iguais_series(a, b))

    with open(os.path.join(run_dir(config, "rotulos_downstream"), "relatorio.json")) as f:
        r_novo = json.load(f)
    with open(os.path.join(ref_dir, "rotulos_downstream", "relatorio.json")) as f:
        r_ref = json.load(f)
    for nome, info in r_ref.items():
        for chave in ["n_linhas_determinaveis", "base_rate_geral"]:
            if chave in info:
                a, b = r_novo[nome].get(chave), info[chave]
                checar(f"relatório {nome}.{chave}", a == b or (isinstance(a, float) and abs(a - b) < 1e-12), f"{a} vs {b}")

    print(f"\n{'TUDO OK' if not falhas else f'{len(falhas)} FALHA(S): ' + ', '.join(falhas)}")
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
