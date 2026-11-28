"""Teste de vazamento por perturbação do futuro: baselines dos rótulos e features tradicionais.

Ideia: tudo o que o modelo recebe numa âncora de data d só pode depender de eventos até d. Então,
se perturbarmos os eventos a partir de uma data T (valores numéricos multiplicados por fatores
aleatórios, categorias embaralhadas entre si) e refizermos o preparo do zero, os baselines dos
rótulos (`*_baseline`) e todas as features das âncoras anteriores a T têm de sair idênticos.

T é escolhido depois do fim do treino mais o maior horizonte das tarefas: assim, os rótulos de
treino (e as faixas e calibrações tiradas deles) não mudam, e qualquer diferença antes de T é
vazamento. Cobre o que o teste de vazamento de features_tradicionais.py deixa de fora de propósito:
os `*_baseline_feat`, que vêm de tarefa8_rotulos.py. O baseline de LTV vazava até 27/11/2026.

Uso: python3 testes/teste_futuro.py <config.yaml> (em geral, a cópia temporária do testar.py)
"""
import os
import shutil
import subprocess
import sys
from datetime import timedelta

import numpy as np
import polars as pl
import yaml

AQUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHAVES = {"evento_id", "entidade_id", "data_evento", "pedido_id"}


def main():
    cfg_path = sys.argv[1]
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    nome = cfg["nome"]
    run = os.path.join(AQUI, "runs", nome)
    for p in ("rotulos_downstream/rotulos.parquet", "features_tradicionais/features.parquet"):
        assert os.path.exists(os.path.join(run, p)), f"rode o preparo antes: falta {p}"

    fonte = os.path.join(AQUI, cfg["fonte_canonica"])
    ev = pl.read_parquet(fonte)
    fim = ev["data_evento"].max()
    corte_teste = fim - timedelta(days=cfg["splits"]["test_window_days"])
    corte_val = corte_teste - timedelta(days=cfg["splits"]["val_window_days"])
    horizontes = [t.get("dias_escolhido", t.get("horizonte_dias", 0)) for t in cfg["tarefas_downstream"]]
    horizontes += [max(t.get("candidatos_dias", [0])) for t in cfg["tarefas_downstream"]]
    T = corte_val + timedelta(days=max(horizontes) + 1)
    T = T.replace(hour=0, minute=0, second=0, microsecond=0) if hasattr(T, "replace") else T
    assert T < fim, f"base curta demais para o teste: T={T} fim={fim}"

    estaticos = set(cfg.get("campos_estaticos_categoricos", [])) | set(cfg.get("campos_estaticos_numericos", []))
    fut = ev["data_evento"] >= T
    n_fut = int(fut.sum())
    rng = np.random.RandomState(0)
    novas = []
    for c, tipo in ev.schema.items():
        if c in CHAVES or c in estaticos:
            continue
        col = ev[c]
        if tipo.is_float() and col.drop_nulls().n_unique() > 2:   # binários guardados como float são embaralhados
            fator = np.ones(ev.height)
            fator[fut.to_numpy()] = rng.uniform(0.5, 2.0, n_fut)
            novas.append((col.cast(pl.Float64) * fator).cast(tipo).alias(c))
        else:   # categorias, inteiros (ids de item) e binários: embaralhados entre as linhas do futuro
            idx = np.arange(ev.height)
            f_idx = np.flatnonzero(fut.to_numpy())
            idx[f_idx] = rng.permutation(f_idx)
            novas.append(col.gather(idx).alias(c))
    pert = ev.with_columns(novas)

    p_nome = f"{nome}__futuro"
    p_run = os.path.join(AQUI, "runs", p_nome)
    p_cfg = os.path.join(AQUI, "runs", f"{p_nome}.yaml")
    shutil.rmtree(p_run, ignore_errors=True)
    os.makedirs(p_run)
    pert.write_parquet(os.path.join(p_run, "eventos_canonicos.parquet"))
    c2 = {**cfg, "nome": p_nome, "fonte_canonica": os.path.relpath(os.path.join(p_run, "eventos_canonicos.parquet"), AQUI)}
    c2.get("motor", {}).pop("adapter", None) if isinstance(c2.get("motor"), dict) else None
    with open(p_cfg, "w") as f:
        yaml.safe_dump(c2, f, allow_unicode=True, sort_keys=False)
    print(f"T = {T:%Y-%m-%d} (treino até {corte_val:%Y-%m-%d} + {max(horizontes)} dias); "
          f"{n_fut:,} de {ev.height:,} eventos perturbados", flush=True)
    falhas = 0
    try:
        r = subprocess.run([sys.executable, "-u", os.path.join(AQUI, "motor.py"), p_cfg, "preparar"], cwd=AQUI,
                           stdout=subprocess.DEVNULL)
        if r.returncode != 0:
            print("  [FALHOU] preparo da base perturbada"); sys.exit(1)
        datas = ev.select("evento_id", "data_evento")
        for arq, filtro in (("rotulos_downstream/rotulos.parquet", lambda c: c.endswith("_baseline")),
                            ("features_tradicionais/features.parquet",
                             lambda c: c not in ("evento_id", "entidade_id", "split") and not c.endswith(("_rotulo", "_baseline")))):
            a = pl.read_parquet(os.path.join(run, arq)).join(datas, on="evento_id").filter(pl.col("data_evento") < T)
            b = pl.read_parquet(os.path.join(p_run, arq)).join(datas, on="evento_id").filter(pl.col("data_evento") < T)
            assert a.height == b.height and a.height > 0, (arq, a.height, b.height)
            a, b = a.sort("evento_id"), b.sort("evento_id")
            cols = [c for c in a.columns if filtro(c)]
            ruins = []
            for c in cols:
                x, y = a[c], b[c]
                if x.dtype.is_numeric():
                    xv, yv = x.cast(pl.Float64).to_numpy(), y.cast(pl.Float64).to_numpy()
                    igual = np.isclose(xv, yv, rtol=1e-5, atol=1e-6, equal_nan=True)
                else:
                    igual = (x.cast(pl.Utf8).fill_null("<nulo>") == y.cast(pl.Utf8).fill_null("<nulo>")).to_numpy()
                if not igual.all():
                    ruins.append(f"{c} ({(~igual).mean():.1%} das âncoras)")
            tag = os.path.dirname(arq)
            if ruins:
                falhas += 1
                print(f"  [FALHOU] {tag}: mudam com o futuro: {', '.join(ruins)}")
            else:
                print(f"  [OK] {tag}: {len(cols)} colunas idênticas em {a.height:,} âncoras anteriores a T")
    finally:
        shutil.rmtree(p_run, ignore_errors=True)
        os.remove(p_cfg)
    sys.exit(1 if falhas else 0)


if __name__ == "__main__":
    main()
