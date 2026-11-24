#!/usr/bin/env python3
"""Motor genérico de modelos de cliente: um comando por etapa, com retomada.

Roda o pipeline inteiro de uma base a partir do YAML, em qualquer máquina
(CPU ou GPU; nada depende do Colab). Cada etapa pula o que já está pronto,
então o mesmo comando retoma de onde parou depois de uma queda. Os tempos de
cada etapa ficam em runs/<nome>/motor_registro.json (registro de esforço).

Etapas, na ordem de `tudo`:
  preparar     adapter (se declarado em `motor.adapter` no YAML), splits, faixas,
               rótulos e as variáveis tradicionais (braço T)
  serializar   histórico do motor v2 (uma posição por pedido ou sessão)
  pretreinar   pré-treino do v2, sem rótulos (retoma do último checkpoint)
  extrair      vetor do cliente em cada momento avaliado
  fusao        braço E: vetor + cabeça leve, 3 sementes
  gbdt         braços T (variáveis + LightGBM com HPO) e H (T + vetor em PCA-32)
  clusterizar  segmentos RFM, T, E e H na validação e no teste
  avaliar      E e H contra T, com IC 95% por bootstrap por cliente
Fora de `tudo`:
  testes       regressão, vazamento e causalidade do v2 (ver `testar`)

Uso:
  python3 motor.py <config.yaml> <etapa> [opções]
  python3 motor.py configs/beleza_b2b_mini.yaml tudo --epocas 2 --n_trials 5
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

import yaml

AQUI = os.path.dirname(os.path.abspath(__file__))
PY = [sys.executable, "-u"]
ETAPAS = ["preparar", "serializar", "pretreinar", "extrair", "fusao", "gbdt", "clusterizar", "avaliar"]


def comum(nome):
    return os.path.join(AQUI, "pipeline", "common", nome)


class Motor:
    def __init__(self, config_path, a):
        self.config_path = config_path
        self.a = a
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.nome = self.cfg["nome"]
        self.R = os.path.join(AQUI, "runs", self.nome)
        self.m = self.cfg.get("motor", {}) or {}
        self.registro_path = os.path.join(self.R, "motor_registro.json")

    # ------------------------------------------------------------ utilidades
    def run(self, *args):
        cmd = PY + [str(x) for x in args]
        print("$", " ".join(cmd[1:]), flush=True)
        r = subprocess.run(cmd, cwd=AQUI)
        if r.returncode != 0:
            sys.exit(f"falhou (código {r.returncode}): {' '.join(cmd[2:])}")

    def existe(self, *partes):
        return os.path.exists(os.path.join(self.R, *partes))

    def registrar(self, etapa, segundos, pulou):
        reg = {}
        if os.path.exists(self.registro_path):
            with open(self.registro_path) as f:
                reg = json.load(f)
        if not pulou:
            reg[etapa] = {"segundos": round(segundos, 1), "quando": time.strftime("%Y-%m-%d %H:%M:%S"),
                          "dispositivo": self.a.dispositivo}
            os.makedirs(self.R, exist_ok=True)
            with open(self.registro_path, "w") as f:
                json.dump(reg, f, indent=2, ensure_ascii=False)

    def tarefas(self):
        return [t.get("nome", t["tipo"]) for t in self.cfg["tarefas_downstream"]]

    def sementes(self):
        return [int(s) for s in self.a.sementes.split(",")]

    # ------------------------------------------------------------ etapas
    def preparar(self):
        feito = True
        ad = self.m.get("adapter")
        if ad and not os.path.exists(os.path.join(AQUI, self.cfg["fonte_canonica"])):
            self.run(os.path.join(AQUI, ad["script"]), *ad.get("args", [])); feito = False
        for pasta, arq, script in (("splits", "splits.parquet", "tarefa7_splits.py"),
                                   ("discretizacao", "discretizado.parquet", "tarefa4_discretizacao.py"),
                                   ("rotulos_downstream", "rotulos.parquet", "tarefa8_rotulos.py"),
                                   ("features_tradicionais", "features.parquet", "features_tradicionais.py")):
            if not self.existe(pasta, arq):
                self.run(comum(script), self.config_path); feito = False
        return feito

    def serializar(self):
        if self.existe("v2", "meta.json"):
            return True
        self.run(comum("v2_serializacao.py"), self.config_path, "--dados", "v2")
        return False

    def pretreinar(self):
        rel = os.path.join(self.R, "v2_pretreino", "v2_relatorio.json")
        if os.path.exists(rel):
            with open(rel) as f:
                if len(json.load(f).get("historico", [])) >= self.a.epocas:
                    return True
        self.run(comum("v2_pretreino.py"), self.config_path, "--dados", "v2", "--saida", "v2_pretreino",
                 "--epocas", self.a.epocas, "--retomar", "--dispositivo", self.a.dispositivo,
                 "--trava_gpu", os.path.join(self.R, ".trava_gpu"))
        return False

    def extrair(self):
        if self.existe("emb_v2", "embeddings_eventos_index.parquet"):
            return True
        self.run(comum("v2_extrair.py"), self.config_path, "--dados", "v2", "--pretreino", "v2_pretreino",
                 "--saida", "emb_v2", "--dispositivo", self.a.dispositivo)
        return False

    def fusao(self):
        feito = True
        for s in self.sementes():
            if not self.existe(f"fus_v2_s{s}", "predicoes_test.parquet"):
                self.run(comum("tarefa13_treinar_fusao.py"), self.config_path, "--emb", "emb_v2", "--seed", s,
                         "--saida", f"fus_v2_s{s}", "--dispositivo", self.a.dispositivo, "--avaliar_teste")
                feito = False
        return feito

    def gbdt(self):
        feito = True
        g = [comum("tarefa14_gbdt.py"), self.config_path, "--modelo", "lgbm", "--seeds", self.a.sementes,
             "--threads", self.a.threads, "--n_trials", self.a.n_trials, "--avaliar_teste"]
        if not self.existe("gbdt", "motor_trad_lgbm", "relatorio.json"):
            self.run(*g, "--features", "trad", "--saida", "motor_trad_lgbm"); feito = False
        if not self.existe("gbdt", "motor_hibrido_lgbm", "relatorio.json"):
            self.run(*g, "--features", "hibrido", "--emb", "emb_v2", "--pca_k", 32,
                     "--pca_cache", os.path.join(self.R, "caches", "pca32_v2.npz"),
                     "--saida", "motor_hibrido_lgbm"); feito = False
        return feito

    def clusterizar(self):
        feito = True
        for fase in ("val", "test"):
            if not self.existe("motor_cluster", f"resultado_{fase}.json"):
                self.run(comum("f3h2_clusterizacao.py"), self.config_path, "--fase", fase,
                         "--emb_cache", "caches/pca32_v2.npz", "--saida", "motor_cluster"); feito = False
        return feito

    def avaliar(self):
        saida = os.path.join(self.R, "motor_bootstrap_test.json")
        if os.path.exists(saida) and not self.a.refazer:
            return True
        if os.path.exists(saida):
            os.remove(saida)
        for t in self.tarefas():
            self.run(comum("avaliacao.py"), self.config_path, "--tarefa", t, "--split", "test", "--B", self.a.B,
                     "--saida", saida, "--a", "gbdt/motor_trad_lgbm", "--b", "fus_v2_s*", "--b", "gbdt/motor_hibrido_lgbm")
        return False

    def testes(self):
        """Regressão, vazamento e causalidade do motor v2 (critério 4 da seção 7 do PLANO_FASE3.md)."""
        self.run(os.path.join(AQUI, "testes", "testar.py"), self.config_path)
        return False

    def executar(self, etapa):
        t0 = time.time()
        print(f"\n=== {etapa} ({self.nome}) {time.strftime('%H:%M:%S')}", flush=True)
        pulou = getattr(self, etapa)()
        dt = time.time() - t0
        print(f"=== {etapa}: {'já estava pronta' if pulou else f'{dt:.0f} s'}", flush=True)
        self.registrar(etapa, dt, pulou)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("etapa", choices=ETAPAS + ["tudo", "testes"])
    ap.add_argument("--dispositivo", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--epocas", type=int, default=8, help="épocas do pré-treino (escolhe-se a melhor na validação)")
    ap.add_argument("--n_trials", type=int, default=40, help="tentativas de HPO do LightGBM por tarefa")
    ap.add_argument("--sementes", default="0,1,2")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--B", type=int, default=300, help="reamostragens do bootstrap")
    ap.add_argument("--refazer", action="store_true", help="refaz a avaliação mesmo que já exista")
    a = ap.parse_args()
    m = Motor(a.config, a)
    for etapa in (ETAPAS if a.etapa == "tudo" else [a.etapa]):
        m.executar(etapa)


if __name__ == "__main__":
    main()
