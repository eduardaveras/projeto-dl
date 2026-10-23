"""
Etapa 2: escassez controlada no Oxford-IIIT Pet.

Para cada k (imagens por classe no treino) e cada seed, sorteia um subconjunto do treino,
treina os 3 regimes (linear, partial, full) com um número fixo de passos e avalia no
conjunto de teste completo.

Uso (da raiz do projeto, com os outros apps fechados):
    caffeinate -i venv/bin/python src/fewshot.py

Se for interrompido, basta rodar de novo: o que já está em results/fewshot/summary.csv é pulado.
"""
import math
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import get_pets_splits, subsample_per_class
from models import BACKBONES, REGIMES, build_model
from train import BACKBONE_LR, RESULTS, ROOT, evaluate, get_device, make_optimizer, set_seed, train_one_epoch

KS = (5, 10, 20, 30)
SEEDS = (0, 1, 2)
# Mesmo número de passos de otimização para todo k: a única diferença entre as rodadas é
# a quantidade de dados. (Arredondado para cima até completar a última época.)
# 1000 passos: com k=30 dá 15 épocas, como no baseline; com k=5, 84 épocas.
STEPS = 1000
BATCH_SIZE = 16
HEAD_LR = 1e-3
OUT = RESULTS / "fewshot"


def train_fewshot(backbone, regime, train_set, test_loader, num_classes, seed, device):
    """
    Treina sem validação: com k imagens por classe, não há imagens sobrando para escolher a
    melhor época. Treina STEPS passos e avalia o modelo final, uma vez, no teste.
    """
    set_seed(seed)
    model = build_model(backbone, num_classes, regime).to(device)
    optimizer = make_optimizer(model, backbone, HEAD_LR, BACKBONE_LR[backbone])
    loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                        generator=torch.Generator().manual_seed(seed),
                        num_workers=2, persistent_workers=True)
    epochs = math.ceil(STEPS / len(loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history, start = [], time.time()
    for epoch in range(1, epochs + 1):
        loss, acc = train_one_epoch(model, loader, optimizer, criterion, device)
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": loss, "train_acc": acc})
    train_time = time.time() - start

    test = evaluate(model, test_loader, criterion, device)
    return pd.DataFrame(history), test, epochs * len(loader), train_time


def load_summary(path):
    """Resultados já salvos, só dos 3 regimes (versões antigas também tinham kNN e protótipo)."""
    if not path.exists():
        return []
    df = pd.read_csv(path).rename(columns={"metodo": "regime"})
    return df[df["regime"].isin(REGIMES)].to_dict("records")


def main():
    device = get_device()
    train, val, test, classes = get_pets_splits(ROOT / "data")
    test_loader = DataLoader(test, batch_size=64)

    (OUT / "csv").mkdir(parents=True, exist_ok=True)
    summary_path = OUT / "summary.csv"
    rows = load_summary(summary_path)
    done = {(r["backbone"], r["regime"], r["k"], r["seed"]) for r in rows}

    # A seed fica no laço de fora: se interromper, as seeds completas já servem para análise.
    for s in SEEDS:
        for k in KS:
            sub = subsample_per_class(train, k, s)
            for b in BACKBONES:
                for r in REGIMES:
                    if (b, r, k, s) in done:
                        continue
                    hist, test_m, steps, t = train_fewshot(b, r, sub, test_loader, len(classes), s, device)
                    hist.to_csv(OUT / "csv" / f"{b}_{r}_k{k}_s{s}.csv", index=False)
                    rows.append({"backbone": b, "regime": r, "k": k, "seed": s,
                                 "test_acc": test_m["acc"], "test_f1": test_m["f1"],
                                 "steps": steps, "train_time_s": t})
                    pd.DataFrame(rows).to_csv(summary_path, index=False)
                    print(f"[{b} {r} k={k} seed={s}] teste: acc {test_m['acc']:.3f} | "
                          f"f1 {test_m['f1']:.3f}", flush=True)


# Obrigatório no macOS: os processos que carregam imagens reimportam este arquivo.
if __name__ == "__main__":
    main()
