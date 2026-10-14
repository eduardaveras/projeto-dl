"""
Treino e avaliação: loop por época, métricas em CSV, checkpoint pela validação e teste final.
"""
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from models import HEADS, build_model, count_params, train_mode

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
CHECKPOINTS = ROOT / "checkpoints"

# LR do backbone (partes descongeladas) por modelo. O ViT precisa de uma LR 10x menor:
# com 1e-4, o full fine-tuning do DINOv2 destruiu as features e ficou no nível do chute.
BACKBONE_LR = {"resnet18": 1e-4, "mobilenet_v2": 1e-4, "dinov2_vits14": 1e-5}


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed):
    """Fixa as fontes de aleatoriedade (inicialização da cabeça, dropout, aumento de dados)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_optimizer(model, backbone, head_lr, backbone_lr, weight_decay=1e-4):
    """
    AdamW com duas taxas de aprendizado: uma maior para a cabeça, que começa aleatória e
    precisa aprender do zero, e uma menor para o backbone, que já é bom e só deve ser ajustado.
    Só entram no otimizador os parâmetros que o regime deixou treináveis.
    """
    head_params = list(getattr(model, HEADS[backbone]).parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]

    groups = [{"params": head_params, "lr": head_lr}]
    if backbone_params:  # no linear probing esta lista é vazia
        groups.append({"params": backbone_params, "lr": backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def train_one_epoch(model, loader, optimizer, criterion, device):
    train_mode(model)
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()           # zera os gradientes do batch anterior
        logits = model(x)               # forward: uma pontuação por classe
        loss = criterion(logits, y)     # o quanto as pontuações erraram
        loss.backward()                 # backward: calcula os gradientes
        optimizer.step()                # atualiza os pesos treináveis

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


@torch.no_grad()  # avaliação não precisa de gradientes: mais rápido e usa menos memória
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds, labels = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * len(y)
        preds.append(logits.argmax(1).cpu())
        labels.append(y.cpu())
    preds, labels = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return {
        "loss": total_loss / len(labels),
        "acc": (preds == labels).mean(),
        "f1": f1_score(labels, preds, average="macro"),  # média simples entre classes
        "preds": preds,
        "labels": labels,
    }


def run_experiment(backbone, regime, loaders, num_classes, dataset="pets",
                   epochs=15, head_lr=1e-3, backbone_lr=None, seed=42, experiment="baseline"):
    """
    Treina um modelo, salva as métricas de cada época em CSV e o melhor checkpoint pela
    validação. No final, carrega esse checkpoint e avalia UMA vez no teste.

    experiment separa variações da receita (ex.: "lr_alta") do baseline: entra no nome dos
    arquivos, para não sobrescrever o experimento com a receita padrão.
    """
    backbone_lr = backbone_lr or BACKBONE_LR[backbone]
    train_loader, val_loader, test_loader = loaders
    name = f"{dataset}_{backbone}_{regime}_s{seed}"
    if experiment != "baseline":
        name += f"_{experiment}"
    device = get_device()
    (RESULTS / "csv").mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(exist_ok=True)
    ckpt_path = CHECKPOINTS / f"{name}.pt"

    set_seed(seed)  # antes de criar o modelo, para a cabeça nova nascer sempre igual
    model = build_model(backbone, num_classes, regime).to(device)
    optimizer = make_optimizer(model, backbone, head_lr, backbone_lr)
    # Cosine: a LR começa no valor escolhido e cai suavemente até ~0 na última época,
    # dando passos grandes no início e ajustes finos no final.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history, best_acc = [], -1.0
    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val["loss"], "val_acc": val["acc"], "val_f1": val["f1"],
            "time_s": time.time() - start,
        })
        # Reescreve o CSV a cada época: se o treino for interrompido, o que já rodou fica salvo.
        pd.DataFrame(history).to_csv(RESULTS / "csv" / f"{name}.csv", index=False)

        if val["acc"] > best_acc:
            best_acc = val["acc"]
            torch.save(model.state_dict(), ckpt_path)
        print(f"[{name}] época {epoch:2d}/{epochs} | treino: loss {train_loss:.3f} acc {train_acc:.3f}"
              f" | val: loss {val['loss']:.3f} acc {val['acc']:.3f} | {time.time() - start:.0f}s")

    # Teste: só com o melhor modelo da validação, uma única vez.
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test = evaluate(model, test_loader, criterion, device)
    # Predições do teste, para a matriz de confusão e a análise por classe.
    (RESULTS / "preds").mkdir(exist_ok=True)
    pd.DataFrame({"label": test["labels"], "pred": test["preds"]}).to_csv(
        RESULTS / "preds" / f"{name}.csv", index=False
    )
    total, trainable = count_params(model)
    hist = pd.DataFrame(history)
    summary = {
        "run": name, "experiment": experiment, "dataset": dataset,
        "backbone": backbone, "regime": regime, "seed": seed,
        "epochs": epochs, "head_lr": head_lr, "backbone_lr": backbone_lr,
        "best_epoch": int(hist["val_acc"].idxmax()) + 1, "best_val_acc": best_acc,
        "test_acc": test["acc"], "test_f1": test["f1"],
        "params_total": total, "params_trainable": trainable,
        "train_time_s": hist["time_s"].sum(),
    }
    # Uma linha por experimento em results/summary.csv (substitui se o mesmo run rodar de novo).
    summary_path = RESULTS / "summary.csv"
    old = pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame()
    if not old.empty:
        old = old[old["run"] != name]
    pd.concat([old, pd.DataFrame([summary])], ignore_index=True).to_csv(summary_path, index=False)

    print(f"[{name}] teste: acc {test['acc']:.3f} | f1 {test['f1']:.3f}")
    return hist, summary, test
