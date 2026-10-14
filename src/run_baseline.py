"""
Baseline completo no Oxford-IIIT Pet: 3 backbones x 3 regimes, com a receita padrão,
seguido do teste de LR alta no full fine-tuning dos três modelos.

Uso (da raiz do projeto, com os outros apps fechados):
    caffeinate -i venv/bin/python src/run_baseline.py

caffeinate -i impede o Mac de dormir durante o treino. Se o script for interrompido,
basta rodar de novo: os experimentos que já estão em results/summary.csv são pulados.
"""
import pandas as pd

from data import get_pets_splits, make_loaders
from models import BACKBONES, REGIMES
from train import RESULTS, ROOT, run_experiment

SEED = 42

# Teste da LR: o mesmo full fine-tuning do baseline, mudando SÓ a LR do backbone para 10x
# a receita padrão. Se a LR alta destrói as features pré-treinadas, a acurácia deve cair.
# Rodam por último: se faltar tempo, dá para interromper depois do baseline (Ctrl+C).
LR_ALTA = [("resnet18", "full", 1e-3), ("mobilenet_v2", "full", 1e-3), ("dinov2_vits14", "full", 1e-4)]


def main():
    train, val, test, classes = get_pets_splits(ROOT / "data", seed=SEED)

    summary_path = RESULTS / "summary.csv"
    done = set(pd.read_csv(summary_path)["run"]) if summary_path.exists() else set()

    runs = [(b, r, None, "baseline") for b in BACKBONES for r in REGIMES]
    runs += [(b, r, lr, "lr_alta") for b, r, lr in LR_ALTA]

    for backbone, regime, backbone_lr, experiment in runs:
        name = f"pets_{backbone}_{regime}_s{SEED}" + ("" if experiment == "baseline" else f"_{experiment}")
        if name in done:
            print(f"[{name}] já existe, pulando")
            continue
        # Loaders novos a cada experimento: assim a ordem dos batches e o aumento de dados
        # começam do mesmo ponto em todos, não importa quantos experimentos rodaram antes.
        loaders = make_loaders(train, val, test, seed=SEED)
        run_experiment(backbone, regime, loaders, num_classes=len(classes), seed=SEED,
                       backbone_lr=backbone_lr, experiment=experiment)


# Obrigatório no macOS: os processos que carregam as imagens (num_workers) reimportam
# este arquivo, e sem essa proteção cada um deles começaria o treino de novo.
if __name__ == "__main__":
    main()
