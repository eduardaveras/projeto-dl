"""
Dados: transforms, divisões treino/validação/teste e DataLoaders.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

# Média e desvio padrão por canal (R, G, B) das imagens do ImageNet.
# Os modelos pré-treinados aprenderam com imagens normalizadas por esses valores,
# então as nossas precisam passar pela mesma normalização. O DINOv2 usa os mesmos.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_transforms(train, img_size=224):
    """Transforms de treino (com aumento de dados) ou de avaliação (determinística)."""
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    if train:
        # A cada época, cada imagem vira uma versão diferente: um recorte aleatório
        # (área e proporção variáveis) redimensionado para img_size, às vezes espelhado.
        # Isso funciona como "mais dados" e reduz o overfitting.
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    # Avaliação: sempre o mesmo resultado para a mesma imagem.
    # Redimensiona o lado menor para 256 (mantendo a proporção) e recorta o centro 224x224.
    # Diferente de Resize((224, 224)), não achata nem estica o animal.
    return transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])


def get_pets_splits(root="data", val_frac=0.2, seed=42, img_size=224):
    """
    Oxford-IIIT Pet dividido em treino / validação / teste.

    O split oficial só tem 'trainval' e 'test'. Separamos uma parte do 'trainval'
    como validação para escolher épocas e hiperparâmetros sem olhar o teste.
    """
    root = str(root)

    # O 'trainval' é carregado duas vezes, uma com cada transform: as imagens de
    # treino recebem aumento de dados, e as de validação não. Os dois objetos
    # apontam para os mesmos arquivos; os índices abaixo decidem quem vai para onde.
    trainval_aug = datasets.OxfordIIITPet(root, split="trainval", transform=get_transforms(True, img_size))
    trainval_eval = datasets.OxfordIIITPet(root, split="trainval", transform=get_transforms(False, img_size))
    test = datasets.OxfordIIITPet(root, split="test", transform=get_transforms(False, img_size))

    # _labels é atributo interno do torchvision, mas evita abrir as 3.680 imagens só para ler o rótulo.
    labels = trainval_aug._labels
    # stratify: cada raça mantém a mesma proporção no treino e na validação.
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)), test_size=val_frac, stratify=labels, random_state=seed
    )

    train = Subset(trainval_aug, train_idx)
    val = Subset(trainval_eval, val_idx)
    return train, val, test, trainval_aug.classes


def subsample_per_class(subset, k, seed):
    """
    Sorteia k imagens de cada classe de um Subset (usado na etapa de escassez controlada).

    O sorteio é aninhado: com a mesma seed, as 5 imagens de k=5 estão entre as 10 de k=10,
    e assim por diante. Isso acontece porque cada classe é embaralhada sempre na mesma ordem
    e pegamos as k primeiras. Assim, a diferença entre dois valores de k vem só das imagens
    a mais, e não de um sorteio totalmente diferente.
    """
    indices = np.asarray(subset.indices)
    labels = np.asarray(subset.dataset._labels)[indices]
    rng = np.random.default_rng(seed)
    chosen = []
    for c in np.unique(labels):
        pos = np.flatnonzero(labels == c)
        rng.shuffle(pos)
        chosen.extend(indices[pos[:k]])
    return Subset(subset.dataset, sorted(chosen))


GATINHAS_CLASSES = ("branca", "bolas", "vazio")


def load_gatinhas(root="data"):
    """
    Tabela do dataset das gatas, com uma linha por foto: sessão, classe, arquivo, hora,
    condição (dia/noite/ir), caminho do recorte e rótulo numérico.

    Lê os selecao.csv de cada dia (em gatinhas_raw), que já dizem quais fotos entraram e em
    que condição; as imagens em si são os recortes salvos pelo prepare_gatinhas.py.
    """
    root = Path(root)
    tables = [pd.read_csv(sel).assign(sessao=sel.parent.name)
              for sel in (root / "gatinhas_raw").glob("*/selecao.csv")]
    df = pd.concat(tables, ignore_index=True)
    df["path"] = [root / "gatinhas" / s / c / a for s, c, a in zip(df["sessao"], df["classe"], df["arquivo"])]
    df["label"] = df["classe"].map(GATINHAS_CLASSES.index)
    # Ordena as sessões pela data (o nome da pasta é DD-MM-AAAA, que não ordena como texto).
    data = pd.to_datetime(df["sessao"], format="%d-%m-%Y")
    return df.assign(data=data).sort_values(["data", "classe", "arquivo"]).drop(columns="data").reset_index(drop=True)


class GatinhasDataset(Dataset):
    """Dataset do PyTorch a partir de (um pedaço da) tabela do load_gatinhas."""

    def __init__(self, table, transform):
        self.paths = list(table["path"])
        self.labels = list(table["label"])
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]


def make_loaders(train, val, test, batch_size=16, num_workers=2, seed=42):
    """
    DataLoaders: entregam os dados em lotes (batches) para o modelo.

    batch_size=16 porque, no M1 com 8 GB, o full fine-tuning do ViT com 32 não cabe na memória.
    O mesmo valor é usado em todos os modelos, para a comparação ser justa.
    """
    # O gerador com seed fixa a ordem do embaralhamento, para o treino ser reproduzível.
    g = torch.Generator().manual_seed(seed)
    # num_workers > 0 carrega imagens em paralelo enquanto o modelo treina.
    # persistent_workers evita recriar esses processos a cada época (lento no macOS).
    # Só o treino usa workers: cada worker carrega o PyTorch inteiro na memória, e no Mac de
    # 8 GB não vale gastar isso com validação e teste, que são menores e mais rápidos.
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, generator=g,
                              num_workers=num_workers, persistent_workers=num_workers > 0)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def denormalize(img):
    """Desfaz a normalização de um tensor (C, H, W) para exibir com matplotlib (H, W, C)."""
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    return np.clip(img, 0, 1)
