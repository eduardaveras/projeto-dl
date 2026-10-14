"""
Modelos: backbone pré-treinado + cabeça nova + regime de especialização.
"""
import timm
import torch.nn as nn
from torchvision import models

BACKBONES = ("resnet18", "mobilenet_v2", "dinov2_vits14")
REGIMES = ("linear", "partial", "full")
# Nome do atributo onde cada modelo guarda a cabeça (última camada).
HEADS = {"resnet18": "fc", "mobilenet_v2": "classifier", "dinov2_vits14": "head"}


def _create(name, num_classes):
    """
    Carrega o backbone pré-treinado e troca a última camada por uma nova com num_classes saídas.

    Retorna (modelo, cabeça, últimos blocos). Os "últimos blocos" são o último quarto
    dos blocos da rede, que é o que o partial fine-tuning libera para treinar.
    """
    if name == "resnet18":
        # IMAGENET1K_V1: pesos treinados com a receita original do artigo.
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        # 8 blocos residuais em 4 estágios; layer4 = os 2 últimos blocos.
        return model, model.fc, [model.layer4]

    if name == "mobilenet_v2":
        # IMAGENET1K_V1 pelo mesmo motivo (o DEFAULT do torchvision é uma receita mais nova).
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # classifier = [Dropout, Linear]; só a Linear é trocada.
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        # features[0] é a convolução inicial, features[1:18] são 17 blocos e features[18] é a
        # convolução final. features[14:] = os 4 últimos blocos + a convolução final.
        return model, model.classifier, [model.features[14:]]

    if name == "dinov2_vits14":
        # img_size=224: o DINOv2 foi treinado em 518 px; a timm adapta os embeddings de posição.
        # A cabeça classifica a partir do token [CLS], o resumo da imagem inteira.
        model = timm.create_model(
            "vit_small_patch14_dinov2.lvd142m", pretrained=True, img_size=224, num_classes=num_classes
        )
        # 12 blocos de transformer; os 3 últimos + a normalização final.
        return model, model.head, [model.blocks[-3:], model.norm]

    raise ValueError(f"backbone desconhecido: {name}")


def build_model(name, num_classes, regime):
    """Monta o modelo e define quais parâmetros treinam, de acordo com o regime."""
    if regime not in REGIMES:
        raise ValueError(f"regime desconhecido: {regime}")
    model, head, last_blocks = _create(name, num_classes)

    if regime == "full":
        return model  # tudo já vem com requires_grad=True

    # Congela tudo e depois libera só o que o regime treina.
    # requires_grad=False: o otimizador não recebe gradiente para esses pesos, então eles não mudam.
    for p in model.parameters():
        p.requires_grad = False
    unfrozen = [head] + (last_blocks if regime == "partial" else [])
    for module in unfrozen:
        for p in module.parameters():
            p.requires_grad = True
    return model


def train_mode(model):
    """
    Coloca o modelo em modo de treino, mas mantém as BatchNorm congeladas em modo de avaliação.

    requires_grad=False impede que os pesos mudem, mas NÃO impede que a BatchNorm atualize
    suas estatísticas (média e variância) em model.train(). Sem isso, um backbone "congelado"
    continuaria mudando por dentro. (O ViT usa LayerNorm, que não tem esse problema.)
    """
    model.train()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def count_params(model):
    """Total de parâmetros e quantos deles vão ser treinados."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
