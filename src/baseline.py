import torch
from torchvision import datasets

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Dispositivo: {device}")

# Carrega a base de treino
treino = datasets.OxfordIIITPet(root='data', split='trainval', download=False)

# Carrega a base de teste
teste = datasets.OxfordIIITPet(root='data', split='test', download=False)

print(f"Imagens para treino: {len(treino)}")
print(f"Imagens para teste: {len(teste)}")
print(f"Total real: {len(treino) + len(teste)}")