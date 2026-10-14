# gatinhas-dl

**Reconhecimento de indivíduos (gatas) com poucos dados: CNNs vs. ViT em diferentes regimes de especialização**

Projeto da disciplina de Deep Learning 
---

## 1. Ideia geral

Identificar qual das minhas gatas aparece numa câmera fixa ("Gata Branca", "Gata com Bolinhas Pretas" ou "Vazio") usando apenas **20–30 imagens por classe**.

O problema é de **reconhecimento de indivíduos** (distinguir dois animais da mesma espécie), não de classificação genérica. Por isso, o projeto compara como diferentes backbones pré-treinados e diferentes níveis de especialização se comportam quando os dados são escassos.

### Pergunta de pesquisa

> A melhor forma de especializar uma rede pré-treinada muda quando os dados ficam escassos? E isso depende do backbone (CNN supervisionada vs. ViT auto-supervisionado)?

### Hipóteses

- **H1:** Com muitos dados (Pets completo), o full fine-tuning é o melhor regime. Com poucos dados (Pets few-shot e gatas), o linear probing e o kNN passam a ganhar.
- **H2:** No full fine-tuning com k muito pequeno, as CNNs (ResNet18, MobileNetV2) sofrem menos overfitting que o ViT, por causa do viés indutivo convolucional.
- **H3:** No linear probing e no kNN, o ViT-S/DINOv2 supera as CNNs, porque suas features congeladas já separam bem instâncias parecidas.
- **H4 (qualitativa):** Com câmera fixa, parte dos acertos pode vir de atalhos (fundo, posição, iluminação). O Grad-CAM e os mapas de atenção devem revelar isso.

---

## 2. Etapas

```
Pets completo (~3.7k)  →  Pets com k imagens/classe  →  Gatinhas (~20–30/classe)
 [BASELINE, 20/10]          [INOVAÇÃO: escassez         [INOVAÇÃO: problema real]
                             controlada]
 mesmos 3 regimes ──────────── mesmos 3 regimes ────────── mesmos 3 regimes
```

### Etapa 1: Baseline

Não há um artigo base com resultados no Oxford Pets para as CNNs (os artigos da ResNet e da MobileNetV2 reportam ImageNet). Por isso, a etapa de "implementar sem modificações" é feita assim:

1. **Reprodução:** cada modelo é treinado no Pets completo com a **receita padrão** dele, fixada antes de rodar e igual para todos:
   - CNNs: full fine-tuning.
   - DINOv2: linear probing, como no artigo original, que reporta resultado no Pets *(número a confirmar)*.
2. **Extra:** a tabela completa de 3 modelos × 3 regimes no Pets completo. Ela serve de **referência com bastante dado** para comparar com as etapas de escassez.

Sem busca extensa de hiperparâmetros: no máximo uma varredura pequena e igual para todos (ex.: 3 valores de LR), escolhida pela **validação**, nunca pelo test.

### Etapa 2: Inovação, escassez controlada (Pets few-shot)

Os mesmos 3 regimes e o kNN, com k ∈ {5, 10, 20, 30} imagens por classe e 3 seeds. Isso mostra em que quantidade de dados cada regime passa a compensar.

### Etapa 3: Inovação, as gatas

1. **Coleta:** vários dias e horários diferentes, com frames espaçados (nada de sequências quase idênticas). Organização: `data/gatinhas/<sessao>/<classe>/img.jpg`.
2. **Divisão: leave-one-session-out.** Cada fold testa numa sessão que o modelo nunca viu. A pergunta respondida é: *o modelo reconhece a gata num dia novo?*
3. **Treino:** o mesmo código do Pets, trocando só o dataset (cabeça com 3 saídas) e com várias seeds.
4. **Avaliação:** métricas por classe (principalmente Gata Branca vs. Bolinhas), gap de generalização e Grad-CAM e mapas de atenção.

**De onde parte o modelo das gatas?**
- **A (principal):** direto dos pesos ImageNet/DINOv2, os mesmos de partida do Pets. A comparação fica limpa, porque só muda o dataset.
- **B (extra, se der tempo):** ImageNet → Pets → gatas. *Aprender raças antes ajuda a reconhecer indivíduos?* Pode ajudar (features de gato) ou atrapalhar (a rede aprendeu a ignorar diferenças dentro de uma raça). Para deixar essa porta aberta, os checkpoints do Pets são salvos já no baseline.

---

## 3. Abordagem

### Modelos (backbones)

| Modelo | Params | Pré-treino | Por quê |
|---|---|---|---|
| ResNet18 | ~11,7M | ImageNet supervisionado | CNN clássica (He et al., 2016) |
| MobileNetV2 | ~3,5M | ImageNet supervisionado | CNN eficiente (Sandler et al., 2018) |
| ViT-S/14 (DINOv2) | ~21M | Auto-supervisionado (LVD-142M) | Sem viés convolucional; features genéricas fortes (Oquab et al., 2023) |

### Regimes de especialização

Em todos os regimes, a última camada é **substituída** por uma nova (37 saídas no Pets, 3 nas gatas), que começa aleatória e sempre é treinada. O que muda é quanto do resto da rede também treina:

| Regime | Camada nova | Resto da rede | Definição por modelo |
|---|---|---|---|
| **kNN** (baseline sem treino) | — | congelado | kNN sobre as features congeladas |
| **Linear probing** | treina | congelado | backbone com `requires_grad = False` |
| **Partial fine-tuning** | treina | último quarto dos blocos | ResNet18: `layer4` (2 de 8 blocos, 75% dos params) · MobileNetV2: `features[14:]` (4 de 17 blocos + conv final, 76%) · ViT: últimos 3 de 12 blocos + norm (25%) |
| **Full fine-tuning** | treina | tudo treina | LR baixa (ou LR discriminativa) para não destruir as features pré-treinadas |

### Receita padrão (igual para todos os modelos e regimes)

- AdamW, weight decay `1e-4`, com duas LRs:
  - cabeça nova: `1e-3` (começa aleatória e precisa aprender do zero)
  - backbone (partes descongeladas): `1e-4` nas CNNs e `1e-5` no ViT. Com `1e-4`, o full fine-tuning do DINOv2 ficou no nível do chute (~3%); com `1e-5`, chegou a 82% de validação em 2 épocas num subconjunto
- Scheduler cosine, 15 épocas, batch 16 (com 32, o ViT em full fine-tuning não cabe nos 8 GB do M1)
- Treino: `RandomResizedCrop(224)` + flip horizontal · Avaliação: `Resize(256)` + `CenterCrop(224)`
- Melhor checkpoint escolhido pela validação; o teste é avaliado uma única vez, com esse checkpoint

**Hardware:** MacBook M1, 8 GB de RAM (MPS). Tempo estimado do baseline completo (9 rodadas + 2 de LR alta): ~4 h. Fechar outros apps durante o treino para evitar swap.

### Protocolo experimental

- **Divisões:** no Pets, o `trainval` é dividido em treino e validação, e o `test` é usado **só** no final. Nas gatas, a divisão é leave-one-session-out.
- **Seeds:** 1 seed no baseline (Pets completo). 3 seeds nas etapas de escassez. Reportar média ± desvio.
- **Logs:** métricas por época salvas em CSV (`results/csv/`) para análise em Pandas/Seaborn.
- **Estatística:** teste pareado (Wilcoxon ou t pareado) entre configurações.

### Métricas

- Acurácia e **F1 macro**, além das métricas por classe nas gatas (a classe "Vazio" é fácil e infla a média).
- Gap de generalização (acurácia de treino − acurácia de validação) como medida de overfitting.
- Tempo de inferência (batch = 1, com warm-up, `torch.mps.synchronize()`, em MPS e CPU), nº de parâmetros e FLOPs.
- Qualitativo: Grad-CAM (CNNs) e mapas de atenção (ViT).

---

## 4. Resultados esperados

- **Baseline (Pets completo):** CNNs com a receita padrão na faixa de ~90% de acurácia. O DINOv2 em linear probing deve ficar próximo do valor do artigo. Na tabela 3×3, o full fine-tuning deve ser o melhor ou empatar.
- **Pets few-shot:** curvas acurácia × k com um "ponto de cruzamento" entre regimes. O DINOv2 deve dominar com k pequeno em linear probing e kNN.
- **Gatinhas:** a ordem dos regimes deve se inverter em relação ao Pets completo, com linear probing e kNN competitivos ou melhores. O Grad-CAM deve indicar se o modelo olha para a gata ou para o fundo.
- **Custo-benefício:** a MobileNetV2 deve ter o menor tempo de inferência. O trade-off acurácia × latência será mostrado num gráfico.

### Resultados do baseline (acurácia no teste, %)

| Backbone | Linear | Parcial | Completo |
|---|---|---|---|
| ResNet18 | 89,0 | 89,8 | 89,7 |
| MobileNetV2 | 90,5 | 90,8 | 90,3 |
| ViT-S DINOv2 | 94,7 | 95,2 | 94,5 |

- Com dados abundantes, o regime quase não importa (≤ 0,8 ponto de diferença); o backbone pesa muito mais.
- O DINOv2 linear (94,7%) fica 0,4 ponto abaixo do valor do artigo (95,1%), o que valida a replicação.
- Nas CNNs, o ajuste das camadas ajuda nas raças de gatos (ResNet18: 82,1% → 88,0%) e piora levemente nas de cães.

### Teste da taxa de aprendizado

Uma primeira rodada exploratória (ResNet18 com Adam e `lr=1e-3` na rede inteira, sem validação separada) ficou em ~61% de acurácia, bem abaixo do esperado. A hipótese é que uma LR alta no full fine-tuning destrói as features pré-treinadas. Para testar isso de forma controlada, o `run_baseline.py` repete o full fine-tuning mudando **só** a LR do backbone para 10× a receita padrão:

| Modelo | Receita padrão | LR alta |
|---|---|---|
| ResNet18 | `1e-4` | `1e-3` |
| MobileNetV2 | `1e-4` | `1e-3` |
| ViT-S DINOv2 | `1e-5` | `1e-4` |

Os resultados ficam marcados como `experiment = lr_alta` em `results/summary.csv` e aparecem na seção 3.4 do notebook.

---

## 5. Estrutura do repositório

O código é construído **do zero e passo a passo**. As funções reutilizáveis ficam em `src/`, e os notebooks só chamam essas funções e mostram o que está acontecendo (imagens, shapes, curvas, tabelas).

```
gatinhas-dl/
├── data/                    # datasets (ignorado pelo git)
│   ├── oxford-iiit-pet/
│   └── gatinhas/            # (a criar) <sessao>/<classe>/img.jpg
├── src/
│   ├── check_env.py         # verifica o dispositivo (MPS) e o tamanho do dataset
│   ├── data.py              # transforms, divisão treino/val/teste, DataLoaders
│   ├── models.py            # backbone pré-treinado + cabeça nova + regime (o que congela)
│   ├── train.py             # loop de treino/avaliação, CSV por época, checkpoint, teste
│   └── run_baseline.py      # roda as 9 combinações no Pets (no terminal)
├── notebooks/
│   └── 01_baseline.ipynb    # dados, modelos e análise dos resultados do baseline
├── checkpoints/             # modelos salvos (ignorado pelo git)
└── results/                 # summary.csv, csv/ (por época), preds/ (teste), figures/
```

### Ambiente

Python 3.12 (Homebrew). O DINOv2 é carregado pela `timm` (`vit_small_patch14_dinov2.lvd142m`), com os mesmos pesos oficiais: ela já cria a cabeça com o número de classes certo e ajusta a resolução de entrada para 224 numa única chamada.

```bash
/opt/homebrew/bin/python3.12 -m venv venv
source venv/bin/activate
pip install torch torchvision timm scikit-learn pandas matplotlib seaborn ipykernel
python src/check_env.py          # confere MPS e dataset
```

Versões usadas: torch 2.14.1, torchvision 0.29.1, timm 1.0.30.

Notebooks: abrir em `notebooks/` no VS Code com o kernel do `venv`.

### Rodar o baseline (~4 h)

No terminal, da raiz do projeto, com os outros apps fechados:

```bash
caffeinate -i venv/bin/python src/run_baseline.py
```

Se for interrompido, é só rodar de novo: os experimentos que já estão em `results/summary.csv` são pulados. Depois, a seção 3 do notebook lê os resultados e gera tabela, curvas e matriz de confusão.

---

## Referências

- K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016.
- M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, L. Chen. *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018.
- A. Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.
- M. Oquab et al. *DINOv2: Learning Robust Visual Features without Supervision.* TMLR 2024.
- S. Kornblith, J. Shlens, Q. V. Le. *Do Better ImageNet Models Transfer Better?* CVPR 2019.
- O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar. *Cats and Dogs.* CVPR 2012 (Oxford-IIIT Pet).
