# Segmentação de Verduras Verdes

Projeto de Processamento de Imagens focado na segmentação semântica binária (**verduras verdes vs resto**) utilizando o dataset **FoodSeg103** (`json9473/FoodSeg103`).

## Pré-requisitos
- Python `>= 3.10`
- Gerenciador [`uv`](https://docs.astral.sh/uv/)

## Como configurar e baixar o dataset
1. Instale as dependências:
   ```
   uv sync
   ```
2. Execute o menu principal e escolha a opção `1`:
   ```
   uv run main.py
   ```

## Estrutura gerada em `Dataset/`
```text
Dataset/
├── id2label.json                    # Mapa oficial de classes (0..103)
├── verduras_resumo.json             # Entropia geral/por split, estatísticas, classes e entropia por canal
├── train/
│   ├── images/                      # Imagens RGB originais (.jpg)
│   ├── hsv/                         # Imagens convertidas manualmente para HSV (.jpg)
│   ├── masks/                       # Máscaras originais multiclasse (0..103)
│   ├── masks_binarias_verduras/     # Máscaras binárias (0 = resto, 255 = verdura verde)
│   └── imagens_com_verduras.txt     # Lista de IDs que contêm verduras verdes
└── validation/
    ├── images/
    ├── hsv/
    ├── masks/
    ├── masks_binarias_verduras/
    └── imagens_com_verduras.txt
```

## Guia rápido
- **Entradas:** `Dataset/<split>/images/` (RGB) ou `Dataset/<split>/hsv/` (HSV).
- **Gabarito (*ground truth*):** `Dataset/<split>/masks_binarias_verduras/<id>.png` os pixels são **`0`** (resto) ou **`255`** (verdura verde); normalize dividindo por `255` (ou `mask > 0`) no `DataLoader`.
- **Classes incluídas como verduras verdes (13 IDs):** `76` (`spring onion`), `77` (`rape`), `79` (`okra`), `80` (`lettuce`), `82` (`cucumber`), `85` (`asparagus`), `87` (`broccoli`), `88` (`celery stick`), `89` (`cilantro mint`), `90` (`snow peas`), `95` (`green beans`), `96` (`French beans`), `102` (`salad`).

## Estrutura de scripts
- `main.py`: menu interativo que executa todo o pipeline de preparação.
- `scripts/download_dataset.py`: download e extração do FoodSeg103 do Hugging Face.
- `scripts/rgb_to_hsv.py`: conversão manual RGB → HSV em paralelo.
- `scripts/binary_masks.py`: geração das máscaras binárias de verduras verdes e contagem por classe.
- `scripts/entropy.py`: cálculo de entropia de Shannon (geral, por split e por canal RGB/HSV).
- `scripts/utils.py`: utilitários compartilhados de arquivos e JSON.
