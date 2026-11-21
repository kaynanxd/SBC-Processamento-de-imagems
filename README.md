# Segmentação de Verduras Verdes em Refeições

Projeto desenvolvido para a disciplina de **Processamento de Imagens** (tema: **Gastronomia**), com o objetivo de segmentar **verduras verdes** em fotos de refeições e estimar a proporção que elas ocupam no prato.

- **Dataset utilizado:** [FoodSeg103 no Hugging Face](https://huggingface.co/datasets/json9473/FoodSeg103) (`7.118` imagens no total: `4.983` de treino e `2.135` de validação).

---

## O que já está pronto neste repositório

Esta primeira etapa cuida do **download, organização e pré-processamento do dataset**, deixando tudo pronto para o treinamento do modelo e para a análise final de área/proporção:

- **Download e organização automática** do FoodSeg103 em splits (`train` e `validation`).
- **Conversão manual `RGB → HSV`** aplicada em todas as imagens do dataset (sem usar funções prontas de conversão).
- **Geração de máscaras binárias (`verduras verdes` vs `resto`)** focadas em **13 classes estritamente verdes**:
  `76` (`spring onion`), `77` (`rape`), `79` (`okra`), `80` (`lettuce`), `82` (`cucumber`), `85` (`asparagus`), `87` (`broccoli`), `88` (`celery stick`), `89` (`cilantro mint`), `90` (`snow peas`), `95` (`green beans`), `96` (`French beans`) e `102` (`salad`).
- **Análise descritiva da informação (cálculo manual)** em todas as imagens:
  - **Entropia de Shannon** (geral, por split e por canal `R, G, B, H, S, V` separando verduras vs resto);
  - **Assimetria (*Skewness*)** e **Curtose (*Kurtosis*)**.

---

## Como rodar na sua máquina

### Pré-requisitos
- Python `>= 3.10`
- Gerenciador de pacotes [`uv`](https://docs.astral.sh/uv/)

### Passo a passo
1. Instale as dependências do projeto:
   ```
   uv sync
   ```
2. Execute o menu interativo:
   ```
   uv run main.py
   ```
3. Escolha a opção **`1 - Baixar e preparar dataset completo`**.

---

## O que é gerado na pasta `Dataset/`

```text
Dataset/
├── id2label.json                    # Mapa oficial das 104 classes (0..103)
├── verduras_resumo.json             # Métricas consolidadas do dataset (entropia, assimetria, curtose, contagens)
├── train/                           # 4.983 fotos
│   ├── images/                      # Fotos originais em RGB (.jpg)
│   ├── hsv/                         # Fotos convertidas manualmente para HSV (.jpg)
│   ├── masks/                       # Máscaras originais multiclasse (0..103, .png)
│   ├── masks_binarias_verduras/     # Gabarito binário: 0 = resto, 255 = verdura verde (.png)
│   └── imagens_com_verduras.txt     # Nomes (sem extensão) das imagens que têm verduras verdes
└── validation/                      # 2.135 exemplos (mesma estrutura acima)
```

---

## Dicas importantes para quem vai treinar e analisar

### Para o treino do modelo
- **Entradas:** use `Dataset/<split>/images/<id>.jpg` (RGB) ou `Dataset/<split>/hsv/<id>.jpg` (HSV) inclusive seria interessante treinar nos dois e comparar os resultados.
- **Máscaras (`masks_binarias_verduras/`):**
  - Os pixels estão salvos como **`0`** (resto) e **`255`** (verdura verde) para facilitar a visualização no computador e comparacao.
- **Desbalanceamento:** apenas **~6,14%** dos pixels do treino são verduras verdes (`2.526` de `4.983` imagens têm verduras). Use `imagens_com_verduras.txt` para filtrar/balancear os *batches* e prefira uma *loss* combinada .

### Para o cálculo de área e visualização
- O gabarito de referência para comparar com as máscaras preditas pelo modelo está em `Dataset/validation/masks_binarias_verduras/`.
- Todas as estatísticas descritivas já calculadas do dataset (para usar nas tabelas do artigo) ficam em `Dataset/verduras_resumo.json`.

---

## Organização do código

- `main.py`: menu principal interativo que executa todo o pipeline de preparação.
- `scripts/download_dataset.py`: baixa e extrai o FoodSeg103 do Hugging Face.
- `scripts/rgb_to_hsv.py`: conversão manual `RGB → HSV` em paralelo.
- `scripts/binary_masks.py`: gera `masks_binarias_verduras/`, `imagens_com_verduras.txt` e contagem por verdura.
- `scripts/entropy.py`: calcula manualmente entropia, assimetria, curtose e entropia por canal (`R, G, B, H, S, V`).
- `scripts/utils.py`: helpers compartilhados de arquivos e atualização do `verduras_resumo.json`.
