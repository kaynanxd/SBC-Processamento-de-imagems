"""Script para baixar e preparar o dataset FoodSeg103."""

from pathlib import Path
import shutil

from scripts.utils import DEFAULT_DATASET_DIR

REPO_ID = "json9473/FoodSeg103"


def _salvar_mascara(mask, destino: Path) -> None:
    """Salva a máscara preservando os IDs exatos das classes (0..103) em PNG."""
    if mask.mode in ("L", "P", "1"):
        mask.save(destino, format="PNG")
    elif mask.mode in ("RGB", "RGBA"):
        mask.getchannel(0).save(destino, format="PNG")
    else:
        mask.convert("L").save(destino, format="PNG")


def _salvar_imagem(image, destino: Path) -> None:
    """Salva a imagem RGB em JPEG."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image.save(destino, format="JPEG", quality=95)


def baixar_e_preparar_dataset(dataset_dir: Path = DEFAULT_DATASET_DIR) -> None:
    """Baixa o FoodSeg103 do Hugging Face e organiza em imagens e máscaras."""
    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
        from tqdm import tqdm
    except ImportError as exc:
        raise RuntimeError(
            "Dependências ausentes. Execute: uv sync"
        ) from exc

    dataset_dir = Path(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    print(f"Baixando id2label.json de {REPO_ID}...")
    id2label_cache = hf_hub_download(
        repo_id=REPO_ID,
        filename="id2label.json",
        repo_type="dataset",
    )
    shutil.copy2(id2label_cache, dataset_dir / "id2label.json")

    print(f"Carregando dataset {REPO_ID}...")
    dataset = load_dataset(REPO_ID)

    for split in dataset.keys():
        split_data = dataset[split]
        images_dir = dataset_dir / split / "images"
        masks_dir = dataset_dir / split / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        salvos = 0
        ignorados = 0

        for idx, sample in enumerate(
            tqdm(split_data, desc=f"Preparando '{split}'", unit="img")
        ):
            sample_id = sample.get("id", idx)
            img_path = images_dir / f"{sample_id}.jpg"
            mask_path = masks_dir / f"{sample_id}.png"

            if img_path.exists() and mask_path.exists():
                ignorados += 1
                continue

            _salvar_imagem(sample["image"], img_path)
            _salvar_mascara(sample["label"], mask_path)
            salvos += 1

        print(
            f"Split '{split}': {salvos} salvos, {ignorados} já existentes - "
            f"{dataset_dir / split}"
        )

    print(f"\nDataset pronto em: {dataset_dir}")


if __name__ == "__main__":
    baixar_e_preparar_dataset()
