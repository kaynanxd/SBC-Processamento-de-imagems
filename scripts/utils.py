"""Utilitários compartilhados entre os scripts de preparação e análise."""

import json
from pathlib import Path

DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent / "Dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUMMARY_FILENAME = "verduras_resumo.json"


def listar_pastas_split(dataset_dir: Path, subpasta: str = "images") -> list[Path]:
    """Retorna as subpastas de cada split (ex.: train/images, validation/images)."""
    pastas = sorted(Path(dataset_dir).glob(f"*/{subpasta}"))
    if not pastas:
        raise FileNotFoundError(
            f"Nenhuma pasta '*/{subpasta}' encontrada em {dataset_dir}. "
            "Execute primeiro a opção 1 para baixar o dataset."
        )
    return pastas


def listar_imagens(pasta: Path) -> list[Path]:
    """Lista arquivos de imagem ordenados dentro de uma pasta."""
    return sorted(
        p for p in pasta.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def atualizar_resumo_json(dataset_dir: Path, **campos) -> Path:
    """Atualiza chaves no arquivo Dataset/verduras_resumo.json preservando as demais."""
    caminho = Path(dataset_dir) / SUMMARY_FILENAME
    resumo: dict = {}
    if caminho.exists():
        with caminho.open("r", encoding="utf-8") as f:
            resumo = json.load(f)

    resumo.update(campos)
    with caminho.open("w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    return caminho
