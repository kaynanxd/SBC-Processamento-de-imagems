"""Cálculo de entropia de Shannon para análise do dataset."""

from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

from scripts.binary_masks import GREEN_VEGETABLE_IDS
from scripts.rgb_to_hsv import rgb_para_hsv_manual
from scripts.utils import (
    DEFAULT_DATASET_DIR,
    atualizar_resumo_json,
    listar_imagens,
    listar_pastas_split,
)


_NIVEIS_CINZA = np.arange(256, dtype=np.float64)


def _entropia_de_hist(hist: np.ndarray) -> float:
    """Calcula a entropia de Shannon (bits) a partir de um histograma."""
    total = int(hist.sum())
    if total == 0:
        return 0.0
    p = hist[hist > 0] / float(total)
    return float(-np.sum(p * np.log2(p)))


def _para_cinza(rgb_array: np.ndarray) -> np.ndarray:
    """Converte uma matriz RGB ou cinza para luminância uint8 (0..255)."""
    if rgb_array.ndim == 3 and rgb_array.shape[2] >= 3:
        r = rgb_array[:, :, 0].astype(np.float32)
        g = rgb_array[:, :, 1].astype(np.float32)
        b = rgb_array[:, :, 2].astype(np.float32)
        return np.clip(np.round(0.299 * r + 0.587 * g + 0.114 * b), 0, 255).astype(np.uint8)
    if rgb_array.ndim == 2:
        return rgb_array.astype(np.uint8)
    raise ValueError(f"Formato de imagem não suportado: {rgb_array.shape}")


def calcular_descritores_imagem(rgb_array: np.ndarray) -> tuple[float, float, float]:
    """Calcula manualmente entropia (bits), assimetria e curtose (excesso) de uma imagem."""
    gray = _para_cinza(rgb_array)
    hist = np.bincount(gray.ravel(), minlength=256)
    total = int(hist.sum())
    if total == 0:
        return 0.0, 0.0, 0.0

    p = hist / float(total)
    nz = p > 0
    entropia = float(-np.sum(p[nz] * np.log2(p[nz])))

    media = float(np.sum(_NIVEIS_CINZA * p))
    desvios = _NIVEIS_CINZA - media
    variancia = float(np.sum((desvios ** 2) * p))
    if variancia <= 1e-12:
        return entropia, 0.0, 0.0

    std = float(np.sqrt(variancia))
    z = desvios / std
    assimetria = float(np.sum((z ** 3) * p))
    curtose = float(np.sum((z ** 4) * p) - 3.0)
    return entropia, assimetria, curtose


def calcular_entropia_imagem(rgb_array: np.ndarray) -> float:
    """Calcula a entropia de Shannon (em bits) de uma imagem como um único número."""
    return calcular_descritores_imagem(rgb_array)[0]


def _resumir_descritores(valores: list[tuple[float, float, float]]) -> dict:
    """Agrega entropia, assimetria e curtose em média, desvio padrão, mín e máx."""
    arr = np.array(valores, dtype=np.float64)
    ent, assim, curt = arr[:, 0], arr[:, 1], arr[:, 2]
    return {
        "total_imagens": len(valores),
        "entropia_media_bits": round(float(ent.mean()), 6),
        "desvio_padrao_bits": round(float(ent.std()), 6),
        "min_bits": round(float(ent.min()), 6),
        "max_bits": round(float(ent.max()), 6),
        "assimetria_media": round(float(assim.mean()), 6),
        "assimetria_desvio_padrao": round(float(assim.std()), 6),
        "curtose_media": round(float(curt.mean()), 6),
        "curtose_desvio_padrao": round(float(curt.std()), 6),
    }


def calcular_entropia_dataset(dataset_dir: Path = DEFAULT_DATASET_DIR) -> float:
    """Calcula entropia, assimetria e curtose manuais por split e geral."""
    image_dirs = listar_pastas_split(dataset_dir, "images")
    todos_descritores: list[tuple[float, float, float]] = []
    por_split: dict[str, dict] = {}

    for images_dir in image_dirs:
        split_name = images_dir.parent.name
        imagens = listar_imagens(images_dir)
        if not imagens:
            continue

        descritores_split: list[tuple[float, float, float]] = []
        for img_path in tqdm(imagens, desc=f"Descritores ({split_name})", unit="img"):
            with Image.open(img_path) as img:
                arr = np.asarray(img.convert("RGB"))
            descritores_split.append(calcular_descritores_imagem(arr))

        por_split[split_name] = _resumir_descritores(descritores_split)
        todos_descritores.extend(descritores_split)

    resumo_geral = _resumir_descritores(todos_descritores)
    resumo_geral["entropia_geral_bits"] = resumo_geral.pop("entropia_media_bits")
    resumo_geral["por_split"] = por_split

    arquivo_saida = atualizar_resumo_json(dataset_dir, entropia_geral=resumo_geral)

    print(
        f"\nGeral ({resumo_geral['total_imagens']} imagens): "
        f"entropia={resumo_geral['entropia_geral_bits']:.4f} ± {resumo_geral['desvio_padrao_bits']:.4f} bits | "
        f"assimetria={resumo_geral['assimetria_media']:.4f} | "
        f"curtose={resumo_geral['curtose_media']:.4f}"
    )
    for split_name, info in por_split.items():
        print(
            f"  - {split_name} ({info['total_imagens']} imgs): "
            f"entropia={info['entropia_media_bits']:.4f} bits | "
            f"assimetria={info['assimetria_media']:.4f} | "
            f"curtose={info['curtose_media']:.4f}"
        )
    print(f"Resultado salvo em: {arquivo_saida}")
    return resumo_geral["entropia_geral_bits"]


def calcular_entropia_canais_verduras(dataset_dir: Path = DEFAULT_DATASET_DIR) -> dict:
    """Compara a entropia por canal (R,G,B e H,S,V) entre verduras verdes e o resto."""
    dataset_dir = Path(dataset_dir)
    image_dirs = listar_pastas_split(dataset_dir, "images")

    canais = ("R", "G", "B", "H", "S", "V")
    hist_verdura = {c: np.zeros(256, dtype=np.int64) for c in canais}
    hist_resto = {c: np.zeros(256, dtype=np.int64) for c in canais}
    imagens_analisadas = 0

    for images_dir in image_dirs:
        split_dir = images_dir.parent
        masks_dir = split_dir / "masks"
        bin_dir = split_dir / "masks_binarias_verduras"
        hsv_dir = split_dir / "hsv"

        for img_path in tqdm(listar_imagens(images_dir), desc=f"Entropia canais ({split_dir.name})", unit="img"):
            bin_path = bin_dir / f"{img_path.stem}.png"
            mask_path = masks_dir / f"{img_path.stem}.png"

            if bin_path.exists():
                with Image.open(bin_path) as m:
                    is_veg = np.asarray(m) > 0
            elif mask_path.exists():
                with Image.open(mask_path) as m:
                    m_arr = np.asarray(m)
                    is_veg = np.isin(m_arr[:, :, 0] if m_arr.ndim == 3 else m_arr, list(GREEN_VEGETABLE_IDS))
            else:
                continue

            if not np.any(is_veg):
                continue

            with Image.open(img_path) as img:
                rgb_arr = np.asarray(img.convert("RGB"))

            hsv_path = hsv_dir / f"{img_path.stem}.jpg"
            if not hsv_path.exists():
                hsv_path = hsv_dir / f"{img_path.stem}.png"
            if hsv_path.exists():
                with Image.open(hsv_path) as h_img:
                    hsv_arr = np.asarray(h_img.convert("RGB"))
            else:
                hsv_arr = rgb_para_hsv_manual(rgb_arr)

            not_veg = ~is_veg
            for idx, c in enumerate(("R", "G", "B")):
                hist_verdura[c] += np.bincount(rgb_arr[:, :, idx][is_veg], minlength=256)
                hist_resto[c] += np.bincount(rgb_arr[:, :, idx][not_veg], minlength=256)

            for idx, c in enumerate(("H", "S", "V")):
                hist_verdura[c] += np.bincount(hsv_arr[:, :, idx][is_veg], minlength=256)
                hist_resto[c] += np.bincount(hsv_arr[:, :, idx][not_veg], minlength=256)

            imagens_analisadas += 1

    if imagens_analisadas == 0:
        raise FileNotFoundError("Nenhuma máscara com verduras encontrada.")

    resultado = {
        "imagens_com_verduras_analisadas": imagens_analisadas,
        "verduras_verdes_bits": {c: round(_entropia_de_hist(hist_verdura[c]), 6) for c in canais},
        "resto_bits": {c: round(_entropia_de_hist(hist_resto[c]), 6) for c in canais},
    }

    arquivo_saida = atualizar_resumo_json(dataset_dir, entropia_por_canal=resultado)

    print(f"\n=== Entropia por canal ({imagens_analisadas} imagens com verduras) ===")
    for c in canais:
        print(
            f"  {c}: verduras={resultado['verduras_verdes_bits'][c]:.4f} bits | "
            f"resto={resultado['resto_bits'][c]:.4f} bits"
        )
    print(f"Resultado salvo em: {arquivo_saida}")
    return resultado


if __name__ == "__main__":
    calcular_entropia_dataset()
