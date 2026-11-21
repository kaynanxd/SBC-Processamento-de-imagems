"""Conversão manual de imagens RGB para o espaço de cores HSV."""

from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

from scripts.utils import DEFAULT_DATASET_DIR, listar_imagens, listar_pastas_split


def rgb_para_hsv_manual(rgb_array: np.ndarray) -> np.ndarray:
    """Converte uma matriz RGB (H, W, 3) uint8 para HSV (H, W, 3) uint8 manualmente."""
    if rgb_array.ndim != 3 or rgb_array.shape[2] != 3:
        raise ValueError(f"Esperado array (H, W, 3), recebido {rgb_array.shape}")

    r, g, b = rgb_array[:, :, 0], rgb_array[:, :, 1], rgb_array[:, :, 2]
    c_max = rgb_array.max(axis=2)
    c_min = rgb_array.min(axis=2)
    delta = c_max - c_min

    h = np.zeros(c_max.shape, dtype=np.float32)
    nonzero_delta = delta > 0
    if np.any(nonzero_delta):
        delta_f = delta[nonzero_delta].astype(np.float32)
        c_max_nz = c_max[nonzero_delta]
        r_nz = r[nonzero_delta].astype(np.float32)
        g_nz = g[nonzero_delta].astype(np.float32)
        b_nz = b[nonzero_delta].astype(np.float32)

        h_nz = np.empty_like(delta_f)
        is_r = c_max_nz == r[nonzero_delta]
        is_g = (~is_r) & (c_max_nz == g[nonzero_delta])
        is_b = ~(is_r | is_g)

        h_nz[is_r] = 60.0 * (((g_nz[is_r] - b_nz[is_r]) / delta_f[is_r]) % 6.0)
        h_nz[is_g] = 60.0 * (((b_nz[is_g] - r_nz[is_g]) / delta_f[is_g]) + 2.0)
        h_nz[is_b] = 60.0 * (((r_nz[is_b] - g_nz[is_b]) / delta_f[is_b]) + 4.0)
        h[nonzero_delta] = h_nz

    s_u8 = np.zeros(c_max.shape, dtype=np.uint8)
    nonzero_max = c_max > 0
    if np.any(nonzero_max):
        s_vals = (delta[nonzero_max].astype(np.float32) / c_max[nonzero_max].astype(np.float32)) * 255.0
        s_u8[nonzero_max] = np.clip(np.round(s_vals), 0, 255).astype(np.uint8)

    h_u8 = np.clip(np.round(h * (255.0 / 360.0)), 0, 255).astype(np.uint8)
    return np.stack((h_u8, s_u8, c_max), axis=-1)


def _converter_worker(par: tuple[Path, Path]) -> None:
    origem, destino = par
    with Image.open(origem) as img:
        rgb_array = np.asarray(img.convert("RGB"))
    hsv_array = rgb_para_hsv_manual(rgb_array)
    Image.fromarray(hsv_array, mode="RGB").save(
        destino, format="JPEG", quality=95, subsampling=0
    )


def converter_dataset_para_hsv(dataset_dir: Path = DEFAULT_DATASET_DIR) -> int:
    """Converte todas as imagens do dataset em paralelo para Dataset/<split>/hsv."""
    image_dirs = listar_pastas_split(dataset_dir, "images")
    workers = max(1, (os.cpu_count() or 2) // 2)
    total_convertidas = 0

    for images_dir in image_dirs:
        split_name = images_dir.parent.name
        hsv_dir = images_dir.parent / "hsv"
        hsv_dir.mkdir(parents=True, exist_ok=True)

        # Remove eventuais PNGs antigos para liberar espaço em disco
        for antigo_png in hsv_dir.glob("*.png"):
            antigo_png.unlink()

        imagens = listar_imagens(images_dir)
        pendentes = [
            (img_path, hsv_dir / f"{img_path.stem}.jpg")
            for img_path in imagens
            if not (hsv_dir / f"{img_path.stem}.jpg").exists()
        ]

        if pendentes:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                list(
                    tqdm(
                        executor.map(_converter_worker, pendentes, chunksize=16),
                        total=len(pendentes),
                        desc=f"RGB -> HSV ({split_name}, {workers} workers)",
                        unit="img",
                    )
                )

        total_convertidas += len(pendentes)
        print(
            f"Split '{split_name}': {len(pendentes)} convertidas, "
            f"{len(imagens) - len(pendentes)} já existentes -> {hsv_dir}"
        )

    return total_convertidas


if __name__ == "__main__":
    converter_dataset_para_hsv()
