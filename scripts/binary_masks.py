"""Geração de máscaras binárias (verduras verdes vs resto)."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

from scripts.utils import (
    DEFAULT_DATASET_DIR,
    SUMMARY_FILENAME,
    atualizar_resumo_json,
    listar_pastas_split,
)

# IDs estritamente verdes de verduras/vegetais no FoodSeg103 (id2label.json)
GREEN_VEGETABLE_IDS: frozenset[int] = frozenset({
    76,   # spring onion
    77,   # rape
    79,   # okra
    80,   # lettuce
    82,   # cucumber
    85,   # asparagus
    87,   # broccoli
    88,   # celery stick
    89,   # cilantro mint
    90,   # snow peas
    95,   # green beans
    96,   # French beans
    102,  # salad
})

_GREEN_LUT = np.zeros(256, dtype=np.uint8)
for _cid in GREEN_VEGETABLE_IDS:
    _GREEN_LUT[_cid] = 255


def _processar_mascara_worker(args: tuple[Path, Path, bool]) -> tuple[str, int, int, list[int]]:
    """Converte uma máscara de classes (0..103) em binária (0 ou 255)."""
    origem, destino, sobrescrever = args
    with Image.open(origem) as img:
        mask_arr = np.asarray(img)
        if mask_arr.ndim == 3:
            mask_arr = mask_arr[:, :, 0]

    bin_arr = _GREEN_LUT[mask_arr]
    if sobrescrever or not destino.exists():
        Image.fromarray(bin_arr, mode="L").save(destino, format="PNG", compress_level=1)

    presentes = [int(c) for c in np.unique(mask_arr) if int(c) in GREEN_VEGETABLE_IDS]
    return origem.stem, int(np.count_nonzero(bin_arr)), int(bin_arr.size), presentes


def gerar_mascaras_binarias_verduras(dataset_dir: Path = DEFAULT_DATASET_DIR) -> dict:
    """Gera máscaras binárias (0=resto, 255=verdura verde) e resumo estatístico."""
    dataset_dir = Path(dataset_dir)
    mask_dirs = listar_pastas_split(dataset_dir, "masks")
    workers = max(1, (os.cpu_count() or 2) // 2)
    ids_atuais = sorted(GREEN_VEGETABLE_IDS)

    resumo_path = dataset_dir / SUMMARY_FILENAME
    sobrescrever = True
    if resumo_path.exists():
        with resumo_path.open("r", encoding="utf-8") as f:
            sobrescrever = json.load(f).get("ids_verduras_verdes") != ids_atuais

    resumo_splits: dict[str, dict] = {}
    contagem_classes: Counter[int] = Counter()

    for masks_dir in mask_dirs:
        split_name = masks_dir.parent.name
        bin_dir = masks_dir.parent / "masks_binarias_verduras"
        bin_dir.mkdir(parents=True, exist_ok=True)

        mascaras = sorted(p for p in masks_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png")
        tarefas = [(p, bin_dir / p.name, sobrescrever) for p in mascaras]

        with ProcessPoolExecutor(max_workers=workers) as executor:
            resultados = list(
                tqdm(
                    executor.map(_processar_mascara_worker, tarefas, chunksize=32),
                    total=len(tarefas),
                    desc=f"Máscaras binárias ({split_name})",
                    unit="mask",
                )
            )

        stems_com_verduras = [stem for stem, pos, _, _ in resultados if pos > 0]
        total_positivos = sum(pos for _, pos, _, _ in resultados)
        total_pixels = sum(tot for _, _, tot, _ in resultados)
        pct_pixels = (total_positivos / total_pixels * 100.0) if total_pixels else 0.0
        for _, _, _, presentes in resultados:
            contagem_classes.update(presentes)

        lista_path = masks_dir.parent / "imagens_com_verduras.txt"
        lista_path.write_text("\n".join(stems_com_verduras) + ("\n" if stems_com_verduras else ""), encoding="utf-8")

        resumo_splits[split_name] = {
            "total_imagens": len(resultados),
            "imagens_com_verduras": len(stems_com_verduras),
            "imagens_sem_verduras": len(resultados) - len(stems_com_verduras),
            "percentual_pixels_verduras": round(pct_pixels, 4),
        }

        print(
            f"Split '{split_name}': {len(stems_com_verduras)}/{len(resultados)} imagens "
            f"com verduras ({pct_pixels:.2f}% dos pixels) -> {bin_dir}"
        )

    id2label_path = dataset_dir / "id2label.json"
    id2label = json.loads(id2label_path.read_text(encoding="utf-8")) if id2label_path.exists() else {}
    ocorrencias_por_verdura = {
        id2label.get(str(cid), str(cid)).strip(): contagem_classes[cid]
        for cid, _ in contagem_classes.most_common()
    }

    resumo_path = atualizar_resumo_json(
        dataset_dir,
        ids_verduras_verdes=ids_atuais,
        splits=resumo_splits,
        ocorrencias_por_verdura=ocorrencias_por_verdura,
    )
    print(f"Resumo salvo em: {resumo_path}")
    return resumo_splits


if __name__ == "__main__":
    gerar_mascaras_binarias_verduras()
