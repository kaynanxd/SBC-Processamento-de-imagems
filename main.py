"""Menu principal interativo."""

from scripts.binary_masks import gerar_mascaras_binarias_verduras
from scripts.download_dataset import DEFAULT_DATASET_DIR, baixar_e_preparar_dataset
from scripts.entropy import (
    calcular_entropia_canais_verduras,
    calcular_entropia_dataset,
)
from scripts.rgb_to_hsv import converter_dataset_para_hsv


def preparar_dataset_completo() -> None:
    """Executa download, conversão HSV, máscaras binárias e métricas de entropia."""
    baixar_e_preparar_dataset(DEFAULT_DATASET_DIR)
    converter_dataset_para_hsv(DEFAULT_DATASET_DIR)
    calcular_entropia_dataset(DEFAULT_DATASET_DIR)
    gerar_mascaras_binarias_verduras(DEFAULT_DATASET_DIR)
    calcular_entropia_canais_verduras(DEFAULT_DATASET_DIR)


def exibir_menu() -> None:
    print("\n=== Segmentação Verduras Verdes ===")
    print("1 - Baixar e preparar dataset completo")
    print("0 - Sair")


def executar_opcao(opcao: str) -> bool:
    """Executa a opção escolhida. Retorna False para encerrar o menu."""
    if opcao == "1":
        try:
            preparar_dataset_completo()
        except Exception as exc:
            print(f"\nErro ao executar preparação completa: {exc}")
        return True

    if opcao == "0":
        print("Saindo...")
        return False

    print("Opção inválida. Escolha uma opção do menu.")
    return True


def main() -> None:
    while True:
        exibir_menu()
        try:
            opcao = input("\nEscolha uma opção: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando...")
            break

        if not executar_opcao(opcao):
            break


if __name__ == "__main__":
    main()
