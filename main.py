"""Безаргументная точка запуска поиска GMV-аномалий."""

from __future__ import annotations

from pathlib import Path

from .config import (
    CURRENT_CAL_DATE,
    DIM_COLUMNS,
    INPUT_PATH,
    OUTPUT_PATH,
    PERIOD,
    RATIO_TREES_OUTPUT_DIR,
    SHEET_NAME,
    THRESHOLDS,
    TREE_OUTPUT_PATH,
)
from .pipeline import run_pipeline


def main() -> None:
    """Запустить pipeline с параметрами исключительно из ``config.py``.

    Args:
        Нет аргументов.

    Returns:
        None.

    Raises:
        ValueError: Если входные данные или параметры конфигурации некорректны.
        ImportError: Если запрошено дерево, но matplotlib недоступен.
        OSError: Если входной или выходной файл недоступен.

    Examples:
        >>> # python -m gmv_anomaly
    """

    result = run_pipeline(
        input_path=INPUT_PATH,
        output_path=OUTPUT_PATH,
        sheet_name=SHEET_NAME,
        period=PERIOD,
        dim_cols=DIM_COLUMNS,
        current_cal_date=CURRENT_CAL_DATE,
        thresholds=THRESHOLDS,
        tree_output_path=TREE_OUTPUT_PATH,
        ratio_trees_output_dir=RATIO_TREES_OUTPUT_DIR,
    )

    control = result["control"].set_index("показатель")["значение"].to_dict()
    print("Готово.")
    print(f"Итоговый файл: {OUTPUT_PATH}")
    if TREE_OUTPUT_PATH is not None:
        tree_path = Path(TREE_OUTPUT_PATH)
        if not tree_path.suffix:
            tree_path = tree_path.with_suffix(".png")
        print(f"Дерево аномалий: {tree_path.resolve()}")
    ratio_status = result.get("ratio_status")
    if (
        RATIO_TREES_OUTPUT_DIR is not None
        and ratio_status is not None
        and not ratio_status.empty
    ):
        generated = ratio_status[ratio_status["tree_status"].eq("GENERATED")]
        print(
            f"Графы относительных метрик: {len(generated)}; "
            f"каталог: {Path(RATIO_TREES_OUTPUT_DIR).resolve()}"
        )
    print(f"Кандидатов: {control.get('candidate_count')}")
    print(f"Выбрано аномалий: {control.get('selected_count')}")
    print(f"Нарушения пересечения атомов: {control.get('double_count_violation_count')}")


if __name__ == "__main__":
    main()
