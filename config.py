"""Конфигурация поиска аномальных GMV-сегментов."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# FIXED: После переноса всех файлов в пакет сохраняем прежнюю базу путей — корень проекта.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

date_string = "28_07"

# ADDED: Все параметры запуска задаются здесь; CLI-аргументы больше не используются.
INPUT_PATH = _PROJECT_ROOT / f"payoffline_pulse_hier_{date_string}.xlsx"
OUTPUT_PATH = _PROJECT_ROOT / f"gmv_anomaly_report_{date_string}.xlsx"
TREE_OUTPUT_PATH: Path | None = _PROJECT_ROOT / f"Граф_Anomaly_{date_string}.png"
SHEET_NAME: int | str = 0
# FIXED: Период обязателен для запуска. Его значение определяет и фильтр
# входной витрины, и ожидаемый шаг календарной оси.
PERIOD: str = "1W"
# FIXED: Dimensions фиксированы контрактом витрины; новые колонки Excel не влияют на segment_id.
DIM_COLUMNS: tuple[str, ...] = (
    "geo",
    "products",
    "merchants_type",
    "is_terminal_or_cpqr",
)
CURRENT_CAL_DATE: int | None = None

# Сохранены как compatibility aliases для внешнего кода.
DEFAULT_INPUT_PATH = INPUT_PATH
DEFAULT_OUTPUT_PATH = OUTPUT_PATH
DEFAULT_TREE_OUTPUT_PATH = TREE_OUTPUT_PATH

# Для исторического anomaly-файла фиксируем служебные и метрические колонки.
# FIXED: При заданном DIM_COLUMNS все остальные колонки входного Excel также
# считаются техническими: они не участвуют в построении segment_id/segment_key.
ANOMALY_TECH_COLUMNS = {
    "period",
    "cal_date",
    "calendar_date",
    "slice_depth",
    "gmv",
    "tx",
    "au",
    "am",
    "aov",
    "tpm",
    "freq",
    "share_in_total_gmv",
    "segment_id",
    "segment_key",
    "segment_level",
}

METRIC_COLUMNS = ["tx", "au", "am", "aov", "tpm", "freq"]
MANAGER_METRIC_PCT_COLUMNS = {
    "relative_wow": "GMV WoW %",
    "tx_wow_pct": "TX WoW %",
    "au_wow_pct": "AU WoW %",
    "am_wow_pct": "AM WoW %",
    "aov_wow_pct": "AOV WoW %",
    "tpm_wow_pct": "TPM WoW %",
    "freq_wow_pct": "Freq WoW %",
}


@dataclass(frozen=True)
class AnomalyThresholds:
    """Пороги алгоритма поиска необычных сегментов.

    Args:
        min_anomaly_abs: Минимальная абсолютная величина аномального вклада в рублях.
        min_z_score: Минимальный robust z-score для обычного сегмента.
        min_materiality_share: Минимальная доля изменения GMV в gross movement атомарного слоя.
        sigma_floor: Нижняя граница масштаба колебаний относительного роста.
        lifecycle_z_score: Фиксированный z-score для lifecycle-события, когда
            относительный WoW не определён из-за нулевого предыдущего GMV.
        hierarchy_reconciliation_abs_tolerance: Абсолютный допуск сверки GMV
            каждого родителя с суммой покрытых атомов максимальной глубины.
        aggregation_bonus_lambda: Наклон линейной hierarchy-корректировки.
        single_child_factor: Коэффициент родителя при одном потомке в сильнейшей
            eligible-группе.
        max_hierarchy_descendants: Максимальное число eligible-потомков одного
            родителя, для которого разрешено полное физическое перечисление
            непересекающихся групп. Перебор растёт как ``2^n − 1``, поэтому
            превышение останавливает расчёт вместо молчаливого зависания.
        dominant_child_capture_threshold: Минимальная доля абсолютного движения
            атомов родителя, объяснённая единственным сильным потомком, для
            применения dominance cap.
        dominant_child_score_margin: Относительный запас, на который score
            родителя ограничивается ниже score доминирующего потомка.
        set_packing_gap_tolerance: Максимальный относительный MIP gap для признания решения оптимальным.
        max_exact_fallback_size: Максимальный размер компоненты для собственного exact fallback.
        max_manager_facts: Максимальное число фактов в менеджерском выводе.

    Returns:
        Экземпляр с порогами.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> AnomalyThresholds().min_z_score
        2.0
    """

    min_anomaly_abs: float = 200_000.0
    min_z_score: float = 2.0
    min_materiality_share: float = 0.0001
    sigma_floor: float = 0.00001
    # FIXED: Параметр больше не называется cap: непрерывный robust z-score не ограничивается.
    lifecycle_z_score: float = 6.0
    # ADDED: Fail-fast допуск бухгалтерской сверки иерархической витрины.
    hierarchy_reconciliation_abs_tolerance: float = 1e-4
    # ADDED: Параметры согласованной hierarchy-корректировки score.
    aggregation_bonus_lambda: float = 0.3
    single_child_factor: float = 0.85
    # ADDED: Предохранитель экспоненциального перечисления hierarchy-групп.
    max_hierarchy_descendants: int = 25
    # ADDED: Системный cap для родителя, пересказывающего одного потомка.
    dominant_child_capture_threshold: float = 0.80
    dominant_child_score_margin: float = 0.02
    set_packing_gap_tolerance: float = 1e-9
    max_exact_fallback_size: int = 25
    max_manager_facts: int = 10


# ADDED: Единственный экземпляр порогов, используемый безаргументным запуском.
THRESHOLDS = AnomalyThresholds()
