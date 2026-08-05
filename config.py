"""Конфигурация поиска аномальных GMV-сегментов."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# FIXED: После переноса всех файлов в пакет сохраняем прежнюю базу путей — корень проекта.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

date_string = "05_08"

# ADDED: Все параметры запуска задаются здесь; CLI-аргументы больше не используются.
INPUT_PATH = _PROJECT_ROOT / f"payoffline_pulse_hier_{date_string}.xlsx"
OUTPUT_PATH = _PROJECT_ROOT / f"gmv_anomaly_report_{date_string}.xlsx"
TREE_OUTPUT_PATH: Path | None = _PROJECT_ROOT / f"Граф_Anomaly_{date_string}.png"
# FIXED: Для каждой относительной метрики строится отдельный PNG в общем каталоге.
RATIO_TREES_OUTPUT_DIR: Path | None = (
    _PROJECT_ROOT / f"Графы_Anomaly_{date_string}_Доли"
)
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
DEFAULT_RATIO_TREES_OUTPUT_DIR = RATIO_TREES_OUTPUT_DIR
# ADDED: Deprecated compatibility aliases; значения теперь указывают каталог.
RATIO_TREE_OUTPUT_PATH = RATIO_TREES_OUTPUT_DIR
DEFAULT_RATIO_TREE_OUTPUT_PATH = DEFAULT_RATIO_TREES_OUTPUT_DIR


@dataclass(frozen=True)
class RatioMetricSpec:
    """Описать входной и расчётный контракт относительной метрики.

    Args:
        name: Имя метрики в long-отчёте и имени PNG-файла.
        value_column: Готовое значение отношения из YQL.
        numerator_column: Аддитивный числитель.
        denominator_column: Аддитивный знаменатель.
        change_mode: Формула межнедельного изменения.
        contribution_mode: Способ расчёта materiality и hierarchy contribution.
        bounded: Ограничена ли метрика интервалом `[0, 1]`.
        validation_abs_tolerance: Допуск сверки значения с отношением компонентов.

    Returns:
        Неизменяемая спецификация метрики.

    Raises:
        ValueError: Не выбрасывается при создании.

    Examples:
        >>> RATIO_METRICS[0].name
        'success_rate'
    """

    name: str
    value_column: str
    numerator_column: str
    denominator_column: str
    change_mode: str = "absolute_delta"
    contribution_mode: str = "exact_atomic"
    bounded: bool = True
    validation_abs_tolerance: float = 1e-10


# ADDED: Все локальные относительные метрики используют единый scoring-контур.
# ``share_in_total_gmv`` намеренно не входит в реестр: его знаменатель глобален
# и не удовлетворяет аддитивному контракту текущей exact-attribution.
RATIO_METRICS = (
    RatioMetricSpec("success_rate", "success_rate", "tx", "tx0"),
    RatioMetricSpec(
        "refund_tx_share", "refund_tx_share", "refund_tx_numerator", "tx"
    ),
    RatioMetricSpec(
        "authzone_tx_share", "authzone_tx_share", "authzone_tx_numerator", "tx"
    ),
    RatioMetricSpec(
        "payapp_tx_share", "payapp_tx_share", "payapp_tx_numerator", "tx"
    ),
    RatioMetricSpec(
        "split_gmv_share", "split_gmv_share", "split_gmv_numerator", "gmv"
    ),
    RatioMetricSpec(
        "credlim_gmv_share", "credlim_gmv_share", "credlim_gmv_numerator", "gmv"
    ),
    RatioMetricSpec(
        "tips_gmv_share",
        "tips_gmv_share",
        "tips_gmv_numerator",
        "gmv",
        bounded=False,
    ),
    RatioMetricSpec(
        "cashback_gmv_share",
        "cashback_gmv_share",
        "cashback_gmv_numerator",
        "gmv",
        bounded=False,
    ),
)

# ADDED: Компоненты выводятся из реестра — новую метрику нельзя забыть
# перенести в полную недельную сетку.
_BASE_ADDITIVE_COLUMNS = {"gmv", "tx", "au", "am"}
RATIO_ADDITIVE_COLUMNS = tuple(
    sorted(
        {
            column
            for spec in RATIO_METRICS
            for column in (spec.numerator_column, spec.denominator_column)
            if column not in _BASE_ADDITIVE_COLUMNS
        }
    )
)

# ADDED: Compatibility alias сохраняет прежний одноэлементный пилотный контракт;
# внутри пакета используется полный реестр ``RATIO_METRICS``.
PILOT_RATIO_METRICS = tuple(
    spec for spec in RATIO_METRICS if spec.name == "authzone_tx_share"
)

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
    "tx0",
    "refund_tx_numerator",
    "authzone_tx_numerator",
    "payapp_tx_numerator",
    "split_gmv_numerator",
    "credlim_gmv_numerator",
    "tips_gmv_numerator",
    "cashback_gmv_numerator",
    "success_rate",
    "refund_tx_share",
    "authzone_tx_share",
    "payapp_tx_share",
    "split_gmv_share",
    "credlim_gmv_share",
    "tips_gmv_share",
    "cashback_gmv_share",
    "segment_id",
    "segment_key",
    "segment_level",
} | {spec.value_column for spec in RATIO_METRICS} | set(RATIO_ADDITIVE_COLUMNS)

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
        single_child_factor: Коэффициент родителя при одном доминирующем
            потомке в сильнейшей eligible-группе.
        max_hierarchy_descendants: Максимальное число eligible-потомков одного
            родителя, для которого разрешено полное физическое перечисление
            непересекающихся групп.
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
    aggregation_bonus_lambda: float = 0.3  # +-17.5% к score родителя
    single_child_factor: float = 0.85
    # FIXED: Порог переключения с перечисления на exact Set Packing.
    max_hierarchy_descendants: int = 25
    # ADDED: Системный cap для родителя, пересказывающего одного потомка.
    dominant_child_capture_threshold: float = 0.80
    dominant_child_score_margin: float = 0.02
    set_packing_gap_tolerance: float = 1e-9
    max_exact_fallback_size: int = 25
    max_manager_facts: int = 10


# ADDED: Единственный экземпляр порогов, используемый безаргументным запуском.
THRESHOLDS = AnomalyThresholds()
