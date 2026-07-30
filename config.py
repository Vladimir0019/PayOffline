"""Конфигурация поиска аномальных GMV-сегментов."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# FIXED: После переноса всех файлов в пакет сохраняем прежнюю базу путей — корень проекта.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ADDED: Все параметры запуска задаются здесь; CLI-аргументы больше не используются.
INPUT_PATH = _PROJECT_ROOT / "payoffline_pulse_hier_22_07.xlsx"
OUTPUT_PATH = _PROJECT_ROOT / "gmv_anomaly_report_22_07_test.xlsx"
TREE_OUTPUT_PATH: Path | None = _PROJECT_ROOT / "Граф_Anomaly_22_07_test.png"
SHEET_NAME: int | str = 0
PERIOD: str | None = "1W"
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
        sigma_floor: Нижняя граница масштаба колебаний доли.
        z_cap: Верхняя граница z-score, участвующего в score.
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
    z_cap: float = 6.0
    set_packing_gap_tolerance: float = 1e-9
    max_exact_fallback_size: int = 25
    max_manager_facts: int = 10


# ADDED: Единственный экземпляр порогов, используемый безаргументным запуском.
THRESHOLDS = AnomalyThresholds()
