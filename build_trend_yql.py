"""Собрать self-contained YQL поиска трендов из актуального Python-кода.

Из папки ``C:\\Python`` запустите:
``python -m gmv_anomaly.build_trend_yql``.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
from pathlib import Path
import re
from typing import Dict, Sequence, Tuple
from zoneinfo import ZoneInfo

from .build_yql import (
    _identifier,
    _source_hash,
    _struct_fields,
    _validate_python_bootstrap,
)
from .config import PERIOD
from .trend_manager_output import MANAGER_TREND_COLUMN_LABELS
from .trend_udf_runtime import TREND_UDF_OUTPUT_SCHEMA, UDF_INPUT_SCHEMA


DEFAULT_YQL_PATH = Path(__file__).resolve().parent / "trend_prod.yql"
DEFAULT_INPUT_TABLE = (
    "//home/fdt/payoffline/projects/qr_yandex_pay/bi/payoffline_pulse_hier"
)
DEFAULT_OUTPUT_TABLE = (
    "//home/fdt/payoffline/projects/qr_yandex_pay/bi/payoffline_pulse_hier_Trend"
)
_BUNDLED_MODULES = (
    "config.py",
    "segment_keys.py",
    "data_preparation.py",
    "anomaly_scoring.py",
    "set_packing.py",
    "udf_runtime.py",
    "trend_analysis.py",
    "trend_most_recent_cp.py",
    "trend_scoring.py",
    "trend_manager_output.py",
    "trend_udf_runtime.py",
)


# ADDED: публичные имена YT повторяют заголовки Excel-листа
# «Менеджерский вывод». Единственная нормализация — удаление точек из
# сокращения «п.п.», которое не принимается именами полей YQL и DataLens.
_YT_COLUMN_LABEL_REPLACEMENTS = (("п.п.", "пп"),)


def _yt_output_label(column: str) -> str:
    """Получить проверенное публичное имя колонки YT/DataLens.

    Args:
        column: Техническое имя поля manager_output.

    Returns:
        Заголовок Excel без недопустимых для YQL точек и обратных кавычек.

    Raises:
        ValueError: Если после нормализации имя пусто или всё ещё содержит
            недопустимый символ.

    Examples:
        >>> _yt_output_label("global_trend_total_gmv_share_change_pp")
        'Изменение доли в Total GMV, пп'
    """

    label = MANAGER_TREND_COLUMN_LABELS.get(column, column).strip()
    for old, new in _YT_COLUMN_LABEL_REPLACEMENTS:
        label = label.replace(old, new)
    if not label or "." in label or "`" in label:
        raise ValueError(
            f"Некорректное имя колонки YT/DataLens для {column!r}: {label!r}"
        )
    return label


def _yt_output_projection(*, source_alias: str, indent: str) -> str:
    """Сформировать SELECT технических полей с публичными Excel-алиасами.

    Args:
        source_alias: Алиас результата REDUCE.
        indent: Отступ каждой строки проекции.

    Returns:
        Многострочный список SELECT-выражений.

    Raises:
        ValueError: Если публичные имена повторяются или не экранируются.

    Examples:
        >>> "Название сегмента" in _yt_output_projection(source_alias="r", indent="")
        True
    """

    labels = [_yt_output_label(column) for column, _ in TREND_UDF_OUTPUT_SCHEMA]
    if len(labels) != len(set(labels)):
        duplicates = sorted({label for label in labels if labels.count(label) > 1})
        raise ValueError(f"Повторяющиеся имена колонок YT/DataLens: {duplicates}")
    return ",\n".join(
        f"{indent}{source_alias}.{_identifier(column)} AS {_identifier(label)}"
        for (column, _), label in zip(TREND_UDF_OUTPUT_SCHEMA, labels)
    )


def _read_bundled_sources(package_dir: Path) -> Dict[str, str]:
    """Прочитать точные исходники модулей трендовой UDF.

    Args:
        package_dir: Директория пакета ``gmv_anomaly``.

    Returns:
        Соответствие полного имени модуля его UTF-8 исходнику.

    Raises:
        OSError: Если один из обязательных модулей недоступен.

    Examples:
        >>> # sources = _read_bundled_sources(Path("gmv_anomaly"))
    """

    return {
        f"gmv_anomaly.{filename[:-3]}": (package_dir / filename).read_text(
            encoding="utf-8"
        )
        for filename in _BUNDLED_MODULES
    }


def _python_bootstrap(sources: Dict[str, str]) -> str:
    """Собрать in-memory загрузчик модулей трендовой Python3 UDF.

    Args:
        sources: Исходники модулей в порядке зависимостей.

    Returns:
        Самодостаточный Python-скрипт с функцией ``run_algorithm``.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> "trend_udf_runtime" in _python_bootstrap({"gmv_anomaly.config": "x = 1"})
        True
    """

    encoded = {
        name: base64.b64encode(source.encode("utf-8")).decode("ascii")
        for name, source in sources.items()
    }
    module_items = ",\n".join(
        f"    {name!r}: {payload!r}" for name, payload in encoded.items()
    )
    return f'''import base64
import sys
import types

_BUNDLED_MODULES = {{
{module_items}
}}

_package = types.ModuleType("gmv_anomaly")
_package.__package__ = "gmv_anomaly"
_package.__path__ = []
sys.modules["gmv_anomaly"] = _package

for _module_name, _payload in _BUNDLED_MODULES.items():
    _module = types.ModuleType(_module_name)
    _module.__file__ = "<embedded>/" + _module_name.replace(".", "/") + ".py"
    _module.__package__ = _module_name.rpartition(".")[0]
    sys.modules[_module_name] = _module
    _source = base64.b64decode(_payload).decode("utf-8")
    exec(compile(_source, _module.__file__, "exec"), _module.__dict__)

from gmv_anomaly.trend_udf_runtime import run_algorithm
'''


def render_yql(
    *,
    input_table: str,
    output_table: str,
    period: str,
    generated_at: str,
    algorithm_version: str,
    sources: Dict[str, str],
) -> str:
    """Сформировать production YQL поиска трендов.

    Args:
        input_table: Путь входной YT-таблицы.
        output_table: Путь атомарно перезаписываемого результата.
        period: Период из ``config.PERIOD``.
        generated_at: ISO-время генерации скрипта.
        algorithm_version: Хеш включённых Python-исходников.
        sources: Исходники модулей UDF.

    Returns:
        Полный текст YQL.

    Raises:
        ValueError: Если путь или период нельзя безопасно встроить в YQL.

    Examples:
        >>> # text = render_yql(input_table="//in", output_table="//out", period="1W", generated_at="now", algorithm_version="v", sources={})
    """

    if "`" in input_table or "`" in output_table:
        raise ValueError("YT-путь не должен содержать обратную кавычку")
    if re.fullmatch(r"[1-9][0-9]*W", period) is None:
        raise ValueError(f"Неподдерживаемый формат периода: {period!r}")

    py_script = _python_bootstrap(sources)
    input_fields = _struct_fields(UDF_INPUT_SCHEMA, "            ")
    output_fields = _struct_fields(TREND_UDF_OUTPUT_SCHEMA, "            ")
    input_projection = ",\n".join(
        (
            f"        CAST(source.{_identifier(name)} AS Int64?) AS {_identifier(name)}"
            if name == "cal_date"
            else f"        source.{_identifier(name)} AS {_identifier(name)}"
        )
        for name, _ in UDF_INPUT_SCHEMA
    )
    # FIXED: публичные имена назначаются единственным внешним SELECT, который
    # читает REDUCE напрямую без отдельного именованного `$result`.
    output_projection = _yt_output_projection(
        source_alias="result",
        indent="        ",
    )
    return f'''USE hahn;
PRAGMA OrderedColumns;
PRAGMA SimpleColumns;
PRAGMA DqEngine = "disable";
PRAGMA yt.TmpFolder = "//tmp/fdt/payoffline/projects/qr_yandex_pay/bi";

-- GENERATED FILE. Не редактировать вручную.
-- Дата и время генерации скрипта: {generated_at}
-- Версия алгоритма (SHA-256/16): {algorithm_version}
-- Период трендового расчёта: {period}

$pyScript = @@
{py_script}@@;

$input = (
    SELECT
{input_projection}
    FROM `{input_table}` AS source
    WHERE source.period == "{period}"
);

$run_algorithm = Python3::run_algorithm(
    Callable<
        (Stream<Struct<
{input_fields}
        >>)
        ->
        Stream<Struct<
{output_fields}
        >>
    >,
    $pyScript
);

-- WITH TRUNCATE выполняется транзакционно: при исключении UDF предыдущая
-- корректная таблица остаётся без изменений. UDF возвращает только бизнес-
-- строки листа «Менеджерский вывод», без Excel-строки пояснений.
INSERT INTO `{output_table}` WITH TRUNCATE
SELECT
{output_projection}
FROM (
    REDUCE $input
    ON period
    USING ALL $run_algorithm(TableRows())
) AS result;
'''


def build_yql(
    output_path: Path = DEFAULT_YQL_PATH,
    *,
    input_table: str = DEFAULT_INPUT_TABLE,
    output_table: str = DEFAULT_OUTPUT_TABLE,
    period: str = PERIOD,
) -> Tuple[Path, str, str]:
    """Собрать трендовый YQL-файл и вернуть его метаданные.

    Args:
        output_path: Локальный путь генерируемого YQL.
        input_table: Входная YT-таблица.
        output_table: Выходная YT-таблица.
        period: Единственный период трендового расчёта.

    Returns:
        Путь, время генерации и версия алгоритма.

    Raises:
        OSError: Если исходники нельзя прочитать или YQL нельзя записать.
        RuntimeError: Если embedded Python3 UDF не импортируется.

    Examples:
        >>> # path, generated_at, version = build_yql()
    """

    package_dir = Path(__file__).resolve().parent
    sources = _read_bundled_sources(package_dir)
    algorithm_version = _source_hash(sources)
    generated_at = datetime.now(ZoneInfo("Europe/Moscow")).isoformat(
        timespec="seconds"
    )
    _validate_python_bootstrap(_python_bootstrap(sources))
    rendered = render_yql(
        input_table=input_table,
        output_table=output_table,
        period=period,
        generated_at=generated_at,
        algorithm_version=algorithm_version,
        sources=sources,
    )
    output_path.write_text(rendered, encoding="utf-8", newline="\n")
    return output_path, generated_at, algorithm_version


def main(argv: Sequence[str] | None = None) -> int:
    """Обработать CLI-параметры однокнопочной сборки трендового YQL.

    Args:
        argv: Аргументы без имени программы.

    Returns:
        Код завершения процесса.

    Raises:
        OSError: Если сборка файла невозможна.
        RuntimeError: Если embedded Python3 UDF не импортируется.

    Examples:
        >>> # raise SystemExit(main())
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_YQL_PATH)
    parser.add_argument("--input-table", default=DEFAULT_INPUT_TABLE)
    parser.add_argument("--output-table", default=DEFAULT_OUTPUT_TABLE)
    parser.add_argument("--period", default=PERIOD)
    args = parser.parse_args(argv)
    path, generated_at, version = build_yql(
        args.output,
        input_table=args.input_table,
        output_table=args.output_table,
        period=args.period,
    )
    print(f"YQL: {path}")
    print(f"Generated at: {generated_at}")
    print(f"Algorithm version: {version}")
    print(f"Period: {args.period}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
