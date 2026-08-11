"""Собрать self-contained YQL с Python3 UDF из актуальных модулей пакета. 
Из папки C:\Python запусти: python -m gmv_anomaly.build_yql"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import hashlib
from pathlib import Path
import re
import subprocess
import sys
from typing import Dict, Sequence, Tuple
from zoneinfo import ZoneInfo

from .udf_runtime import UDF_INPUT_SCHEMA, UDF_OUTPUT_SCHEMA


DEFAULT_YQL_PATH = Path(__file__).resolve().parent / "anomaly_prod.yql"
DEFAULT_INPUT_TABLE = (
    "//home/fdt/payoffline/projects/qr_yandex_pay/bi/payoffline_pulse_hier"
)
DEFAULT_OUTPUT_TABLE = (
    "//home/fdt/payoffline/projects/qr_yandex_pay/bi/payoffline_pulse_insights_v3"
)
_BUNDLED_MODULES = (
    "config.py",
    "segment_keys.py",
    "data_preparation.py",
    "anomaly_scoring.py",
    "set_packing.py",
    "udf_runtime.py",
)
_RESERVED_IDENTIFIERS = {"rank"}


def _identifier(name: str) -> str:
    """Экранировать имя колонки только когда это требуется YQL.

    Args:
        name: Имя поля Struct или колонки.

    Returns:
        Исходное либо заключённое в обратные кавычки имя.

    Raises:
        ValueError: Если имя содержит обратную кавычку.

    Examples:
        >>> _identifier("segment_id")
        'segment_id'
        >>> _identifier("GMV WoW %")
        '`GMV WoW %`'
    """

    if "`" in name:
        raise ValueError(f"Имя YQL-поля содержит обратную кавычку: {name!r}")
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) and name.lower() not in _RESERVED_IDENTIFIERS:
        return name
    return f"`{name}`"


def _optional_type(yql_type: str) -> str:
    """Получить Optional-вариант простого YQL-типа.

    Args:
        yql_type: Тип из контракта Struct.

    Returns:
        Тип с суффиксом ``?``.

    Raises:
        ValueError: Не выбрасывается для используемых примитивных типов.

    Examples:
        >>> _optional_type("String")
        'String?'
    """

    return yql_type if yql_type.endswith("?") else f"{yql_type}?"


def _read_bundled_sources(package_dir: Path) -> Dict[str, str]:
    """Прочитать точные исходники модулей, включаемых в UDF.

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


def _source_hash(sources: Dict[str, str]) -> str:
    """Рассчитать стабильную версию алгоритма по включённым исходникам.

    Args:
        sources: Модули будущей UDF.

    Returns:
        Первые 16 символов SHA-256.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> len(_source_hash({"a": "b"}))
        16
    """

    digest = hashlib.sha256()
    for module_name in sorted(sources):
        digest.update(module_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sources[module_name].encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _python_bootstrap(sources: Dict[str, str]) -> str:
    """Собрать загрузчик in-memory модулей для embedded Python3 UDF.

    Args:
        sources: Исходники модулей в порядке зависимостей.

    Returns:
        Самодостаточный Python-скрипт с функцией ``run_algorithm``.

    Raises:
        ValueError: Не выбрасывается.

    Examples:
        >>> "run_algorithm" in _python_bootstrap({"gmv_anomaly.config": "x = 1"})
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

from gmv_anomaly.udf_runtime import run_algorithm
'''


def _validate_python_bootstrap(py_script: str) -> None:
    """Проверить embedded-UDF в чистом локальном Python-процессе.

    Args:
        py_script: Сформированный bootstrap-код будущей UDF.

    Returns:
        None.

    Raises:
        RuntimeError: Если модуль не компилируется, не импортируется или забыта
            новая внутренняя зависимость пакета.

    Examples:
        >>> _validate_python_bootstrap("value = 1")
    """

    completed = subprocess.run(
        [sys.executable, "-"],
        input=py_script,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            "Embedded Python3 UDF не прошла локальную проверку; "
            "предыдущий YQL-файл сохранён.\n"
            + diagnostic[-4000:]
        )


def _struct_fields(schema: Sequence[Tuple[str, str]], indent: str) -> str:
    """Сформировать поля ``Struct`` из единой схемы.

    Args:
        schema: Пары ``имя, YQL-тип``.
        indent: Отступ каждой строки.

    Returns:
        Многострочный список полей.

    Raises:
        ValueError: Если имя нельзя экранировать.

    Examples:
        >>> "x: Int64" in _struct_fields((("x", "Int64"),), "")
        True
    """

    return ",\n".join(
        f"{indent}{_identifier(name)}: {yql_type}" for name, yql_type in schema
    )


def _select_fields(
    schema: Sequence[Tuple[str, str]],
    *,
    source_alias: str,
    indent: str,
) -> str:
    """Сформировать явную проекцию полей с AS-алиасами.

    Args:
        schema: Схема полей.
        source_alias: Алиас исходной таблицы.
        indent: Отступ строк.

    Returns:
        Список SELECT-выражений.

    Raises:
        ValueError: Если имя нельзя экранировать.

    Examples:
        >>> _select_fields((("x", "Int64"),), source_alias="r", indent="")
        'r.x AS x'
    """

    return ",\n".join(
        f"{indent}{source_alias}.{_identifier(name)} AS {_identifier(name)}"
        for name, _ in schema
    )


def _technical_null_fields(indent: str) -> str:
    """Сформировать NULL-поля продуктовой схемы для технической строки.

    Args:
        indent: Отступ строк.

    Returns:
        SELECT-выражения с типизированными NULL.

    Raises:
        ValueError: Если имя нельзя экранировать.

    Examples:
        >>> "CAST(NULL AS String?)" in _technical_null_fields("")
        True
    """

    return ",\n".join(
        f"{indent}CAST(NULL AS {_optional_type(yql_type)}) AS {_identifier(name)}"
        for name, yql_type in UDF_OUTPUT_SCHEMA
    )


def render_yql(
    *,
    input_table: str,
    output_table: str,
    generated_at: str,
    algorithm_version: str,
    sources: Dict[str, str],
) -> str:
    """Сформировать полный production YQL.

    Args:
        input_table: Путь входной YT-таблицы.
        output_table: Путь атомарно перезаписываемого результата.
        generated_at: ISO-время генерации скрипта.
        algorithm_version: Хеш включённых Python-исходников.
        sources: Исходники модулей UDF.

    Returns:
        Полный текст YQL.

    Raises:
        ValueError: Если путь содержит обратную кавычку.

    Examples:
        >>> # text = render_yql(input_table="//in", output_table="//out", generated_at="now", algorithm_version="v", sources={})
    """

    if "`" in input_table or "`" in output_table:
        raise ValueError("YT-путь не должен содержать обратную кавычку")
    py_script = _python_bootstrap(sources)
    input_fields = _struct_fields(UDF_INPUT_SCHEMA, "            ")
    output_fields = _struct_fields(UDF_OUTPUT_SCHEMA, "            ")
    input_projection = ",\n".join(
        (
            f"        CAST(source.{_identifier(name)} AS Int64?) AS {_identifier(name)}"
            if name == "cal_date"
            else f"        source.{_identifier(name)} AS {_identifier(name)}"
        )
        for name, _ in UDF_INPUT_SCHEMA
    )
    udf_projection = _select_fields(
        UDF_OUTPUT_SCHEMA,
        source_alias="result",
        indent="        ",
    )
    final_projection = _select_fields(
        UDF_OUTPUT_SCHEMA,
        source_alias="rows",
        indent="        ",
    )
    technical_nulls = _technical_null_fields("        ")
    return f'''USE hahn;
PRAGMA OrderedColumns;
PRAGMA SimpleColumns;
PRAGMA DqEngine = "disable";
PRAGMA yt.TmpFolder = "//tmp/fdt/payoffline/projects/qr_yandex_pay/bi";

-- GENERATED FILE. Не редактировать вручную.
-- Дата и время генерации скрипта: {generated_at}
-- Версия алгоритма (SHA-256/16): {algorithm_version}

$pyScript = @@
{py_script}@@;

$input = (
    SELECT
{input_projection}
    FROM `{input_table}` AS source
    WHERE source.period IN ("1W", "4W", "13W")
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

$input_1w = (SELECT * FROM $input WHERE period == "1W");
$input_4w = (SELECT * FROM $input WHERE period == "4W");
$input_13w = (SELECT * FROM $input WHERE period == "13W");

$result_1w = (
    REDUCE $input_1w
    ON period
    USING ALL $run_algorithm(TableRows())
);
$result_4w = (
    REDUCE $input_4w
    ON period
    USING ALL $run_algorithm(TableRows())
);
$result_13w = (
    REDUCE $input_13w
    ON period
    USING ALL $run_algorithm(TableRows())
);

$udf_result = (
    SELECT * FROM $result_1w
    UNION ALL
    SELECT * FROM $result_4w
    UNION ALL
    SELECT * FROM $result_13w
);

$data_rows = (
    SELECT
        "Данные" AS `Тип строки`,
        CAST(NULL AS String?) AS `Дата и время генерации скрипта`,
        CAST(NULL AS String?) AS `Версия алгоритма`,
{udf_projection}
    FROM $udf_result AS result
);

$technical_row = (
    SELECT
        "Техническая информация" AS `Тип строки`,
        "{generated_at}" AS `Дата и время генерации скрипта`,
        "{algorithm_version}" AS `Версия алгоритма`,
{technical_nulls}
);

$all_rows = (
    SELECT * FROM $data_rows
    UNION ALL
    SELECT * FROM $technical_row
);

-- WITH TRUNCATE выполняется транзакционно: при исключении UDF предыдущая
-- корректная таблица остаётся без изменений.
INSERT INTO `{output_table}` WITH TRUNCATE
SELECT
        rows.`Тип строки` AS `Тип строки`,
        rows.`Дата и время генерации скрипта` AS `Дата и время генерации скрипта`,
        rows.`Версия алгоритма` AS `Версия алгоритма`,
{final_projection}
FROM $all_rows AS rows;
'''


def build_yql(
    output_path: Path = DEFAULT_YQL_PATH,
    *,
    input_table: str = DEFAULT_INPUT_TABLE,
    output_table: str = DEFAULT_OUTPUT_TABLE,
) -> Tuple[Path, str, str]:
    """Собрать YQL-файл и вернуть его метаданные.

    Args:
        output_path: Локальный путь генерируемого YQL.
        input_table: Входная YT-таблица.
        output_table: Выходная YT-таблица.

    Returns:
        Путь, время генерации и версия алгоритма.

    Raises:
        OSError: Если исходники нельзя прочитать или YQL нельзя записать.

    Examples:
        >>> # path, generated_at, version = build_yql()
    """

    package_dir = Path(__file__).resolve().parent
    sources = _read_bundled_sources(package_dir)
    algorithm_version = _source_hash(sources)
    generated_at = datetime.now(ZoneInfo("Europe/Moscow")).isoformat(
        timespec="seconds"
    )
    # ADDED: Ошибка импорта/новой зависимости обнаруживается до перезаписи
    # предыдущего корректного YQL-артефакта.
    _validate_python_bootstrap(_python_bootstrap(sources))
    rendered = render_yql(
        input_table=input_table,
        output_table=output_table,
        generated_at=generated_at,
        algorithm_version=algorithm_version,
        sources=sources,
    )
    output_path.write_text(rendered, encoding="utf-8", newline="\n")
    return output_path, generated_at, algorithm_version


def main(argv: Sequence[str] | None = None) -> int:
    """Обработать CLI-параметры однокнопочной сборки.

    Args:
        argv: Аргументы без имени программы.

    Returns:
        Код завершения процесса.

    Raises:
        OSError: Если сборка файла невозможна.

    Examples:
        >>> # raise SystemExit(main())
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_YQL_PATH)
    parser.add_argument("--input-table", default=DEFAULT_INPUT_TABLE)
    parser.add_argument("--output-table", default=DEFAULT_OUTPUT_TABLE)
    args = parser.parse_args(argv)
    path, generated_at, version = build_yql(
        args.output,
        input_table=args.input_table,
        output_table=args.output_table,
    )
    print(f"YQL: {path}")
    print(f"Generated at: {generated_at}")
    print(f"Algorithm version: {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
