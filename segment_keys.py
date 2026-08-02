"""Единственный разбор человекочитаемого ключа сегмента.

Модуль хранит канонический разделитель ``segment_key`` и единственную функцию
его разбора. Раньше в пакете существовали две независимые копии парсера с
разным поведением: одна молча пропускала неразбираемую часть, вторая
выбрасывала исключение. Копии разошлись, поэтому один и тот же ключ мог быть
принят на этапе валидации входа и отвергнут на этапе построения покрытия.

Инварианты:

1. Ключ собирается только в ``data_preparation.build_segment_key_and_level``
   через ``SEGMENT_KEY_SEPARATOR``; здесь используется ровно тот же разделитель.
2. Любая неразбираемая часть — ошибка контракта, а не повод пропустить часть
   ключа: молчаливый пропуск приводил бы к заниженной глубине сегмента и
   неверному атомарному покрытию.
"""

from __future__ import annotations

from typing import List, Tuple


# ADDED: Канонический разделитель ключа. Значение обязано совпадать с тем,
# которым build_segment_key_and_level склеивает части ключа.
SEGMENT_KEY_SEPARATOR = " × "

# ADDED: Разделитель имени признака и значения внутри одной части ключа.
SEGMENT_KEY_ASSIGNMENT = "="

# ADDED: Ключ total-слоя; признаков не содержит и разбирается в пустой список.
TOTAL_SEGMENT_KEY = "ИТОГО"

# FIXED: Разбор идёт по точной строке-разделителю, а не по регулярному
# выражению с альтернативами. Прежний regex допускал латинскую "x" и
# мойибаку "Г—" (символ × в UTF-8, прочитанный как cp1251). Латинская "x"
# ломала разбор значений вида "A x B", а "Г—" не встречается в данных вообще.


def parse_segment_key_parts(segment_key: object) -> List[Tuple[str, str]]:
    """Разобрать ключ сегмента на пары «признак — значение».

    Args:
        segment_key: Человекочитаемый ключ сегмента, собранный через
            ``SEGMENT_KEY_SEPARATOR``. Значение ``ИТОГО`` и пустая строка
            описывают total-слой и дают пустой список.

    Returns:
        Список пар ``(dimension, value)`` в порядке следования в ключе.

    Raises:
        ValueError: Если непустая часть ключа не содержит ``=`` либо имя
            признака или его значение пусто.

    Examples:
        >>> parse_segment_key_parts("products=FULLPAYMENT × merchants_type=SMB")
        [('products', 'FULLPAYMENT'), ('merchants_type', 'SMB')]
        >>> parse_segment_key_parts("ИТОГО")
        []
        >>> parse_segment_key_parts("geo=РФ")
        [('geo', 'РФ')]
    """

    text = str(segment_key).strip()
    # ADDED: Total-слой не содержит ни одного признака и разбирается в пустой
    # список, а не в ошибку контракта.
    if not text or text == TOTAL_SEGMENT_KEY:
        return []

    parts: List[Tuple[str, str]] = []
    for raw_part in text.split(SEGMENT_KEY_SEPARATOR):
        part = raw_part.strip()
        if not part:
            continue
        if SEGMENT_KEY_ASSIGNMENT not in part:
            raise ValueError(
                f"Некорректная часть segment_key без '=': {part!r}"
            )
        dimension, value = part.split(SEGMENT_KEY_ASSIGNMENT, 1)
        dimension = dimension.strip()
        value = value.strip()
        if not dimension or not value:
            raise ValueError(
                f"Некорректная пустая dimension/value в segment_key: {part!r}"
            )
        parts.append((dimension, value))
    return parts


def segment_feature_set_from_key(segment_key: object) -> frozenset[Tuple[str, str]]:
    """Восстановить набор признаков сегмента из человекочитаемого ключа.

    Args:
        segment_key: Значение колонки ``segment_key``.

    Returns:
        Набор пар ``(dimension, value)`` без сохранения порядка.

    Raises:
        ValueError: Если ключ не разбирается по контракту.

    Examples:
        >>> segment_feature_set_from_key('geo=RF × product=QR') == frozenset(
        ...     {('geo', 'RF'), ('product', 'QR')}
        ... )
        True
    """

    return frozenset(parse_segment_key_parts(segment_key))


__all__ = [
    "SEGMENT_KEY_ASSIGNMENT",
    "SEGMENT_KEY_SEPARATOR",
    "TOTAL_SEGMENT_KEY",
    "parse_segment_key_parts",
    "segment_feature_set_from_key",
]
