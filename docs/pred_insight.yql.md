# `pred_insight.yql` — upstream-контракт иерархической витрины

## Роль

Запрос формирует входную иерархическую витрину для `gmv_anomaly`. Он читает
транзакционный `payoffline_pulse_raw`, строит все комбинации четырёх измерений
для периодов `1W`, `4W`, `13W` и перезаписывает `payoffline_pulse_hier`.

Изученная актуальная версия:

`arcadia_payoffline_patch/fintech/fdt/payoffline/projects/qr_yandex_pay/bi/pred_insight.yql`.

Версия в `.arcadia_source` отличается: она ограничена `1W` и 13 неделями.
Patch-версия добавляет независимые полные окна `1W/4W/13W` и TOP-5 отдельно
по каждому периоду. Перед production-запуском нужно проверить, что patch
действительно перенесён в основной источник.

## Источник и результат

| Назначение | YT-путь |
|---|---|
| Источник | `//home/fdt/payoffline/projects/qr_yandex_pay/bi/payoffline_pulse_raw` |
| Результат | `//home/fdt/payoffline/projects/qr_yandex_pay/bi/payoffline_pulse_hier` |
| Кластер | `hahn` |

Запись выполняется через `INSERT INTO ... WITH TRUNCATE`: каждый запуск полностью
заменяет таблицу результата.

## Алгоритм по блокам

1. `$periods` задаёт `1W=7`, `4W=28`, `13W=91` дней.
2. `$source_bounds` находит фактические границы данных для каждого периода.
3. `$source_complete_periods` оставляет до 13 окон с `cal_date_index < 13` и
   исключает интервалы, чья левая или правая граница отсутствует в источнике.
4. `$merchant_amounts` считает captured-оборот предыдущих окон с индексами
   `1..12`; текущий интервал не участвует.
5. `$ranked_merchants` и `$top_merchants` выбирают TOP-5 `merchants_type`
   отдельно внутри каждого периода. Остальные значения превращаются в `Прочее`.
6. `$source_history` нормализует dimension values: большинство `NULL`
   заменяются на `Unknown`.
7. `$aggregated` одним `GROUPING SETS` строит 16 комбинаций четырёх измерений.
8. `$total_by_date` выделяет total-слой.
9. `$new_rows` рассчитывает производные метрики и долю сегмента в total GMV.
10. Результат полностью перезаписывает output table.

## Иерархия

Измерения:

1. `geo`;
2. `products`;
3. `merchants_type`;
4. `is_terminal_or_cpqr`.

`slice_depth` равна числу заполненных измерений:

| Глубина | Число grouping sets |
|---:|---:|
| 0 | 1 total |
| 1 | 4 |
| 2 | 6 |
| 3 | 4 |
| 4 | 1 атомарный слой |

Всего 16 grouping sets на каждую пару `(period, cal_date)`.
Каждый grouping set задаёт уникальный набор заполненных измерений, поэтому
результат содержит не более одной строки на
`period × cal_date × geo × products × merchants_type × is_terminal_or_cpqr`.
После фильтрации одного `period` это соответствует уникальному Python-ключу
`segment_id × cal_date`.

`users_newness` и `users_activity` в patch-версии читаются/нормализуются, но
не входят в `GROUPING SETS` и не попадают в результат.

## Выходные поля

| Поле | Семантика |
|---|---|
| `period`, `cal_date` | Период и конец интервала |
| `slice_depth` | Глубина 0–4 |
| четыре dimension columns | `NULL` означает, что измерение не входит в срез |
| `gmv` | Captured amount, делённый на 100; в этой витрине GMV в рублях |
| `tx` | Captured-транзакции |
| `au` | Уникальные `phone_token` captured-транзакций |
| `am` | Уникальные `merchant_id` captured-транзакций |
| `aov` | `gmv / tx` |
| `tpm` | `tx / am` |
| `freq` | `tx / au` |
| `share_in_total_gmv` | `gmv сегмента / total_gmv` той же даты и периода |

Важно: документация `payoffline_pulse_mvp.md` относится к Family B/MVP и
описывает `payoffline_pulse_hier_mvp`, где GMV хранится в копейках. Это другой
output. В `pred_insight.yql` для `payoffline_pulse_hier` GMV явно делится на 100.

## Контракт с Python

`gmv_anomaly.data_preparation.load_history_table` ожидает, что:

- экспорт `cal_date` после `pd.to_numeric` даёт целые значения с шагом 7;
- `slice_depth=0` присутствует на каждой неделе;
- total GMV положителен;
- выбранный `period` существует;
- максимальная глубина 4 действительно является физическим атомарным слоем.

Если формат выгрузки `Date` меняется, сначала проверь фактическое представление
`cal_date` в Excel/CSV: Python не парсит календарные строки как даты, а приводит
колонку к числу.

## Риски и проверки перед запуском

- TOP-5 динамичен между запусками и независим между периодами; история одного
  бренда может переходить между отдельной категорией и `Прочее`.
- `Unknown` может быть не бизнес-сегментом, а сигналом деградации заполненности.
- Полнота окна проверяется по общей границе периода, а не по каждому сегменту;
  пропуски отдельных segment rows позже станут нулями в Python.
- YQL-запрос модифицирует постоянную таблицу. Безопасная проверка — VALIDATE;
  фактический RUN должен быть осознанным.
- После изменения dimensions нужно синхронно обновить `DIM_COLUMNS` либо
  исключения технических колонок в Python и тесты coverage.
