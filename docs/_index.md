# База знаний `gmv_anomaly`

Актуальность: 2026-07-30. База описывает модульную реализацию поиска аномальных
GMV-сегментов и актуальную patch-версию upstream-запроса `pred_insight.yql`.

## Что делает система

Пакет получает историческую иерархическую витрину GMV, восстанавливает полную
сетку `сегмент × неделя`, оценивает необычность текущего WoW-изменения,
выбирает точный непересекающийся набор аномалий через Maximum Weighted Set
Packing и формирует Excel-отчёт с необязательным DAG-графом.

```text
pred_insight.yql
  → payoffline_pulse_hier / Excel
  → data_preparation.py
  → anomaly_scoring.py
  → set_packing.py
  → reporting.py
  → Excel + PNG/SVG/PDF
```

Оркестрация находится в `pipeline.py`, безаргументный запуск — в `main.py`,
параметры — в `config.py`.

## Быстрая маршрутизация

| Задача | Читать |
|---|---|
| Понять входную витрину, периоды, TOP-5, единицы GMV | [`pred_insight.yql.md`](pred_insight.yql.md) |
| Изменить загрузку, признаки, `segment_id`, пропуски, недельную сетку | [`data_preparation.py.md`](data_preparation.py.md) |
| Изменить robust z-score, lifecycle, материальность или depth penalty | [`anomaly_scoring.py.md`](anomaly_scoring.py.md) |
| Изменить конфликты, coverage, solver, статусы отбора | [`set_packing.py.md`](set_packing.py.md) |
| Изменить Excel-листы, менеджерский вывод или граф | [`reporting.py.md`](reporting.py.md) |
| Изменить безаргументный запуск | [`main.py.md`](main.py.md) |
| Проверить известные слабые места бизнес-логики | [`refactoring-findings.md`](refactoring-findings.md) |

## Сквозной контракт данных

Минимальный вход Python:

- `cal_date` — числовая временная ось; после приведения к `int` соседние недели
  total-слоя должны отличаться ровно на 7;
- `slice_depth` — глубина среза, `0` означает total;
- `gmv` — GMV сегмента;
- `period` — необязателен, но при наличии фильтруется по значению запуска;
- dimension columns строго заданы в `config.py`: `geo`, `products`,
  `merchants_type`, `is_terminal_or_cpqr`;
- все остальные колонки Excel, включая `segment_id`, `segment_key` и
  `segment_level`, считаются техническими (`ANOMALY_TECH_COLUMNS`) и не влияют
  на построение сегмента;
- `tx`, `au`, `am`, `aov`, `tpm`, `freq` необязательны и используются для
  менеджерского WoW-разложения.

Ключевые производные сущности:

| Сущность | Значение |
|---|---|
| `segment_id` | Значения всех dimension columns через `|`, пропуск = `∅` |
| `segment_key` | Только заполненные признаки: `geo=... × products=...` |
| атом | Строка глобальной максимальной `slice_depth` |
| coverage | Множество атомов, покрытых кандидатом |
| eligible-кандидат | Не total, прошёл пороги z-score, материальности и `abs(ΔGMV)` |
| итоговая аномалия | Eligible-кандидат, выбранный точным Set Packing |

## Основные формулы

```text
relative_wow = (gmv_current - gmv_previous) / gmv_previous
baseline = median(исторических relative_wow до текущей недели)
sigma = max(1.4826 × MAD, sigma_floor)
robust_z = (relative_wow - baseline) / sigma

materiality_share = abs(ΔGMV сегмента) / Σ abs(ΔGMV атомов)
base_anomaly_score = abs_z × materiality_share × reliability_factor
anomaly_score = base_anomaly_score × 0.9^local_depth_gap
```

Set Packing максимизирует сумму `anomaly_score`; каждый атом разрешено покрыть
не более одного раза.

## Инварианты, которые нельзя менять неявно

1. Total-слой задаёт полный календарь и должен иметь положительный GMV каждую неделю.
2. Отсутствующая строка сегмента сейчас трактуется как `gmv = 0`, но сохраняется
   флаг `row_missing_in_source`.
3. Первичный фильтр не удаляет строки: они остаются в диагностике.
4. Все прошедшие первичный фильтр сегменты должны получить доказанный статус
   Set Packing; неразрешённый кандидат останавливает расчёт.
5. Итоговый набор не содержит пересекающихся атомарных покрытий.
6. `robust_z_capped` исторически называется capped, но фактически не ограничен.
7. Excel-контракт состоит из девяти листов; его состав защищён регрессионным тестом.
8. До расчёта аномалий `data_preparation.py` сверяет `slice_depth` с количеством
   заполненных dimensions и отклоняет противоречивую metadata одного сегмента.

## Исключённые из отдельной документации файлы

- `__init__.py` — публичные re-export пакета.
- `__main__.py` — вызывает `main()`.
- `config.py` — пути запуска, dimension settings и `AnomalyThresholds`.
- `Anomaly.py` — compatibility wrapper старого API.
- `pipeline.py` — последовательная orchestration всех стадий.
- `test_gmv_anomaly_refactor.py` — пять регрессионных сценариев: конфигурация
  запуска, lifecycle, coverage/Set Packing, пустой результат, Excel-контракт.

## Проверка

Базовый тестовый прогон:

```powershell
python -m unittest gmv_anomaly.test_gmv_anomaly_refactor
```

На момент создания базы: 5 тестов проходят.
