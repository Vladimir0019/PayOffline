# База знаний `gmv_anomaly`

Актуальность: 2026-08-04. База описывает поиск GMV-аномалий, независимый контур
восьми относительных метрик и patch-версию upstream-запроса `pred_insight.yql`.

## Что делает система

Пакет получает историческую иерархическую витрину, восстанавливает полную сетку
`сегмент × неделя`, отдельно оценивает GMV и каждую включённую долю, а затем для каждой метрики
независимо выбирает оптимальный в пределах `gap_tolerance` непересекающийся набор
аномалий через Maximum Weighted Set
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
| **Контракт входных данных: типы, NULL, окна, идентификаторы** | [`data_preparation.py.md`](data_preparation.py.md) |
| Понять логику витрины, периоды, TOP-5, единицы GMV | [`pred_insight.yql.md`](pred_insight.yql.md) |
| Изменить загрузку, признаки, `segment_id`, пропуски, недельную сетку | [`data_preparation.py.md`](data_preparation.py.md) |
| Изменить robust z-score, lifecycle, материальность или hierarchy score | [`anomaly_scoring.py.md`](anomaly_scoring.py.md) |
| Понять правило доминирующего потомка и его калибровку | [`hierarchy-dominance-cap.md`](hierarchy-dominance-cap.md) |
| Изменить конфликты, coverage, solver, статусы отбора | [`set_packing.py.md`](set_packing.py.md) |
| Изменить Excel-листы, менеджерский вывод или граф | [`reporting.py.md`](reporting.py.md) |
| Изменить безаргументный запуск | [`main.py.md`](main.py.md) |

## Сквозной контракт данных

Полный контракт входных данных — в [`data_preparation.py.md`](data_preparation.py.md).
Ниже только минимум для ориентации.

Минимальный вход Python:

- `cal_date` — числовая временная ось; после приведения к `int` соседние недели
  total-слоя должны отличаться ровно на `7 × N` дней, где `period = NW`;
- `slice_depth` — глубина среза, `0` означает total;
- `gmv` — GMV сегмента;
- `period` — обязательный параметр запуска и обязательная колонка: определяет
  фильтр данных и шаг `cal_date`;
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
| `segment_id` | Компактный **JSON-массив** значений всех dimension columns по фиксированному порядку, пропуск = JSON `null`: `["РФ",null,"SMB",null]` |
| `segment_key` | Только заполненные признаки через ` × `: `geo=РФ × products=QR`. Разбор — единственной функцией `segment_keys.parse_segment_key_parts` |
| атом | Строка глобальной максимальной `slice_depth` |
| coverage | Множество атомов, покрытых кандидатом |
| eligible-кандидат | Не total, прошёл пороги z-score, материальности и `abs(ΔGMV)` |
| итоговая аномалия | Eligible-кандидат, выбранный Set Packing в пределах `gap_tolerance` |

## Основные формулы

```text
relative_wow = (gmv_current - gmv_previous) / gmv_previous
baseline = median(исторических relative_wow до текущей недели)
sigma = max(1.4826 × MAD, sigma_floor)
robust_z = (relative_wow - baseline) / sigma

materiality_share = abs(ΔGMV сегмента) / Σ abs(ΔGMV атомов)
base_anomaly_score = abs_robust_z × materiality_share × reliability_factor
hierarchy_balance = min(B_max, B_eff)
hierarchy_coherence = direction_unity × hierarchy_balance
hierarchy_score_factor = 1 + 0.3 × (hierarchy_coherence − 0.5)
anomaly_score = base_anomaly_score × hierarchy_score_factor
```

Для каждого eligible-родителя физически перечисляются все непустые попарно
непересекающиеся группы eligible-потомков любых более глубоких уровней.
Неполное покрытие родителя разрешено. Сильнейшая группа максимизирует сумму
уже скорректированных score потомков; расчёт идёт снизу вверх. Если сильнейшая
группа состоит из одного потомка, сначала используется коэффициент `0.85`.
Когда этот потомок совпадает с родителем по направлению и объясняет не менее
80% абсолютного движения атомов родителя, score родителя дополнительно
ограничивается значением `0.98 × anomaly_score` потомка. Так широкий сегмент,
фактически пересказывающий одну узкую аномалию, не вытесняет её за счёт
агрегированного GMV.

Set Packing максимизирует сумму `anomaly_score` среди прошедших первичные
фильтры кандидатов; каждый атом разрешено покрыть не более одного раза.
Полное покрытие total, минимальная объяснённая доля, совпадение знака с total,
равенство суммы вкладов total и заданное число сегментов не требуются.
Аномалии и компенсации любого знака могут попасть в итог в зависимости от score.

## Инварианты, которые нельзя менять неявно

0. Последняя неделя считается **полностью закрытой**; неполный интервал не
   попадает в витрину из-за границ `$source_bounds` в YQL. Исключать текущую
   неделю в Python не нужно.
1. Total-слой задаёт полный календарь и должен иметь положительный GMV каждую неделю.
2. Отсутствующая строка сегмента сейчас трактуется как `gmv = 0`, но сохраняется
   флаг `row_missing_in_source`.
3. Первичный фильтр не удаляет строки: они остаются в диагностике.
4. Все прошедшие первичный фильтр сегменты должны получить доказанный статус
   Set Packing; неразрешённый кандидат останавливает расчёт.
5. Итоговый набор не содержит пересекающихся атомарных покрытий.
6. `robust_z` не ограничивается сверху; источник масштаба отражают
   `z_scale_source` и `z_uses_sigma_floor`.
7. До scoring каждый parent/date сверяется с суммой покрытых атомов максимальной
   глубины с абсолютным допуском `hierarchy_reconciliation_abs_tolerance`.
8. Factual coverage обязательно для production-вызова `search_anomal`; fallback
   по `segment_key` требует `allow_segment_key_fallback=True`.
9. Excel-контракт состоит из девяти неизменённых GMV-листов и двух long-листов долевых
   метрик; его состав защищён регрессионным тестом.
10. До расчёта аномалий `data_preparation.py` сверяет `slice_depth` с количеством
   заполненных dimensions и отклоняет противоречивую metadata одного сегмента.
11. Set Packing даёт глобальный оптимум **в пределах множества кандидатов**,
   прошедших первичный фильтр, и допуска `set_packing_gap_tolerance`. Сегменты,
   отсечённые фильтром, в задачу не попадают.
12. `segment_key` разбирает единственная функция
   `segment_keys.parse_segment_key_parts`; неразбираемый ключ останавливает
   расчёт, а не пропускается.
13. Полное перечисление hierarchy-групп используется до
   `max_hierarchy_descendants = 25`; выше лимита сильнейшая группа выбирается
   точным Set Packing без перебора `2^n − 1` комбинаций.

## Исключённые из отдельной документации файлы

- `__init__.py` — публичные re-export пакета.
- `__main__.py` — вызывает `main()`.
- `config.py` — пути запуска, dimension settings и `AnomalyThresholds`.
- `Anomaly.py` — compatibility wrapper старого API.
- `pipeline.py` — последовательная orchestration всех стадий.
- `segment_keys.py` — канонический разделитель и единственный разбор
  `segment_key`; контракт описан в [`data_preparation.py.md`](data_preparation.py.md).
- `test_gmv_anomaly_refactor.py` — регрессионные и сценарные проверки
  конфигурации, lifecycle, coverage/Set Packing, полного mixed-level перебора,
  hierarchy balance, пустого результата и Excel-контракта.

## Проверка

Базовый тестовый прогон:

```powershell
python -m unittest gmv_anomaly.test_gmv_anomaly_refactor
```

Тесты: [`../test_gmv_anomaly_refactor.py`](../test_gmv_anomaly_refactor.py).
На момент актуализации базы: **32 теста проходят** (~2.3 с).

Запуск пайплайна на данных из `config.py`:

```powershell
python -m gmv_anomaly
```
