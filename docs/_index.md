# База знаний `gmv_anomaly`

Актуальность: 2026-08-14. База описывает поиск GMV-аномалий, независимый контур
восьми относительных метрик, отдельный Python-модуль анализа GMV-тренда,
product-YQL с Python3 UDF и patch-версию upstream-запроса `pred_insight.yql`.

## Что делает система

Пакет получает историческую иерархическую витрину, восстанавливает полную сетку
`сегмент × неделя`, отдельно оценивает GMV и каждую включённую долю, а затем для каждой метрики
независимо выбирает оптимальный в пределах `gap_tolerance` непересекающийся набор
аномалий через Maximum Weighted Set
Packing и формирует Excel-отчёт с необязательным DAG-графом.

Модуль `trend_analysis.py` решает другую задачу: поверх той же полной панели
отдельно определяет текущий устойчивый GMV-тренд и максимум одну подтверждённую
смену направления. Его результаты не входят в anomaly score, hierarchy adjustment,
Set Packing, Excel-отчёт или production UDF.

```text
pred_insight.yql
  → payoffline_pulse_hier / Excel
  → data_preparation.py
      ↘ trend_analysis.py
          → legacy → 3 диагностических DataFrame
          → most_recent_cp → trend_most_recent_cp.py → 4 DataFrame
  → anomaly_scoring.py
  → set_packing.py
  → reporting.py → Excel + PNG/SVG/PDF
  → udf_runtime.py → единая YT-таблица 1W/4W/13W
```

Оркестрация находится в `pipeline.py`, безаргументный запуск — в `main.py`,
параметры — в `config.py`.

## Быстрая маршрутизация

| Задача | Читать |
|---|---|
| **Контракт входных данных: типы, NULL, окна, идентификаторы** | [`data_preparation.py.md`](data_preparation.py.md) |
| Определить текущий GMV-тренд или одну смену направления | раздел **«Независимый анализ GMV-тренда»** ниже и `trend_analysis.py` |
| Понять логику витрины, периоды, TOP-5, единицы GMV | [`pred_insight.yql.md`](pred_insight.yql.md) |
| Изменить загрузку, признаки, `segment_id`, пропуски, недельную сетку | [`data_preparation.py.md`](data_preparation.py.md) |
| Изменить robust z-score, lifecycle, материальность или hierarchy score | [`anomaly_scoring.py.md`](anomaly_scoring.py.md) |
| Понять правило доминирующего потомка и его калибровку | [`hierarchy-dominance-cap.md`](hierarchy-dominance-cap.md) |
| Изменить конфликты, coverage, solver, статусы отбора | [`set_packing.py.md`](set_packing.py.md) |
| Изменить Excel-листы, менеджерский вывод или граф | [`reporting.py.md`](reporting.py.md) |
| Изменить product-UDF или пересобрать YQL после изменения Python | `udf_runtime.py`, `build_yql.py` |
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
   глубины с абсолютным допуском `hierarchy_reconciliation_abs_tolerance = 0.01`.
   Относительный допуск намеренно не применяется.
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
14. Trend analysis работает только поверх `build_full_week_grid`: собственный
   календарь и восстановление пропущенных строк в нём отсутствуют.
15. Ведущие нули сегмента удаляются один раз до построения trend-window; любой
   ноль после первого положительного GMV остаётся реальным наблюдением.

## [ADDED] Независимый анализ GMV-тренда

Публичный вызов:

```python
from gmv_anomaly import (
    TrendThresholds,
    build_full_week_grid,
    build_trend_analysis,
    load_history_table,
)

history_df, dims, dates = load_history_table(input_path, period="1W")
panel_df = build_full_week_grid(history_df, dims, dates)
trend_result = build_trend_analysis(panel_df, dates, TrendThresholds())

trend_summary = trend_result["trend_summary"]
trend_windows = trend_result["trend_window_diagnostics"]
trend_changes = trend_result["trend_change_diagnostics"]
```

### Контракт и формулы

- Используется вся доступная последовательная история; технического лимита
  `N=13` нет. Типичная история около 13 точек — только эксплуатационный контекст.
- До первого `GMV > 0` точки удаляются. Внутренние и конечные нули сохраняются.
- Если активной истории нет, статус `NO_ACTIVE_HISTORY`; если осталось меньше
  четырёх точек — `INSUFFICIENT_HISTORY`.
- Единственная `evaluate_trend()` используется и для suffix-window, и для обеих
  сторон возможного breakpoint.
- Наклон `b` — медиана всех `m(m−1)/2` попарных наклонов Тейла–Сена;
  `M = median(GMV)`, `b_rel = b/M`, `T = b(m−1)/M`.
- Тренд требует одновременно `|T| ≥ 0.10`, долю направленных ненулевых
  переходов `P_count ≥ 0.75`, долю направленного абсолютного движения
  `P_move ≥ 0.75`, отношение тренда к остаточному шуму `Q ≥ 2.0` и совпадение
  знака первого изменения GMV с направлением тренда. Нулевое первое изменение
  не проходит это условие; последующие встречные изменения допустимы и
  учитываются через `P_count`, `P_move` и `Q`.
- При нулевом MAD остатков идеальная линия получает `Q=+inf`; если ненулевые
  остатки существуют, знаменателем становится mean absolute residual.
- Текущее направление задаёт первое подтверждённое окно среди последних
  `4, 5, …, N` точек. `NO_TREND` не обрывает расширение, противоположный
  подтверждённый тренд — обрывает.
- Для `N ≥ 8` проверяются все `k=4,…,N−4`, строго `left=values[:k]` и
  `right=values[k:]`. Нулевая модель — одна OLS-линия, альтернатива — непрерывная
  `a + bt + c·max(0,t−k)` при математическом `t=1,…,N`; в Python при
  `t=0,…,N−1` hinge равен `max(0,t−(k−1))`.
- Смена подтверждается только при `F ≥ 8`, двух устойчивых трендах и
  противоположных направлениях. Среди валидных решений с
  `F(k) ≥ 0.95·F_best` выбирается самый поздний `k`.

`trend_summary` содержит одну строку на сегмент. Две другие таблицы сохраняют
полную QA-диагностику каждого suffix-window и каждого допустимого `k`, включая
RSS, F, направления сторон, прохождение порогов и флаг выбранной точки.

### Зафиксированные ограничения MVP

1. Возвращается максимум одна смена направления независимо от длины истории;
   рекурсивной сегментации нет.
2. `F=8` — инженерный порог, а не доказанный уровень статистической значимости.
3. Пороги `10% / 75% / 75% / 2.0` пока одинаковы для всех длин окна и периодов
   `1W / 4W / 13W`; при заметном росте доступной истории нужна перекалибровка.
4. Концентрация движения, смена уровня, сезонность и автокорреляция ошибок не
   моделируются.
5. F-модель использует МНК и чувствительнее к выбросам, чем Тейл–Сен; защита —
   обязательное подтверждение устойчивого тренда на обеих сторонах.
6. Тейл–Сен реализован за `O(m²)`. Это дёшево для коротких рядов, но требует
   отдельного архитектурного решения, если N вырастет до сотен или тысяч.

### [ADDED] Selector модели тренда

Legacy API `build_trend_analysis(...)` не изменён и остаётся источником прежних
трёх таблиц. Общая точка выбора — `build_configured_trend_analysis(...)`:

```python
from gmv_anomaly import (
    TrendModelConfig,
    build_configured_trend_analysis,
)

legacy_result = build_configured_trend_analysis(panel_df, dates)

cp_result = build_configured_trend_analysis(
    panel_df,
    dates,
    model_config=TrendModelConfig(
        trend_search_method="most_recent_cp",
        most_recent_cp_cost="capped",
    ),
)
```

Доступны ровно два `trend_search_method`:

- `legacy` — прежний suffix/F-алгоритм, значение по умолчанию;
- `most_recent_cp` — exact penalized segmentation по рассчитанным независимым
  piecewise-linear segment costs.

`RUN_gmv_TREND.py` использует `TREND_MODEL_CONFIG` и общий
`build_configured_trend_analysis(...)`. При `most_recent_cp` он сохраняет PDF
в прежнем формате: GMV-ряд, подсвеченный текущий режим и пунктирная линия его
тренда; вертикальная красная линия отмечает structural CP. В PDF попадают
только сегменты с `current_trend_exists=True`; один structural CP без
подтверждённого тренда страницу не создаёт.

Неизвестные method/cost отклоняются через `ValueError`. Параметры нового метода
живут в отдельном frozen `TrendModelConfig` рядом с `TrendThresholds`, но не
смешаны с бизнес-порогами `evaluate_trend`.

### Most Recent CP: segment costs

Новый модуль `trend_most_recent_cp.py` получает только готовую полную панель.
Он переиспользует `trim_leading_zero_history`: ведущие нули удаляются один раз,
внутренние/конечные нули и восстановленные `build_full_week_grid` missing rows
остаются наблюдениями. Временная координата глобальна внутри активной истории:
`t = 0, ..., n_active - 1`. Сегменты независимы и не обязаны быть непрерывными,
поэтому модель видит как slope change, так и level shift.

Для каждого `[s,e)` длиной не меньше `L_min = 4` доступно три cost:

```text
ols:
  C1(s,e) = Σ u_t² = RSS(s,e) / sigma²

capped:
  C2(s,e) = min_(a,b) Σ min(u_t², K²), K = 2.0

huber:
  C3(s,e) = min_(a,b) Σ rho_delta(u_t), delta = 1.345
  rho_delta(u) = u²                         при |u| <= delta
                 2*delta*|u| - delta²       при |u| > delta

u_t = (y_t - a - b*t) / sigma
```

Huber намеренно использует `u²`, а не стандартную запись `0.5*u²`: это
сохраняет масштаб квадратичной части C1/C2 при общей penalty. C2 оптимизирует
сам bounded objective, а не делает ошибочный `OLS → clip residuals`.
Используется детерминированный active-set multi-start с четырьмя стартами:
OLS, Theil–Sen, линия крайних точек и median-level. Случайных стартов нет.

Все допустимые `C(s,e)` считаются один раз и кэшируются. Диагностика каждого
сегмента содержит `start/end/points`, global intercept, slope, RSS, cost,
семантический `cost_type` и `optimizer_status`.

### Единая sigma и perfect fit

Scale оценивается один раз на всю активную историю и используется всеми её
сегментами:

```text
d_t = y_t - y_(t-1)
sigma = median(|d_t - median(d)|) / (0.67448975 * sqrt(2))
```

Если основной estimator является машинным нулём, применяется фиксированная
цепочка без абсолютного рублёвого floor:

1. `OLS_RESIDUAL_MAD`: `1.4826 * MAD` residuals одной OLS-линии всей истории;
2. `MAE_FALLBACK`: mean absolute residual, когда residual MAD равен нулю, но
   residuals не являются машинно нулевыми;
3. `PERFECT_FIT`: sigma остаётся нулевой; машинно точный линейный segment имеет
   cost `0`, действительно нелинейный segment при нулевом scale — `+inf`, а не
   `NaN` или случайный ноль.

Итог всегда хранит `sigma` и `sigma_source`: `DIFF_MAD`,
`OLS_RESIDUAL_MAD`, `MAE_FALLBACK` либо `PERFECT_FIT`.

### Penalty, exact DP и профиль последнего CP

Для одного активного ряда:

```text
beta = 3 * ln(n_active)

F(t) = min(
    C(0,t),
    min_s [F(s) + C(s,t) + beta]
)

G(0)   = C(0,n)
G(tau) = F(tau) + C(tau,n) + beta
```

Penalty начисляется за changepoint, а не за первый сегмент. DP сохраняет
predecessor каждого prefix и восстанавливает всю оптимальную segmentation
прошлого. Затем явно рассчитывается полный `G(tau)` для `tau=0` и всех границ,
оставляющих не меньше четырёх точек с обеих сторон. Выбирается только минимум
`G`; при различии исключительно в scale-aware machine tolerance берётся более
поздний `tau`, и summary получает `numeric_tie_break_used=True`. Legacy-правило
`near_best_change_ratio=0.95` и recency penalty сюда не переносятся.

Outer DP глобально оптимален для уже рассчитанной матрицы `C(s,e)`. Для C1 это
closed-form OLS, для C3 objective выпуклый. Для невыпуклого C2 multi-start
solver не является доказательством глобального оптимума bounded regression;
документируется только минимальное найденное им значение. Отдельный unit-тест
сверяет C2 с независимым subset-reference на малом сегменте, а performance-тест
измеряет все 153 costs при `n=20`.

### Structural regime не равен confirmed trend

Выбранный `tau` определяет начало текущего структурного режима. После этого
suffix `[tau,n)` передаётся без нового business-классификатора в существующий
`evaluate_trend(...)`. Поэтому level shift может дать
`structural_change_detected=True`, но `current_trend_direction=NONE`.
Legacy `change_detected` не переиспользуется как синоним structural CP.

Для найденной границы дополнительно считаются:

```text
delta_slope = current_regime_slope - previous_regime_slope
level_shift = fitted_current(tau) - fitted_previous(tau)
```

Summary нового метода содержит метод/cost, raw/used history, sigma/source,
beta, structural CP и его даты, current regime, no-change/selected objective,
objective improvement, подтверждение текущего тренда, slope/level diagnostics и
JSON всех восстановленных breakpoints.

Возвращаются четыре таблицы:

- `trend_summary` — одна итоговая строка на сегмент;
- `trend_cp_profile` — полный `G(tau)`, включая `tau=0`;
- `trend_segment_diagnostics` — весь кэш допустимых `C(s,e)`;
- `trend_segmentation` — только сегменты выбранного решения с cumulative cost.

### Численно важный exact-plateau edge

При строго заданных формулах ряд `[100,100,100,100,300,300,300,300]` допускает
независимые линии с нулевым fit cost при `tau=4`, но fallback даёт
`sigma = 1.4826 * MAD(OLS residuals)`, поэтому `G(0) ≈ 5.971`, а
`G(4) = 3*ln(8) ≈ 6.238`. Строгий минимум — отсутствие CP. Реализация не
подменяет это решение запрещённым near-best коэффициентом. Возможность видеть
чистый level shift проверяется почти плоским scale-invariant сценарием с
ненулевым `DIFF_MAD`, где `tau=4` выигрывает по заданному objective.

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
На момент актуализации базы: **57 тестов проходят, 1 пропущен**.

Независимый трендовый контур:

```powershell
python -m unittest gmv_anomaly.test_trend_analysis
```

Файл: `test_trend_analysis.py`; **25 тест-методов проходят**, покрывая сценарии
T01–T50 из постановки, включая property/invariance проверки.

Запуск пайплайна на данных из `config.py`:

```powershell
python -m gmv_anomaly
```

Product-YQL после любого изменения алгоритма пересобирается одной командой из
`C:\Python`:

```powershell
python -m gmv_anomaly.build_yql
```

Команда заново упаковывает текущие Python-модули в `anomaly_prod.yql`, добавляет
в результат ровно одну строку `Тип строки = "Техническая информация"` с датой
сборки и hash версии. Перед заменой файла embedded-UDF импортируется в отдельном
локальном Python-процессе: забытая внутренняя зависимость сохраняет предыдущий
корректный YQL. Данные содержат только выбранные Set Packing аномалии и GMV-сегменты
с изменением структуры. Ошибка любого периода прерывает запрос до атомарной
перезаписи целевой YT-таблицы.
