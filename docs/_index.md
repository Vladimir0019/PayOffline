# База знаний `gmv_anomaly`

Актуальность: 2026-08-19. База описывает поиск GMV-аномалий, независимый контур
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
смену направления. Post-processing `trend_scoring.py` не меняет найденные тренды,
а рассчитывает для них отдельный score, hierarchy-множитель и exact Set Packing.
Trend-контур не входит в anomaly score и anomaly production UDF; его Excel/PDF
формирует `RUN_gmv_TREND.py`, а отдельную YT-витрину — `trend_prod.yql`.

```text
pred_insight.yql
  → payoffline_pulse_hier / Excel
  → data_preparation.py
      ↘ trend_analysis.py
          → legacy → 3 диагностических DataFrame
          → most_recent_cp → trend_most_recent_cp.py → 5 DataFrame
          → trend_scoring.py → score + hierarchy + exact Set Packing без K
          → trend_manager_output.py → «Менеджерский вывод»
          → trend_udf_runtime.py → trend_prod.yql
              → payoffline_pulse_hier_Trend
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
| Рассчитать score и выбрать непересекающиеся тренды | раздел **«Trend score, hierarchy и оптимизация»** ниже и `trend_scoring.py` |
| Понять логику витрины, периоды, TOP-5, единицы GMV | [`pred_insight.yql.md`](pred_insight.yql.md) |
| Изменить загрузку, признаки, `segment_id`, пропуски, недельную сетку | [`data_preparation.py.md`](data_preparation.py.md) |
| Изменить robust z-score, lifecycle, материальность или hierarchy score | [`anomaly_scoring.py.md`](anomaly_scoring.py.md) |
| Понять правило доминирующего потомка и его калибровку | [`hierarchy-dominance-cap.md`](hierarchy-dominance-cap.md) |
| Изменить конфликты, coverage, solver, статусы отбора | [`set_packing.py.md`](set_packing.py.md) |
| Изменить Excel-листы, менеджерский вывод или граф | [`reporting.py.md`](reporting.py.md) |
| Изменить anomaly product-UDF или пересобрать anomaly YQL | `udf_runtime.py`, `build_yql.py` |
| Изменить trend product-UDF или пересобрать trend YQL | `trend_udf_runtime.py`, `build_trend_yql.py` |
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

### [ADDED] Trend score, hierarchy и оптимизация

`trend_scoring.py` запускается **после** trend analysis. Он не меняет slopes,
границы активного окна, результат `evaluate_trend()`, structural CP или legacy
алгоритм. В hierarchy и оптимизации участвуют только строки с подтверждённым
текущим трендом и `slice_depth > 0`. Атомарные строки, не прошедшие trend-фильтры,
остаются источником фактических GMV-движений и coverage, но не становятся
кандидатами задачи.

Для сегмента `i` вводятся величины:

```text
trend_slope_abs_i       — модуль наклона текущего тренда, GMV за период;
trend_slope_relative_i  — модуль того же наклона относительно типичного GMV;
current_total_gmv       — GMV Total-сегмента в последнем периоде.

absolute_rate_share_i = |trend_slope_abs_i| / current_total_gmv
relative_rate_i        = |trend_slope_relative_i|
impact_i               = sqrt(absolute_rate_share_i × relative_rate_i)

direction_quality_i = sqrt(
    direction_count_share_i × direction_movement_share_i
)
```

Геометрическое среднее в `impact` заменяет произведение двух rate-метрик:
при фиксированном масштабе GMV score теперь линейно, а не квадратично зависит
от slope. Оно также сохраняется при умножении всей GMV-витрины на положительную
константу.

Минимальное качество направления вычисляется тем же способом из действующих
порогов `evaluate_trend`. Интервал от этого минимума до `1` линейно переводится
в сигнал `[-1; 1]`. Собственный score:

```text
own_adjustment_signal_i = direction_adjustment_signal_i
own_score_adjustment_i = 0.10 × own_adjustment_signal_i
own_score_adjustment_factor_i = 1 + own_score_adjustment_i
own_trend_score_i = impact_i × own_score_adjustment_factor_i
```

Поэтому качество направления может изменить impact максимум на `−10% / +10%`.
В Excel и PDF отдельно выводится само знаковое значение
`own_score_adjustment = 0.10 × own_adjustment_signal`: например, `−0.10`
означает штраф 10%, `+0.04` — бонус 4%, а множитель равен соответственно
`0.90` и `1.04`.
`trend_to_noise` остаётся обязательным фильтром и диагностикой `evaluate_trend`,
но повторно в score не входит.

**Длительность тренда намеренно не включена в score.** На фактической витрине
короткие окна сконцентрированы на глубоких срезах, а длинные — на верхних
уровнях. Общая нормировка длительности поэтому системно штрафовала бы глубокие
сегменты независимо от их бизнес-вклада. Это открытое ограничение, а не решение,
что длительность неважна: к ней нужно вернуться после выбора интерпретируемой
нормировки, не зависящей от `slice_depth`. В выходе ограничение фиксирует поле
`trend_duration_score_status = NOT_INCLUDED_BY_DESIGN`.

#### Атомарные движения на активных окнах

OLS-линия не экстраполируется на недели, которые не относятся к текущему тренду
потомка. Для родителя `P` берутся переходы `t−1 → t`, полностью лежащие в его
активном окне. Для каждого атома `a` считается наблюдаемая недельная дельта
`delta[a,t] = GMV[a,t] − GMV[a,t−1]`:

```text
parent_gross(P) = sum_t sum_{a in coverage(P)} |delta[a,t]|

child_movement(C | P) =
    sum_{t in parent_window ∩ child_window}
    sum_{a in coverage(C)} delta[a,t]
```

Если у родителя рост длится шесть недель, а текущий рост потомка — четыре недели
после двух недель падения, раннее падение остаётся в `parent_gross`, но не
подменяется ростом потомка. Это консервативно снижает capture при разных окнах.

#### Сильнейшая группа и отдельный hierarchy-множитель

Среди eligible-потомков любых более глубоких уровней выбирается непустая группа
с попарно непересекающимся атомарным coverage и максимальной суммой уже
финализированных `trend_score`. Полное покрытие родителя не обязательно. Tie-break
совпадает с anomaly hierarchy: меньшая группа, затем лексикографический порядок.

Для группы размера `k ≥ 2`, где `d_j = child_movement(j | P)`:

```text
G = sum_j |d_j|
q_j = |d_j| / G
direction_unity = |sum_j d_j| / G
dominant_share = max_j(q_j)
B_max = (1 − dominant_share) / (1 − 1/k)
effective_count = 1 / sum_j(q_j²)
B_eff = (effective_count − 1) / (k − 1)
balance = min(B_max, B_eff)
hierarchy_coherence = direction_unity × balance
hierarchy_factor = 1 + 0.30 × (hierarchy_coherence − 0.5)
trend_score_parent = own_trend_score_parent × hierarchy_factor
```

Диагностики ограничиваются `0..1`, поэтому полностью перенесён anomaly-диапазон
`hierarchy_factor ∈ [0.85; 1.15]`. При нулевом gross группы множитель нейтральный
`1.0`: данных для бонуса или штрафа нет.

#### Доминирующий потомок

Правило проверяется независимо от размера сильнейшей группы:

```text
capture(C | P) = |child_movement(C | P)| / parent_gross(P)
```

Доминирование подтверждено, только если `capture ≥ 0.85`, направления `P` и `C`
оба подтверждены бизнес-контрактом и совпадают, а знак наблюдаемого атомарного
движения потомка соответствует этому направлению. Тогда переносится anomaly-cap:

```text
uncapped_parent = own_trend_score_parent × 0.85
cap = trend_score_child × (1 − 0.02)
trend_score_parent = min(uncapped_parent, cap)
```

Если условию соответствуют несколько перекрывающихся потомков, выбирается
потомок с максимальным финализированным `trend_score`; далее tie-break идёт по
capture, большей глубине и лексикографическому ключу. Это важно для витрин, где
разные комбинации измерений описывают одинаковое атомарное движение.

Отличия от доминирующего потомка anomaly-алгоритма:

| Аспект | Аномалии | Тренды |
|---|---|---|
| Когда проверяется dominance | Только если сильнейшая hierarchy-группа состоит из одного потомка | Независимо от размера сильнейшей группы |
| Временной смысл | Одно текущее аномальное изменение, обычно WoW | Все недельные переходы активного окна родителя; для потомка — пересечение двух активных окон |
| Числитель capture | Gross-вклад атомов потомка: сумма модулей атомарных вкладов | Модуль net-движения потомка: модуль суммы направленных атомарных дельт |
| Знаменатель capture | Gross атомарное движение/вклад родителя в том же сравнении | Gross атомарное движение всего активного окна родителя |
| Направление | Совпадение знаков net anomaly-движений родителя и потомка | Оба сегмента прошли `evaluate_trend`, направления совпадают и знак наблюдаемого child movement им соответствует |
| Порог по умолчанию | `0.80` | `0.85` |
| Множитель и cap | `base_anomaly_score × 0.85`, затем cap ниже final score потомка на 2% | `own_trend_score × 0.85`, затем тот же cap ниже final trend score потомка на 2% |

Общее: в знаменателях используются все атомы покрытия, но потенциальным
доминирующим ребёнком может быть только сегмент, прошедший первичный фильтр
соответствующего алгоритма.

#### Exact Set Packing без K

Финальная задача переиспользует существующий доказуемый solver:

```text
maximize  sum_i trend_score_i × x_i

subject to:
    для каждого атома a:
        sum_{i: a in coverage(i)} x_i ≤ 1

    x_i ∈ {0, 1}
```

Ограничения на число строк `K` нет: выводятся все сегменты оптимального
непересекающегося набора. Total исключён. `RUN_gmv_TREND.py` сохраняет полный
пул на листе `Тренды`, выбранный набор на листе `Отобранные тренды`. PDF
показывает все eligible-сегменты, участвовавшие в оптимизации: победители Set
Packing выделены толстой тёмной рамкой, родители с доминирующим ребёнком
перечёркнуты крестом. В карточке раскрыты `impact`, значение
`0.10 × own_adjustment_signal` внутри множителя качества направления,
hierarchy-множитель, final score и ненулевая hierarchy-поправка с перечислением
объясняющих детей.

В правой верхней части карточки крупно показан `Итоговый score`. Слева его
формула разложена на три русскоязычных строки: `Базовый вклад (impact)`,
`Множитель качества направления` и `Множитель иерархии`; в скобках у множителей
видна соответствующая процентная поправка.

Основные ограничения текущей версии:

1. Разные активные окна сравниваются через их пересечение, а весь gross родителя
   остаётся в знаменателе capture. Это осознанно консервативно, но потомок с очень
   свежим сильным трендом может не пройти dominance.
2. Direction quality входит и в фильтр, и в бонус/штраф до 10%; это не двойное
   умножение на одну метрику, но кандидаты у порога получают минимальный factor.
3. Score ранжирует скорость изменения, а не накопленный денежный эффект за всё
   окно; пока duration исключена, два одинаковых slope разной длительности имеют
   одинаковый score.
4. Оптимум глобален только среди подтверждённых трендов. Слабый или шумный атом
   участвует в знаменателях и coverage, но не может быть выбран сам.

### Зафиксированные ограничения MVP

1. Возвращается максимум одна смена направления независимо от длины истории;
   рекурсивной сегментации нет.
2. `F=8` — инженерный порог, а не формальный критерий проверки гипотезы.
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

- `legacy` — прежний suffix/F-алгоритм, доступный в режиме совместимости;
- `most_recent_cp` — exact penalized segmentation по рассчитанным независимым
  piecewise-linear segment costs, значение по умолчанию.

`RUN_gmv_TREND.py` использует `TREND_MODEL_CONFIG` и общий
`build_configured_trend_analysis(...)`. При `most_recent_cp` PDF сохраняет
прежний GMV-ряд и линию последнего тренда, а также добавляет светло-зелёную зону
глобального тренда, тёмно-зелёную зону последнего тренда, типы локальных режимов
и красные пунктирные линии для всех CP внутри глобального тренда. В PDF попадают
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
beta = 3 * ln(n_active) * 1.5 = 4.5 * ln(n_active)

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

Эти прежние поля по-прежнему описывают ту модель, которая использовалась в
segment cost (`ols`, `capped` или `huber`). Они не участвуют в новой
post-classification и не смешиваются с OLS standard errors.

### Post-classification последнего structural CP

После окончательного выбора `tau` выполняется отдельный OLS diagnostic fit
предыдущего выбранного режима `[s,tau)` и текущего режима `[tau,n)`. Он не
меняет segment costs, DP, `G(tau)`, tie-breaking, penalty или коэффициенты
C1/C2/C3. Обе OLS-линии используют одну глобальную временную координату
активной истории `t=0,...,n-1`; предыдущая последняя точка имеет индекс
`tau-1`, текущая первая точка — `tau`.

Для режима `j`:

```text
Sxx_j = Σ_i (t_i - mean(t)_j)^2
```

Пусть отдельные OLS slopes равны `b_-` и `b_+`. Тогда slope diagnostics:

```text
delta_b = b_+ - b_-
SE(delta_b) = sigma * sqrt(1 / Sxx_- + 1 / Sxx_+)
Z_slope = abs(delta_b) / SE(delta_b)
```

Level shift оценивается строго на границе текущего режима `t=tau`, а не в двух
несвязанных локальных координатах:

```text
y_hat_-(tau) = a_- + b_- * tau
y_hat_+(tau) = a_+ + b_+ * tau
delta_L = y_hat_+(tau) - y_hat_-(tau)

h_j(tau) = 1 / n_j + (tau - mean(t)_j)^2 / Sxx_j
SE(delta_L) = sigma * sqrt(h_-(tau) + h_+(tau))
Z_level = abs(delta_L) / SE(delta_L)
```

Это uncertainty fitted mean. Дополнительный `+1`, который относился бы к
prediction variance нового наблюдения, не добавляется. В обеих формулах
используется существующая единая `sigma` всей активной истории, а не отдельные
оценки масштаба режимов.

Единый параметр `most_recent_cp_change_z_threshold` живёт в
`TrendModelConfig`; default равен `2.0`, значение должно быть конечным и строго
положительным. При `Z0 = most_recent_cp_change_z_threshold` действует таблица:

| Условие | `structural_change_type` |
|---|---|
| Structural CP отсутствует | `NONE` |
| `Z_level >= Z0`, `Z_slope < Z0` | `LEVEL_SHIFT` |
| `Z_level < Z0`, `Z_slope >= Z0` | `SLOPE_CHANGE` |
| `Z_level >= Z0`, `Z_slope >= Z0` | `LEVEL_AND_SLOPE` |
| Оба score ниже `Z0` | `WEAK_OR_UNCLASSIFIED` |

`Z_level` и `Z_slope` — диагностические standardized effect scores, а не
формальные результаты проверки гипотез после выбора: `tau` уже определён
changepoint-алгоритмом по тем же данным. Порог `2.0` является техническим
диагностическим порогом и не задаёт доказанный inferential criterion. При
машинно нулевой SE score равен `0`, если соответствующий effect также машинно
нулевой, и `+inf` в противном случае; произвольный epsilon к sigma или
знаменателю не добавляется.

`direction_change` — отдельное ортогональное свойство, а не взаимоисключающий
structural type. Для предыдущего выбранного режима, как и для текущего,
вызывается существующий `evaluate_trend()`. Значение True возможно только если
structural CP найден, обе стороны имеют `trend_exists=True`, оба направления
принадлежат `{GROWTH, DECLINE}` и различаются. Поэтому, например,
`LEVEL_AND_SLOPE` может одновременно иметь `direction_change=True`, а
`NONE -> GROWTH` и `GROWTH -> GROWTH` дают False.

Summary нового метода содержит метод/cost, raw/used history, sigma/source,
beta, structural CP и его даты, current regime, no-change/selected objective,
objective improvement, подтверждение текущего и предыдущего режимных трендов,
прежние slope/level diagnostics, отдельные OLS classification fields,
standard errors/scores, `structural_change_type`, `direction_change` и JSON всех
восстановленных breakpoints.

Возвращаются пять таблиц:

- `trend_summary` — одна итоговая строка на сегмент;
- `trend_cp_profile` — полный `G(tau)`, включая `tau=0`;
- `trend_segment_diagnostics` — весь кэш допустимых `C(s,e)`;
- `trend_segmentation` — сегменты выбранного решения с cumulative cost и
  дополнительными признаками локального/глобального тренда;
- `trend_changepoints` — все CP выбранной сегментации, их структурные типы,
  OLS-диагностики и признак вхождения в глобальный тренд.

### Глобальный тренд как надстройка

Глобальный тренд не заменяет и не изменяет последний тренд. Его поиск начинается
с последнего локального режима, если тот уже подтверждён существующим
`evaluate_trend()` как `GROWTH` или `DECLINE`, и движется назад по выбранной
сегментации.

Локальный режим относится к `FLAT`, только если он не прошёл подтверждение
направленного тренда и одновременно выполняются три scale-invariant условия:

```text
abs(relative_slope) <= global_flat_max_relative_slope = 0.01
abs(total_change)   <= global_flat_max_total_change  = 0.05
noise_ratio         <= global_flat_max_noise_ratio   = 0.05
```

Остальные неподтверждённые режимы получают тип `UNCONFIRMED`. В глобальный тренд
включаются только предшествующие подтверждённые режимы того же направления.
Один внутренний `FLAT` разрешён как `FLAT_BRIDGE`, если за ним найден следующий
подтверждённый режим того же направления. Лимит задаётся
`global_max_flat_bridge_regimes=1`; начальный, конечный или второй подряд `FLAT`
не включается.

Каждое предполагаемое расширение проверяется транзакционно: весь объединённый
ряд от начала кандидата до конца истории повторно проходит тот же
`evaluate_trend()` и обязан подтвердить исходное направление. Поэтому два
однонаправленных локальных тренда не объединяются, если разрыв уровня или общая
геометрия делают совокупный тренд неподтверждённым. Значительный неблагоприятный
level shift сохраняется как диагностика CP, но не является отдельным veto.

В Excel эта надстройка добавляет глобальные агрегаты в прежние строки summary,
лист `Структура глоб. тренда` с датами, длительностью и изменением GMV каждого
локального режима и лист `Структурные CP`. Прежние листы, последний тренд,
scoring и exact Set Packing сохраняют прежнюю семантику.

### Менеджерский вывод

`RUN_gmv_TREND.py` после trend scoring создаёт независимую от алгоритма таблицу
`manager_output` через `build_manager_trend_output(...)` и сохраняет её на лист
`Менеджерский вывод`. В неё попадают только `trend_eligible=True` сегменты.
Внутри Python dataframe сохраняет технические имена колонок, Excel показывает
те же значения с заголовками из `MANAGER_TREND_COLUMN_LABELS`, а product-YQL
назначает эти заголовки публичными именами полей YT. Атрибуты сегмента
добавляются динамически по `DIM_COLUMNS`, поэтому появление нового измерения не
требует переписывать builder.

Помимо текущего GMV, локальных и глобальных изменений, витрина рассчитывает
доли сегмента в Total GMV на границах глобального тренда, изменение доли в п.п.,
среднюю фактическую скорость GMV за период, вклад последнего и всех локальных
режимов в глобальное изменение. `Структура глобального тренда` — это компактная
хронологическая строка с длительностями режимов: при подтверждённом
противоположном режиме до старта глобального окна начинается с разворота,
иначе — с начала тренда; далее описывает FLAT, level shifts и ускорение или
замедление направления. Этот builder только читает готовые DataFrame и не
изменяет поиск CP, trend scoring или Set Packing.

### [ADDED] Product-YQL поиска трендов

`trend_udf_runtime.py` повторяет расчёт `RUN_gmv_TREND.py` на потоке строк YT:

1. через канонический `prepare_history_dataframe(...)` типизирует период из
   `config.PERIOD` и строит полную панель `build_full_week_grid(...)`;
2. запускает `build_configured_trend_analysis(...)` с методом
   `most_recent_cp` и действующими default-порогами;
3. без изменения результатов поиска запускает `build_trend_selection(...)`;
4. формирует итог исключительно существующей функцией
   `build_manager_trend_output(...)`.

Сборщик `build_trend_yql.py` по аналогии с anomaly-сборщиком упаковывает точные
исходники всех зависимостей в self-contained Python3 UDF, проверяет импорт в
отдельном локальном Python-процессе и только после успешной проверки заменяет
`trend_prod.yql`. Хеш версии и время сборки сохраняются в комментариях YQL, но
не добавляют техническую строку или технические колонки в бизнес-витрину.

Контракт запуска:

```powershell
cd C:\Python
python -m gmv_anomaly.build_trend_yql
```

По умолчанию сгенерированный YQL читает период `1W` из
`//home/fdt/payoffline/projects/qr_yandex_pay/bi/payoffline_pulse_hier` и
транзакционно перезаписывает
`//home/fdt/payoffline/projects/qr_yandex_pay/bi/payoffline_pulse_hier_Trend`.
Период берётся из `config.PERIOD`, поэтому изменение периода также требует
повторной генерации.

Выход содержит те же 33 бизнес-поля, что и Excel-лист `Менеджерский вывод`, с
теми же публичными заголовками. Единственное безопасное отличие: во всех шести
заголовках сокращение `п.п.` заменяется на `пп`, потому что точка недопустима в
имени поля YQL даже при экранировании и создаёт риск при подключении источника в
DataLens. Нормализация централизована в `_yt_output_label(...)`; генератор
проверяет отсутствие точек, обратных кавычек и дубликатов. Технические имена
сохраняются только во внутреннем контракте Python3 UDF.

Финальная запись не создаёт именованное промежуточное выражение `$result`:
`INSERT INTO ... WITH TRUNCATE` выполняет `SELECT` непосредственно из
`(REDUCE $input ...)`. Внутренний `segment_id` не попадает в Excel и YT. Вторая
Excel-строка с пояснениями колонок, начинающаяся со значения
`Читаемое описание сегмента.`, создаётся только Excel-writer и намеренно не
записывается в YT. Поэтому при одинаковом расчёте в YT на одну строку меньше,
чем в Excel-листе.

Даты в YT сохраняются как Excel serial day (`1899-12-30` = 0), чтобы новый
YQL-результат был совместим с исторически загруженным в эту таблицу Excel-листом.

### Численно важный exact-plateau edge

При строго заданных формулах ряд `[100,100,100,100,300,300,300,300]` допускает
независимые линии с нулевым fit cost при `tau=4`, но fallback даёт
`sigma = 1.4826 * MAD(OLS residuals)`, поэтому `G(0) ≈ 5.971`, а
`G(4) = 4.5*ln(8) ≈ 9.357`. Строгий минимум — отсутствие CP. Реализация не
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
python -m unittest gmv_anomaly.test_trend_most_recent_cp
python -m unittest gmv_anomaly.test_trend_scoring
python -m unittest gmv_anomaly.test_trend_manager_output
python -m unittest gmv_anomaly.test_trend_yql
```

`test_trend_analysis.py`: **26 тест-методов проходят**, покрывая сценарии
T01–T50 из постановки, включая property/invariance проверки.

`test_trend_scoring.py`: **8 тест-методов проходят**; отдельно проверяются
геометрический impact, scale invariance, отсутствие duration в score, пересечение
активных окон, anomaly-формула hierarchy, dominance при группе размера 2,
ограничивающий cap и отсутствие ограничения K.

`test_trend_most_recent_cp.py`: **24 тест-метода проходят**. Дополнительно
проверяются расширение глобального тренда, повторный `evaluate_trend()`, один
`FLAT_BRIDGE`, запрет двух последовательных и конечного `FLAT`, а также
согласованность глобального summary со структурой режимов и CP.

`test_trend_yql.py` защищает 33-колоночную схему YT, исключение внутреннего
`segment_id` и Excel-строки пояснений, преобразование дат в Excel serial day,
фильтр `config.PERIOD` и импорт self-contained embedded UDF до записи файла.

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

Trend product-YQL пересобирается отдельно:

```powershell
python -m gmv_anomaly.build_trend_yql
```

Команда создаёт `trend_prod.yql`; она не запускает запрос и не изменяет YT.
