# `anomaly_scoring.py`

## Назначение

Модуль независимо оценивает необычность каждого сегмента, строит атомарное
покрытие и рассчитывает objective weight для последующего Set Packing.

## Этап 1. Метрики одного сегмента

`calculate_segment_anomaly` использует текущую и предыдущую недели, а baseline
строит только по прошлым относительным WoW-изменениям:

```text
relative_growth[t] = (gmv[t] - gmv[t-1]) / gmv[t-1], если gmv[t-1] > 0
baseline = median(relative_growth до текущего перехода)
MAD = median(abs(relative_growth - baseline))
sigma = max(1.4826 × MAD, sigma_floor)
robust_z = (current_relative_wow - baseline) / sigma
```

Перед расчётом требуется конечный `sigma_floor > 0`; нарушение вызывает
`ValueError`, а не случайное деление на ноль.

Если процент WoW определить нельзя, для необычного lifecycle-состояния с
ненулевым `ΔGMV` используется `lifecycle_z_score`; иначе z-score равен нулю.
Непрерывный `robust_z` не ограничивается сверху. Поля
`z_scale_source = MAD | SIGMA_FLOOR` и `z_uses_sigma_floor` показывают, чем
определён масштаб z-score.

Lifecycle:

| Состояние | Условие |
|---|---|
| `новый сегмент` | Текущий GMV > 0, до него не было ненулевых недель |
| `возобновившийся сегмент` | Текущий GMV > 0, предыдущий = 0, раньше история была |
| `исчезнувший сегмент` | Текущий GMV = 0, предыдущий > 0 |
| `обычный` | Остальные случаи |

Надёжность истории считается по числу **ненулевых исторических недель**
(`history_nonzero_weeks`)

| Ненулевых исторических недель | `reliability_factor` |
|---:|---:|
| ≥ 8 | 1.0 |
| 4–7 | 0.7 |
| 1–3 | 0.4 |
| 0 | **0.1** |

У сегмента без единой ненулевой недели baseline и MAD не определены, поэтому
он получает минимальный вес и попадает в аномалии только при действительно
большом движении GMV.

Дополнительно считаются WoW-проценты для `tx/au/am/aov/tpm/freq`. Для
`aov/tpm/freq` WoW остаётся `NaN`, если предыдущее или текущее отношение не
определено. Штатный NULL из YQL не превращается в ноль или ложные `-100%`.

## Этап 2. Кандидаты и первичный фильтр

`build_anomaly_candidates`:

1. Проверяет наличие выбранной текущей недели.
2. Для каждого parent/date сверяет GMV с суммой покрытых атомов глобальной
   максимальной глубины. Допуск задаёт
   `hierarchy_reconciliation_abs_tolerance=1e-4`; любое превышение немедленно
   останавливает расчёт.
3. Восстанавливает положительный `total_by_date`.
4. Считает метрики для каждого `segment_id`.
5. Считает gross movement на глобальной максимальной глубине:

   ```text
   gross_atomic_movement = Σ abs(wow_delta_gmv атомов)
   materiality_share = abs(wow_delta_gmv сегмента) / gross_atomic_movement
   ```

6. Ставит `passes_initial_anomaly_filter`, если одновременно:

   - `slice_depth > 0`;
   - `abs_robust_z >= min_z_score`;
   - `materiality_share >= min_materiality_share`;
   - `abs(wow_delta_gmv) >= min_anomaly_abs`.

Все сегменты сохраняются в результате независимо от фильтра.

## Этап 3. Атомарное покрытие

`build_atomic_coverage` считает атомами все строки глобальной максимальной
`slice_depth`. Кандидат покрывает атом, если каждое заполненное dimension value
кандидата совпадает с атомом.

Результат: `dict[segment_id, frozenset[atomic_segment_id]]`.

Это предполагает сбалансированную иерархию. Листья более коротких веток атомами
не считаются.

## Пилот долевой метрики

`calculate_ratio_segment_anomaly` и `build_ratio_anomaly_candidates` параметризованы
`RatioMetricSpec`. Сейчас включена только `authzone_tx_share`.

```text
metric_delta[t] = metric_value[t] - metric_value[t-1]
baseline = median(metric_delta до текущего перехода)
sigma = max(1.4826 × MAD, sigma_floor)
robust_z = (current_metric_delta - baseline) / sigma
```

`metric_value` берётся из YQL без повторного расчёта. Reliability в пилоте сохраняет GMV-правило.

Для `contribution_mode = exact_atomic` сначала считается точный signed-вклад
каждого атома `i` в изменение доли Total:

```text
Contribution_i = 1/2 × (
    (Δn_i − R0 × Δd_i) / D1
    +
    (Δn_i − R1 × Δd_i) / D0
)

Σ Contribution_i = R1 − R0
```

Materiality измеряет долю gross-вклада атомов, покрываемых сегментом:

```text
global_gross = Σ abs(Contribution_i) по всем атомам

materiality_share(segment) =
    Σ abs(Contribution_i), i ∈ coverage(segment)
    / global_gross
```

Она находится в `[0, 1]` и аддитивна для непересекающихся атомарных покрытий.
Отдельно сохраняется signed `exact_global_net_contribution(segment)`. Прежние
`numerator_current / atomic_numerator_total` и
`metric_delta × mean_denominator` остаются в полях
`legacy_materiality_share`/`legacy_hierarchy_movement` только для сравнения.

Отдельного фильтра `min_numerator_scale` нет. Eligible-фильтр требует только
определённую текущую и предыдущую долю, `slice_depth > 0`, z- и materiality-пороги.
Нулевой числитель трактуется как бизнес-событие: `новый`, `возобновившийся` или
`исчезнувший`; такая строка остаётся в long-диагностике, даже если не eligible.

Pipeline вызывает `search_anomal` для GMV и для каждой доли отдельно. Технические
`hierarchy_movement` и `wow_delta_gmv` в exact-режиме — compatibility-alias
`exact_global_net_contribution`, а не GMV.

## Этап 4. Hierarchy-корректировка score

`apply_hierarchy_score_adjustment` сначала считает:

```text
base_anomaly_score =
    abs_robust_z × materiality_share × reliability_factor
```

Алгоритм работает только с eligible-кандидатами. Все атомарные сегменты
используются для построения coverage и проверки пересечений, но неeligible
сегменты не участвуют в группах и расчёте `U/B`.

Расчёт идёт снизу вверх по глубине. Для каждого eligible-родителя `p`
определяются eligible-потомки любых более глубоких уровней:

```text
D(p) = {
    c:
    slice_depth(c) > slice_depth(p)
    and coverage(c) ⊆ coverage(p)
}
```

Физически перечисляются все непустые группы `P ⊆ D(p)`, в которых атомарные
покрытия попарно не пересекаются. Группа не обязана полностью покрывать
родителя. Ограничений на смешение глубин нет.

Сильнейшая группа:

```text
P*(p) = argmax_P Σ anomaly_score(c), c ∈ P
```

Используется уже финализированный `anomaly_score` потомков. При равной сумме
предпочитается меньшее число сегментов, затем лексикографически меньший набор
`segment_id`.

Если допустимых потомков нет:

```text
hierarchy_score_factor = 1
```

Если сильнейшая группа состоит из одного потомка, GMV сохраняет расчёт по
`ΔGMV`. Для доли exact-вклады атомов пересчитываются относительно конкретного
родителя `p`, то есть с его `R0(p)`, `R1(p)`, `D0(p)`, `D1(p)`:

```text
single_child_uncapped_score =
    base_anomaly_score(parent) × single_child_factor

capture =
    Σ abs(Contribution(atom | parent)), атом ∈ coverage(child)
    / Σ abs(Contribution(atom | parent)), атом ∈ coverage(parent)

direction_match =
    sign(Σ Contribution(atom | parent), atom ∈ coverage(child))
    = sign(R1(parent) − R0(parent))

dominance_cap_score =
    anomaly_score(child) × (1 − dominant_child_score_margin)

если capture ≥ dominant_child_capture_threshold и direction_match:
    anomaly_score(parent) =
        min(single_child_uncapped_score, dominance_cap_score)
иначе:
    anomaly_score(parent) = single_child_uncapped_score

hierarchy_score_factor =
    anomaly_score(parent) / base_anomaly_score(parent)
```

В знаменатель `capture` входят все атомы родителя, в том числе не прошедшие
первичный anomaly-фильтр. Абсолютные движения исключают искусственное
завышение доли из-за взаимопогашения. Проверка направления не позволяет
поднять узкий разнонаправленный сегмент как объяснение движения родителя.

Если в группе `k ≥ 2`, для изменений GMV её eligible-потомков:

```text
d_i = wow_delta_gmv(c_i)
G = Σ abs(d_i)
q_i = abs(d_i) / G

direction_unity = abs(Σ d_i) / G
dominant_share = max(q_i)

B_max = (1 − dominant_share) / (1 − 1/k)
effective_count = 1 / Σ q_i²
B_eff = (effective_count − 1) / (k − 1)
hierarchy_balance = min(B_max, B_eff)

hierarchy_coherence = direction_unity × hierarchy_balance
hierarchy_score_factor =
    1 + aggregation_bonus_lambda × (hierarchy_coherence − 0.5)
anomaly_score =
    base_anomaly_score × hierarchy_score_factor
```

Значения по умолчанию:

- `aggregation_bonus_lambda = 0.3`, поэтому коэффициент лежит в `[0.85; 1.15]`;
- `single_child_factor = 0.85`;
- `dominant_child_capture_threshold = 0.80`;
- `dominant_child_score_margin = 0.02`.

Штрафа за глубину больше нет.

Пустое/отсутствующее coverage или нечисловой base score у eligible-кандидата
останавливают расчёт.

Диагностика сохраняет число eligible-потомков, число физически перечисленных
групп, состав ID сильнейшей группы в `hierarchy_best_group_ids_json`, её score,
`U`, доминирующую долю, оба баланса,
итоговый баланс, coherence и коэффициент score. Для группы из одного потомка
дополнительно сохраняются `capture`, совпадение направления, score до cap,
граница cap и признак фактического применения ограничения.

Обоснование решения и проверка на кейсе от 22.07:
[`hierarchy-dominance-cap.md`](hierarchy-dominance-cap.md).

## Важные ограничения

- `robust_z` намеренно не ограничивается сверху; при `MAD=0` масштаб задаёт
  `sigma_floor`, что явно отражается в диагностике.
- Фактический денежный вклад хранится только в `wow_delta_gmv`; дублирующие
  aliases `abnormal_gmv` и `abs_abnormal_gmv` удалены.
- Нули аддитивных показателей, восстановленные для действительно отсутствующих
  строк источника, участвуют в lifecycle и baseline; повреждённые существующие
  строки отклоняются до scoring.
- NULL ratio-метрик сохраняется и не участвует в их WoW-сравнении.
- Материальность нормируется на gross movement только максимальной глубины.
- Полное физическое перечисление групп имеет экспоненциальную сложность:
  при `n` попарно непересекающихся потомках создаётся `2^n − 1` групп. Поэтому
  число eligible-потомков одного родителя ограничено параметром
  `max_hierarchy_descendants = 25`: превышение останавливает расчёт с
  диагностикой.

Следующий этап: [`set_packing.py.md`](set_packing.py.md).
