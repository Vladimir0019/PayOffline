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

Если процент WoW определить нельзя, для необычного lifecycle-состояния с
ненулевым `ΔGMV` используется `z_cap`; иначе z-score равен нулю.

Lifecycle:

| Состояние | Условие |
|---|---|
| `новый сегмент` | Текущий GMV > 0, до него не было ненулевых недель |
| `возобновившийся сегмент` | Текущий GMV > 0, предыдущий = 0, раньше история была |
| `исчезнувший сегмент` | Текущий GMV = 0, предыдущий > 0 |
| `обычный` | Остальные случаи |

Надёжность истории:

| Ненулевых исторических недель | `reliability_factor` |
|---:|---:|
| ≥ 8 | 1.0 |
| 4–7 | 0.7 |
| 1–3 | 0.4 |
| 0 | 0.4 |

Дополнительно считаются WoW-проценты для доступных
`tx/au/am/aov/tpm/freq`.

## Этап 2. Кандидаты и первичный фильтр

`build_anomaly_candidates`:

1. Проверяет наличие выбранной текущей недели.
2. Восстанавливает положительный `total_by_date`.
3. Считает метрики для каждого `segment_id`.
4. Считает gross movement на глобальной максимальной глубине:

   ```text
   gross_atomic_movement = Σ abs(wow_delta_gmv атомов)
   materiality_share = abs(wow_delta_gmv сегмента) / gross_atomic_movement
   ```

5. Ставит `passes_initial_anomaly_filter`, если одновременно:

   - `slice_depth > 0`;
   - `abs_z_capped >= min_z_score`;
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

## Этап 4. Локальный штраф глубины

`apply_local_depth_penalty` сначала считает:

```text
base_anomaly_score =
    abs_z_capped × materiality_share × reliability_factor
```

Для каждого eligible-сегмента ищутся более глубокие eligible-потомки, чьё
coverage является подмножеством coverage родителя:

```text
local_depth_gap = max_eligible_descendant_depth - current_depth
depth_score_weight = depth_factor ^ local_depth_gap
anomaly_score = base_anomaly_score × depth_score_weight
```

По умолчанию `depth_factor = 0.9`. Если в ветке нет более глубокого eligible
потомка, вес равен 1.

Пустое/отсутствующее coverage или нечисловой base score у eligible-кандидата
останавливают расчёт.

## Важные ограничения

- `robust_z_capped` и `abs_z_capped` фактически не ограничиваются `z_cap`:
  строки cap закомментированы.
- `abnormal_gmv` сейчас равен простому `wow_delta_gmv`, а не отклонению от
  прогнозного GMV baseline.
- Исторические нули, появившиеся из пропусков источника, участвуют в lifecycle
  и baseline.
- Материальность нормируется на gross movement только максимальной глубины.

Следующий этап: [`set_packing.py.md`](set_packing.py.md).
