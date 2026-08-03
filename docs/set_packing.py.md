# `set_packing.py`

## Назначение

Модуль выбирает оптимальный непересекающийся набор аномалий. Задача:

```text
max Σ anomaly_score[i] × x[i]

для каждого атома a:
Σ x[i], где i покрывает a, ≤ 1

x[i] ∈ {0, 1}
```

Таким образом родитель и его потомок либо два частично пересекающихся сегмента
не могут одновременно объяснять один и тот же атом.

### Область оптимальности

Оптимум глобален **в пределах множества кандидатов**, прошедших первичный
фильтр аномальности, и допуска `set_packing_gap_tolerance`. Сегменты,
отсечённые порогами `min_z_score`, `min_materiality_share` и
`min_anomaly_abs`, в задачу не попадают вовсе, поэтому это не оптимум по всем
возможным сегментам витрины.

Пересечения определяются **только через атомарное coverage**, а не через
отношение «предок — потомок»: любое общее покрытие атома создаёт конфликт,
даже если сегменты не связаны иерархически.

## Главный API

```python
final_df, diagnostics, decision_log = search_anomal(
    candidates,
    thresholds,
    coverage=coverage,
)
```

Обязательные колонки:

- `segment_id`, `segment_key`, `slice_depth`;
- `passes_initial_anomaly_filter`;
- `robust_z`, `abs_robust_z`, `wow_delta_gmv`, `anomaly_score`.

`coverage` из `anomaly_scoring.build_atomic_coverage` обязателен для
production-вызова. Fallback по `segment_key` доступен только при явном
`allow_segment_key_fallback=True` и маркируется как `SEGMENT_KEY_FALLBACK`.

Factual coverage проходит строгую проверку: mapping покрывает все candidate
IDs, значения являются коллекциями, но не `str/bytes`, атомы существуют в
слое `max(slice_depth)`, каждый атом покрывает сам себя, а каждый eligible
кандидат имеет непустое покрытие.

До глобального Set Packing `anomaly_scoring.apply_hierarchy_score_adjustment`
корректирует score eligible-родителей по сильнейшей непересекающейся группе
eligible-потомков. Глобальная оптимизация получает уже финальный
`anomaly_score`; отдельного штрафа за глубину нет.

## Последовательность `search_anomal`

1. Проверяет обязательные колонки и дубли `segment_id`/`segment_key`.
2. Нормализует coverage, полученное из подготовленных данных.
3. Классифицирует каждую строку:

   - total и не прошедшие фильтр не входят в граф;
   - невалидные coverage/score считаются фатальными, если первичный фильтр пройден;
   - только положительный конечный `anomaly_score` становится eligible.

4. Через обратный индекс `atom → segments` строит граф конфликтов.
5. Делит граф на независимые компоненты `C001`, `C002`, ...
6. Решает каждую компоненту с доказанным optimum в пределах
   `set_packing_gap_tolerance`.
7. Заполняет diagnostics, сортирует выбранные строки и присваивает `rank`.
8. Обязательно вызывает `validate_set_packing_solution`.
9. Строит прозрачный `decision_log`.

## Каскад solver-ов

| Порядок | Solver | Когда используется |
|---:|---|---|
| 1 | `TRIVIAL` | В компоненте нет реальных конфликтов |
| 2 | Gurobi | Пакет установлен, лицензия доступна, optimum доказан |
| 3 | SciPy/HiGHS MILP | `scipy.optimize.milp` доступен, gap допустим |
| 4 | Собственный branch-and-bound | Предыдущие solver-ы не доказали optimum и размер ≤ `max_exact_fallback_size` |

Если внешние solver-ы не доказали optimum, а компонента больше лимита
(по умолчанию 25 сегментов), pipeline падает с `RuntimeError`.

Внутренний solver сортирует переменные по score, использует верхнюю границу
суммы оставшихся положительных весов и детерминированный tie-break:
при равном objective предпочитается меньше строк, затем лексикографический
порядок сегментов.

## Диагностические статусы

Основные значения `set_packing_status`:

| Статус | Значение |
|---|---|
| `NOT_IN_SET_PACKING_GRAPH` | Total или не прошёл первичный фильтр |
| `INVALID_ATOMIC_COVERAGE` | Coverage повреждён или отсутствует |
| `EMPTY_ATOMIC_COVERAGE` | Eligible-сегмент не покрывает атомы |
| `INVALID_SCORE` | Score не конечен |
| `NONPOSITIVE_SCORE` | Score ≤ 0 |
| `SET_PACKING_CANDIDATE` | Допущен к оптимизации |
| `SET_PACKING_SELECTED` | Выбран оптимумом |
| `SET_PACKING_NOT_SELECTED` | Корректно проиграл другому набору |
| `SET_PACKING_NOT_PROVEN` | Optimum не доказан; итоговая валидация не пропустит |

## Три результата

### `final_df`

Только выбранные сегменты, отсортированные по:

1. `selection_score`;
2. `abs_robust_z`;
3. `materiality_share`;
4. `abs(wow_delta_gmv)`;
5. `reliability_factor`;
6. `segment_key`.

### `diagnostics`

Все входные кандидаты плюс eligibility, coverage, конфликты, решение, solver,
gap, objective, component metadata и причины исключения. Численную устойчивость
objective показывают `set_packing_component_score_min`,
`set_packing_component_score_max` и
`set_packing_component_score_dynamic_range`.

### `decision_log`

Содержит события:

- `GLOBAL_OPTIMIZATION_SUMMARY`;
- `ATOM_TO_SEGMENTS`;
- `CONFLICT_PAIR`;
- `COMPONENT_SOLVE`;
- `SEGMENT_DECISION`.

## Что доказывает финальная валидация

- Глобальный статус `OPTIMAL`.
- Все прошедшие первичный фильтр строки разрешены как selected/not selected.
- Каждая компонента доказала optimum в пределах `gap_tolerance`.
- Objective компоненты равен сумме score выбранных сегментов.
- `final_df` совпадает с выбранными ID solver-а.
- Глобальный objective согласован с суммой компонент.
- Ни один атом не покрыт двумя выбранными сегментами.

## Риски

- Production-результат зависит от доступности точного MILP solver-а для крупных компонент.
- Fallback coverage по `segment_key` менее надёжен и поэтому включается только
  явным параметром. Он разбирает ключ общей функцией
  `segment_keys.parse_segment_key_parts`, поэтому неразбираемый ключ
  останавливает расчёт, а не даёт частичное покрытие.
- Валидация ключа, dimensions и глубины выполняется в `data_preparation.py` до
  построения кандидатов; Set Packing получает согласованный вход.
- Собственный branch-and-bound экспоненциален и рекурсивен.

## Проверка оптимальности

Помимо `validate_set_packing_solution`, оптимальность подтверждена независимым
brute-force на реальных данных `payoffline_pulse_hier_28_07.xlsx`: перебор всех
непересекающихся комбинаций до 5 сегментов из 28 eligible-кандидатов даёт тот
же objective `2.049265458`, что и solver.

Регрессионные тесты: `test_atomic_coverage_and_parent_child_set_packing`,
`test_score_dynamic_range_is_reported`, `test_factual_coverage_is_required_and_strict`,
`test_empty_anomaly_set` в [`../test_gmv_anomaly_refactor.py`](../test_gmv_anomaly_refactor.py).
