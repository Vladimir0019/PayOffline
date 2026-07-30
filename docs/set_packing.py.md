# `set_packing.py`

## Назначение

Модуль выбирает глобально оптимальный непересекающийся набор аномалий. Задача:

```text
max Σ anomaly_score[i] × x[i]

для каждого атома a:
Σ x[i], где i покрывает a, ≤ 1

x[i] ∈ {0, 1}
```

Таким образом родитель и его потомок либо два частично пересекающихся сегмента
не могут одновременно объяснять один и тот же атом.

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
- `robust_z_capped`, `wow_delta_gmv`, `anomaly_score`.

`coverage` из `anomaly_scoring.build_atomic_coverage` предпочтителен. Если он
не передан, модуль восстанавливает покрытие из `segment_key` и явно маркирует
источник как `SEGMENT_KEY_FALLBACK`.

## Последовательность `search_anomal`

1. Проверяет обязательные колонки и дубли `segment_id`/`segment_key`.
2. Нормализует coverage, полученное из подготовленных данных.
3. Классифицирует каждую строку:

   - total и не прошедшие фильтр не входят в граф;
   - невалидные coverage/score считаются фатальными, если первичный фильтр пройден;
   - только положительный конечный `anomaly_score` становится eligible.

4. Через обратный индекс `atom → segments` строит граф конфликтов.
5. Делит граф на независимые компоненты `C001`, `C002`, ...
6. Решает каждую компоненту точным solver-ом.
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
2. `abs_z_capped`;
3. `materiality_share`;
4. `abs_abnormal_gmv`;
5. `reliability_factor`;
6. `segment_key`.

### `diagnostics`

Все входные кандидаты плюс eligibility, coverage, конфликты, решение, solver,
gap, objective, component metadata и причины исключения.

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
- Каждая компонента доказала optimum в пределах gap.
- Objective компоненты равен сумме score выбранных сегментов.
- `final_df` совпадает с выбранными ID solver-а.
- Глобальный objective согласован с суммой компонент.
- Ни один атом не покрыт двумя выбранными сегментами.

## Риски

- Production-результат зависит от доступности точного MILP solver-а для крупных компонент.
- Fallback coverage по `segment_key` менее надёжен, чем фактическое coverage.
- Валидация ключа, dimensions и глубины выполняется в `data_preparation.py` до
  построения кандидатов; Set Packing получает согласованный вход.
- Собственный branch-and-bound экспоненциален и рекурсивен.
