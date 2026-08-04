# Dominance cap для единственного сильного потомка

Статус: exact contribution для долевых метрик реализован и проверен 2026-08-04.

## Зачем введено правило

Базовый score остаётся без изменений:

```text
base_anomaly_score =
    abs_robust_z × materiality_share × reliability_factor
```

Проблема находилась в иерархической корректировке. Родитель с одним сильным
потомком получал только фиксированный коэффициент `0.85` и мог обогнать
потомка за счёт агрегированного GMV, даже если почти всё движение родителя
физически находилось в одной узкой ветке.

Бизнес-принцип решения:

> Если широкий сегмент почти полностью объясняется одной узкой
> однонаправленной аномалией, в итоговом наборе приоритет получает узкая
> аномалия. Если движение распределено по нескольким веткам, родитель может
> остаться лучшим представлением.

## Формула

Правило применяется только при
`hierarchy_best_group_size = 1`. Для единственного сильного потомка `child`:

```text
capture =
    Σ abs(Contribution(atom | parent)), атом ∈ coverage(child)
    / Σ abs(Contribution(atom | parent)), атом ∈ coverage(parent)

uncapped_parent_score =
    base_anomaly_score(parent) × single_child_factor

cap_score =
    anomaly_score(child) × (1 − dominant_child_score_margin)
```

Если одновременно:

```text
capture ≥ dominant_child_capture_threshold
sign(net Contribution(child | parent)) = sign(R1(parent) − R0(parent))
```

то:

```text
anomaly_score(parent) = min(uncapped_parent_score, cap_score)
```

Иначе остаётся прежнее правило:

```text
anomaly_score(parent) = uncapped_parent_score
```

Для GMV движение по-прежнему равно `ΔGMV`. Для долевой метрики каждый атом
пересчитывается в масштабе рассматриваемого родителя:

```text
Contribution_i = 1/2 × (
    (Δn_i − R0(parent) × Δd_i) / D1(parent)
    +
    (Δn_i − R1(parent) × Δd_i) / D0(parent)
)

Σ Contribution_i = R1(parent) − R0(parent)
```

Поэтому capture использует одну систему координат для всех атомов и ребёнка.
Eligibility по-прежнему определяется отдельно через валидность доли, robust z,
materiality и reliability.

Параметры по умолчанию:

| Параметр | Значение | Смысл |
|---|---:|---|
| `single_child_factor` | 0.85 | Базовый штраф родителя с одним потомком |
| `dominant_child_capture_threshold` | 0.80 | Минимальная доля объяснённого gross movement |
| `dominant_child_score_margin` | 0.02 | Запас потомка над ограниченным родителем |

Порог 80% отделяет локализованную аномалию от движения, где не менее 20%
gross movement остаётся вне сильного потомка. Запас 2% исключает ничьи и
не создаёт существенного нового масштаба штрафа.

## Проверка на данных 22.07

Вход: `payoffline_pulse_hier_22_07.xlsx`.

| Родитель | Потомок | Gross child | Gross parent | Capture |
|---|---|---:|---:|---:|
| `AllTime × QR` | `РФ × FULLPAYMENT × AllTime × QR` | 3 164 310 | 3 509 450 | 90.1654% |
| `SPLIT × Прочее` | `РФ × SPLIT × Прочее × QR` | 858 642 | 898 691 | 95.5437% |

Направление родителя и потомка совпадает в обоих случаях.

| Кандидат | Base score | До cap | Граница cap | Итоговый score | Выбран |
|---|---:|---:|---:|---:|---|
| `AllTime × QR` | 0.125015 | 0.106263 | 0.092023 | 0.092023 | нет |
| `РФ × FULLPAYMENT × AllTime × QR` | 0.093901 | — | — | 0.093901 | да |
| `SPLIT × Прочее` | 0.031120 | 0.026452 | 0.025587 | 0.025587 | нет |
| `РФ × SPLIT × Прочее × QR` | 0.026109 | — | — | 0.026109 | да |

Получен ожидаемый итог: оба детальных сегмента выигрывают своих родителей.

## Диагностика в Excel

В листы анализа и полной диагностики добавлены поля:

- `hierarchy_single_child_capture`;
- `hierarchy_single_child_direction_match`;
- `hierarchy_single_child_uncapped_score`;
- `hierarchy_dominance_cap_score`;
- `hierarchy_dominance_cap_applied`.
- `hierarchy_dominance_cap_status`.
- `hierarchy_parent_exact_metric_delta`;
- `hierarchy_parent_exact_gross_contribution`;
- `hierarchy_single_child_exact_net_contribution`.

Значения `hierarchy_dominance_cap_status`:

- `APPLIED` — правило сработало и ограничило score родителя;
- `NOT_APPLIED_RULE_NOT_MET` — capture ниже порога либо направления не совпали;
- `SKIPPED_NONFINITE_ATOMIC_MOVEMENT` — capture нельзя корректно посчитать:
  хотя бы у одного физического атома родителя движение неопределённо;
- `NOT_APPLICABLE` — у родителя нет единственного сильного потомка.

Параметры порога и запаса записываются в
`00_Параметры_и_контроль`, поэтому результат воспроизводим без чтения кода.

## Границы применимости и риски

- При `best_group_size = 0` score не корректируется.
- При одном потомке и `capture < 80%` сохраняется коэффициент `0.85`.
- При несовпадении направления cap не применяется независимо от capture.
- При `best_group_size ≥ 2` без изменений используется
  `direction_unity × balance`.
- Атом, отсутствующий в обеих сравниваемых неделях, имеет `Δn_i=Δd_i=0` и
  exact contribution `0`. Его собственная доля остаётся невалидной для solver.
- Если атом отсутствует только в одной неделе, его отсутствующие аддитивные
  компоненты равны нулю; exact contribution остаётся конечным и учитывает
  появление или исчезновение атома.
- Если `D0(parent)` или `D1(parent)` равен нулю, доля родителя не определена и
  он не проходит `metric_valid_for_scoring`; dominance для него не считается.
- Качество capture зависит от корректности атомарного coverage. До scoring
  parent/date автоматически сверяется с суммой покрытых атомов максимальной
  глубины; превышение допуска останавливает расчёт.

## Регрессионная проверка

Покрыты сценарии:

1. Низкий capture сохраняет прежний score родителя.
2. Capture ровно 80% и одинаковое направление включает cap.
3. Высокий capture при противоположном направлении не включает cap.
4. Группа из нескольких потомков сохраняет прежний coherence-расчёт.
5. Сумма exact-вкладов атомов равна изменению доли scope.
6. Отсутствие последней, предпоследней или обеих недель атома даёт конечный
   contribution; в последнем случае он равен нулю.
7. Parent-relative capture ровно 80% при совпадающем направлении включает cap.

Команда:

```powershell
python -m unittest gmv_anomaly.test_gmv_anomaly_refactor
```

Основные сценарии exact contribution покрыты unit- и end-to-end тестами.
