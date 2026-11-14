"""Построение распределения (CDF) для числовых и дискретных вопросов Metaculus.

Metaculus принимает числовой прогноз как список значений функции распределения
в равномерно расположенных точках шкалы вопроса (201 точка для числовых,
`inbound_outcome_count + 1` для дискретных). Здесь процентили, которые дала
модель, превращаются в такой список и приводятся к требованиям API:
  - за закрытой границей вероятности нет, за открытой - не меньше 0.1%;
  - каждый шаг растёт не меньше чем на 0.01 / (n - 1);
  - ни один шаг не больше 0.2 * 200 / (n - 1) (с запасом 5%).
"""

from __future__ import annotations

import numpy as np

MAX_PMF_AT_201 = 0.2
PMF_CAP_MARGIN = 0.95


def scale_locations(
    n: int, range_min: float, range_max: float, zero_point: float | None
) -> np.ndarray:
    """Реальные значения шкалы вопроса в n равномерных точках (линейная или логарифмическая шкала)."""
    t = np.linspace(0.0, 1.0, n)
    if zero_point is None:
        return range_min + (range_max - range_min) * t
    deriv_ratio = (range_max - zero_point) / (range_min - zero_point)
    return range_min + (range_max - range_min) * (deriv_ratio**t - 1.0) / (deriv_ratio - 1.0)


def clean_percentiles(percentiles: dict[float, float]) -> list[tuple[float, float]]:
    """Сортирует процентили и делает значения строго возрастающими.

    Модель иногда даёт равные или слегка убывающие значения соседних процентилей;
    такие точки раздвигаются на малую величину, чтобы интерполяция была однозначной.
    """
    points = sorted((float(p) / 100.0, float(v)) for p, v in percentiles.items())
    points = [(q, v) for q, v in points if 0.0 < q < 1.0]
    if len(points) < 2:
        raise ValueError("нужно минимум два процентиля")
    span = max(abs(points[-1][1] - points[0][1]), abs(points[-1][1]), 1e-9)
    eps = span * 1e-9
    cleaned = [points[0]]
    for q, v in points[1:]:
        prev_v = cleaned[-1][1]
        if v <= prev_v:
            v = prev_v + eps
        cleaned.append((q, v))
    return cleaned


def raw_cdf(
    percentiles: dict[float, float],
    range_min: float,
    range_max: float,
    zero_point: float | None,
    open_lower: bool,
    open_upper: bool,
    n: int,
) -> np.ndarray:
    """Кусочно-линейная CDF по процентилям, вычисленная в точках шкалы вопроса."""
    points = clean_percentiles(percentiles)
    qs = [q for q, _ in points]
    vs = [v for _, v in points]

    # Хвосты за крайними процентилями: продолжаем с убывающей плотностью.
    # Шаг хвоста - расстояние, на котором при прежней плотности набралась бы вся масса хвоста.
    low_step = (vs[1] - vs[0]) * qs[0] / (qs[1] - qs[0])
    high_step = (vs[-1] - vs[-2]) * (1.0 - qs[-1]) / (qs[-1] - qs[-2])
    xs = [vs[0] - 4.0 * low_step, vs[0] - 1.5 * low_step] + vs + [
        vs[-1] + 1.5 * high_step,
        vs[-1] + 4.0 * high_step,
    ]
    ps = [0.0, qs[0] * 0.3] + qs + [1.0 - (1.0 - qs[-1]) * 0.3, 1.0]

    grid = scale_locations(n, range_min, range_max, zero_point)
    cdf = np.interp(grid, xs, ps)

    if not open_lower:
        cdf[0] = 0.0
    if not open_upper:
        cdf[-1] = 1.0
    return np.maximum.accumulate(np.clip(cdf, 0.0, 1.0))


def standardize_cdf(cdf: np.ndarray, open_lower: bool, open_upper: bool) -> list[float]:
    """Приводит CDF к требованиям Metaculus (минимальный и максимальный шаг, масса за границами)."""
    cdf = np.asarray(cdf, dtype=float).copy()
    n = len(cdf)

    scale_lower_to = 0.0 if open_lower else cdf[0]
    scale_upper_to = 1.0 if open_upper else cdf[-1]
    inbound_mass = scale_upper_to - scale_lower_to
    if inbound_mass <= 0:
        raise ValueError("вся масса распределения лежит за границами шкалы")

    location = np.linspace(0.0, 1.0, n)
    rescaled = (cdf - scale_lower_to) / inbound_mass
    if open_lower and open_upper:
        cdf = 0.988 * rescaled + 0.01 * location + 0.001
    elif open_lower:
        cdf = 0.989 * rescaled + 0.01 * location + 0.001
    elif open_upper:
        cdf = 0.989 * rescaled + 0.01 * location
    else:
        cdf = 0.99 * rescaled + 0.01 * location

    # Ограничение максимального шага: работаем с приращениями.
    pmf = np.diff(cdf, prepend=0.0, append=1.0)
    cap = MAX_PMF_AT_201 * (200.0 / (n - 1)) * PMF_CAP_MARGIN

    def capped(scale: float) -> np.ndarray:
        return np.concatenate([pmf[:1], np.minimum(cap, scale * pmf[1:-1]), pmf[-1:]])

    lo = hi = scale = 1.0
    for _ in range(200):
        if capped(hi).sum() >= 1.0:
            break
        hi *= 1.2
    for _ in range(100):
        scale = 0.5 * (lo + hi)
        total = capped(scale).sum()
        if total < 1.0:
            lo = scale
        else:
            hi = scale
        if total == 1.0 or (hi - lo) < 2e-5:
            break
    pmf = capped(scale)
    pmf[1:-1] *= (cdf[-1] - cdf[0]) / pmf[1:-1].sum()
    cdf = np.cumsum(pmf)[:-1]
    return np.round(cdf, 10).tolist()


def build_cdf(
    percentiles: dict[float, float],
    range_min: float,
    range_max: float,
    zero_point: float | None,
    open_lower: bool,
    open_upper: bool,
    n: int = 201,
) -> list[float]:
    """Полный путь: процентили модели -> список значений CDF для отправки в Metaculus."""
    cdf = raw_cdf(percentiles, range_min, range_max, zero_point, open_lower, open_upper, n)
    result = standardize_cdf(cdf, open_lower, open_upper)
    check_cdf(result, open_lower, open_upper)
    return result


def check_cdf(cdf: list[float], open_lower: bool, open_upper: bool) -> None:
    """Проверяет требования API; при нарушении бросает ValueError (прогноз не отправляется)."""
    arr = np.asarray(cdf, dtype=float)
    n = len(arr)
    steps = np.diff(arr)
    min_step = 0.01 / (n - 1)
    max_step = MAX_PMF_AT_201 * (200.0 / (n - 1))
    if np.any(steps < min_step * 0.999):
        raise ValueError(f"шаг CDF меньше минимального {min_step}")
    if np.any(steps > max_step * 1.0001):
        raise ValueError(f"шаг CDF больше максимального {max_step}")
    if open_lower and arr[0] < 0.001 - 1e-9:
        raise ValueError("за открытой нижней границей меньше 0.1%")
    if not open_lower and abs(arr[0]) > 1e-9:
        raise ValueError("за закрытой нижней границей есть масса")
    if open_upper and arr[-1] > 0.999 + 1e-9:
        raise ValueError("за открытой верхней границей меньше 0.1%")
    if not open_upper and abs(arr[-1] - 1.0) > 1e-9:
        raise ValueError("за закрытой верхней границей есть масса")
