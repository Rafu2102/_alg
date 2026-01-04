import random
import math
from typing import Callable, List, Tuple

Vector = List[float]
Bounds = List[Tuple[float, float]]  # [(a1,b1), (a2,b2), ..., (an,bn)]


def riemann_nd_recursive(
    f: Callable[[Vector], float],
    bounds: Bounds,
    m: int,
    method: str = "midpoint"  # "left", "right", "midpoint"
) -> float:
    """
    n-dimensional Riemann sum using recursion.
    - f: function f(x) where x is a list of length n
    - bounds: list of (ai, bi) for each dimension
    - m: number of subdivisions per dimension
    - method: choose sample point in each cell
    """
    n = len(bounds)
    if n == 0:
        raise ValueError("bounds must have at least 1 dimension")
    if m <= 0:
        raise ValueError("m must be positive")

    deltas = [(b - a) / m for (a, b) in bounds]
    cell_volume = 1.0
    for d in deltas:
        cell_volume *= d

    x = [0.0] * n

    def sample_coord(a: float, dx: float, i: int) -> float:
        # i is the cell index (0..m-1) in that dimension
        if method == "left":
            return a + i * dx
        elif method == "right":
            return a + (i + 1) * dx
        elif method == "midpoint":
            return a + (i + 0.5) * dx
        else:
            raise ValueError("method must be 'left', 'right', or 'midpoint'")

    def recurse(dim: int) -> float:
        if dim == n:
            return f(x)
        a, _b = bounds[dim]
        dx = deltas[dim]
        s = 0.0
        for i in range(m):
            x[dim] = sample_coord(a, dx, i)
            s += recurse(dim + 1)
        return s

    return recurse(0) * cell_volume


def monte_carlo_nd(
    f: Callable[[Vector], float],
    bounds: Bounds,
    N: int,
    seed: int | None = 0
) -> float:
    """
    n-dimensional Monte Carlo integration (uniform sampling over hyper-rectangle).
    """
    if N <= 0:
        raise ValueError("N must be positive")
    if seed is not None:
        random.seed(seed)

    n = len(bounds)
    volume = 1.0
    for (a, b) in bounds:
        if b <= a:
            raise ValueError("Each bound must satisfy b > a")
        volume *= (b - a)

    total = 0.0
    x = [0.0] * n
    for _ in range(N):
        for i, (a, b) in enumerate(bounds):
            x[i] = a + (b - a) * random.random()
        total += f(x)

    return volume * (total / N)


# ------------------- Demo / Tests -------------------

def test():
    # Test 1: f(x)=1 on [0,1]^n -> integral = 1
    for n in [1, 2, 3, 5]:
        bounds = [(0.0, 1.0)] * n
        f = lambda x: 1.0
        approx_r = riemann_nd_recursive(f, bounds, m=20, method="midpoint")
        approx_mc = monte_carlo_nd(f, bounds, N=20000, seed=0)
        print(f"[n={n}] integral=1")
        print(f"  Riemann(midpoint) ~ {approx_r:.6f}")
        print(f"  MonteCarlo        ~ {approx_mc:.6f}")

    # Test 2: f(x)=prod(x_i) on [0,1]^n -> integral = (1/2)^n
    for n in [1, 2, 3, 4]:
        bounds = [(0.0, 1.0)] * n
        def f(x):
            p = 1.0
            for v in x:
                p *= v
            return p
        true_val = (0.5) ** n
        approx_r = riemann_nd_recursive(f, bounds, m=30, method="midpoint")
        approx_mc = monte_carlo_nd(f, bounds, N=80000, seed=1)
        print(f"\n[n={n}] integral=prod(x_i) on [0,1]^n, true={(true_val):.6f}")
        print(f"  Riemann(midpoint) ~ {approx_r:.6f}")
        print(f"  MonteCarlo        ~ {approx_mc:.6f}")

    # Test 3: Gaussian-like f(x)=exp(-||x||^2) on [-1,1]^n (no simple closed form for general n)
    n = 3
    bounds = [(-1.0, 1.0)] * n
    def f(x):
        s2 = sum(v*v for v in x)
        return math.exp(-s2)
    approx_r = riemann_nd_recursive(f, bounds, m=25, method="midpoint")
    approx_mc = monte_carlo_nd(f, bounds, N=200000, seed=2)
    print(f"\n[n={n}] integral=exp(-||x||^2) on [-1,1]^n")
    print(f"  Riemann(midpoint) ~ {approx_r:.6f}")
    print(f"  MonteCarlo        ~ {approx_mc:.6f}")


if __name__ == "__main__":
    test()
