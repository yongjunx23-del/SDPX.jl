#!/usr/bin/env python3
"""Independent standard-library cross-check of SDPX's five HSD Newton equations.

Unknown order: [dx (n), dy (m), ds (m), dtau, dkappa].
This is documentation/oracle code, not a CI dependency or a solver backend.
"""

from math import fsum


def dot(a, b):
    return fsum(x * y for x, y in zip(a, b))


def matvec(a, x):
    return [dot(row, x) for row in a]


def transpose(a):
    return [list(column) for column in zip(*a)]


def solve_pivoted(a, b):
    """Small dense Gaussian solve with partial pivoting for this fixture."""
    a = [row[:] + [rhs] for row, rhs in zip(a, b)]
    size = len(a)
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(a[row][column]))
        if abs(a[pivot][column]) < 1e-14:
            raise ArithmeticError("oracle fixture is singular")
        a[column], a[pivot] = a[pivot], a[column]
        scale = a[column][column]
        a[column] = [value / scale for value in a[column]]
        for row in range(size):
            if row == column:
                continue
            multiplier = a[row][column]
            a[row] = [
                value - multiplier * pivot_value
                for value, pivot_value in zip(a[row], a[column])
            ]
    return [a[row][-1] for row in range(size)]


def assemble_jacobian(a, b, c, h, tau, kappa):
    m, n = len(a), len(c)
    size = n + 2 * m + 2
    ix = range(n)
    iy = range(n, n + m)
    ids = range(n + m, n + 2 * m)
    itau, ikappa = size - 2, size - 1
    j = [[0.0] * size for _ in range(size)]

    # A*dx + ds - b*dtau
    for row in range(m):
        for column in ix:
            j[row][column] = a[row][column]
        j[row][ids[row]] = 1.0
        j[row][itau] = -b[row]

    # A'*dy + c*dtau
    for column in range(n):
        row = m + column
        for cone_row in range(m):
            j[row][iy[cone_row]] = a[cone_row][column]
        j[row][itau] = c[column]

    # -c'*dx - b'*dy + dkappa
    gap_row = m + n
    for column in range(n):
        j[gap_row][ix[column]] = -c[column]
    for cone_row in range(m):
        j[gap_row][iy[cone_row]] = -b[cone_row]
    j[gap_row][ikappa] = 1.0

    # ds + H*dy
    for cone_row in range(m):
        row = m + n + 1 + cone_row
        for cone_column in range(m):
            j[row][iy[cone_column]] = h[cone_row][cone_column]
        j[row][ids[cone_row]] = 1.0

    # kappa*dtau + tau*dkappa
    j[-1][itau], j[-1][ikappa] = kappa, tau
    return j


def main():
    a = [[1.0, 0.5], [-0.25, 1.5], [0.75, -1.0]]
    b = [1.0, -0.5, 0.25]
    c = [-0.75, 1.25]
    h = [[2.0, 0.2, 0.0], [0.2, 1.4, 0.1], [0.0, 0.1, 1.1]]
    tau, kappa = 1.1, 0.9
    expected = [0.2, -0.3, 0.1, -0.2, 0.15, -0.4, 0.3, 0.2, -0.1, 0.25]

    jacobian = assemble_jacobian(a, b, c, h, tau, kappa)
    rhs = matvec(jacobian, expected)
    recovered = solve_pivoted(jacobian, rhs)
    equation_residual = [
        value - target for value, target in zip(matvec(jacobian, recovered), rhs)
    ]
    direction_error = [
        value - target for value, target in zip(recovered, expected)
    ]
    print(f"max five-equation residual: {max(map(abs, equation_residual)):.3e}")
    print(f"max direction error:        {max(map(abs, direction_error)):.3e}")


if __name__ == "__main__":
    main()
