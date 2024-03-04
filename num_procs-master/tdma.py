import numpy as np

def tdma_solver(matrix, d):
    """
    Solve a tridiagonal linear system Ax = d using TDMA algorithm.

    Parameters:
        matrix: numpy array representing the tridiagonal matrix A.
        d: numpy array representing the right-hand side of the equation.

    Returns:
        x: numpy array representing the solution vector.
    """
    n = len(d)
    a = np.concatenate([[0], np.diag(matrix, k=-1)])
    b = np.diag(matrix)
    c = np.concatenate([np.diag(matrix, k=1), [0]])

    c_ = np.zeros(n)
    d_ = np.zeros(n)
    x = np.zeros(n)

    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]

    # Forward sweep
    for i in range(1, n):
        c_[i] = c[i] / (b[i] - a[i] * c_[i - 1])
        d_[i] = (d[i] - a[i] * d_[i - 1]) / (b[i] - a[i] * c_[i - 1])

    # Backward substitution
    x[-1] = d_[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_[i] - c_[i] * x[i + 1]

    return x

# Example usage:
A = np.array([[2, 1],
              [1, 2]])  # Tridiagonal matrix
d = np.array([3, 4])  # Right-hand side

solution = tdma_solver(A, d)
print("Solution:", solution)

