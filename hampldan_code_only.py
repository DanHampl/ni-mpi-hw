
import numpy as np
from pprint import pprint

class Solver:
    def __init__(self, gamma):
        """
        Initiates the instance for specific gamma value
        """
        # matrix A
        self.a = np.zeros((20, 20), dtype=np.float64)
        np.fill_diagonal(self.a, gamma)
        np.fill_diagonal(self.a[:, 1:], -1)
        np.fill_diagonal(self.a[1:], -1)

        # vector b
        self.b = np.full((20, 1), gamma-2, dtype=np.float64)
        self.b[0], self.b[-1] = gamma - 1, gamma - 1

        # matrix L
        self.l = np.tril(self.a, -1)

        # matrix D
        self.d = np.diag(np.diag(self.a))

        # matrix U
        self.u = self.a - self.l - self.d

        # set jacobi as default
        self.set_jacobi()

        # set default precision
        self.precision = 10**-6

    def set_jacobi(self):
        """
        Sets the Jacobi method
        """
        self.q = self.d
        self.q_inv = np.linalg.inv(self.q)
        self.convergent = None
        self.result = None
        self.iterations = None
        return self

    def set_gs(self, omega = 1):
        """
        Sets the Gauss-Seidel method
        """
        self.q = (self.d / omega) + self.l
        self.q_inv = np.linalg.inv(self.q)
        self.convergent = None
        self.result = None
        self.iterations = None
        return self

    def calculate(self):
        """
        Check if iterative method converges and calculate the result
        """
        if not self.is_convergent():
            self.result = []
            self.iterations = 0
        if self.iterations is None:
            self.result, self.iterations = self._calculate()
        return self.result, self.iterations

    def _calculate(self, xn=None, index=0):
        """
        Calculate with the iterative method
        """
        if xn is None:
            xn = np.zeros((20, 1), np.float64)
        if self.result_is_close_enough((self.a @ xn) - self.b):
            return xn.transpose(), index
        return self._calculate(self.calculate_single(xn), index + 1)

    def calculate_single(self, xn):
        """
        Calculates single iteration
        """
        return (self.q_inv @ (((self.q - self.a) @ xn) + self.b))

    def result_is_close_enough(self, result):
        """
        Calculates if result is precise enough
        """
        return (np.linalg.norm(result) / np.linalg.norm(self.b)) < self.precision

    def is_convergent(self):
        """
        Calculates convergence of iterative method
        """
        if self.convergent is None:
            self.convergent = max(abs(np.linalg.eigvals(np.identity(20, dtype=np.float64) - (self.q_inv @ self.a)))) < 1
        return self.convergent
