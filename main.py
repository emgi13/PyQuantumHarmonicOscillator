import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from scipy import linalg as la


#  INFO: PARAMETERS

size = 1_000 + 1
end_k = 1e8
k = 1e6
mass = 1
hbar = 1

#  INFO: BASIS

xs, dx = np.linspace(0, 1, size, retstep=True)

#  INFO: Potential

vs = 0.5 * k * np.power((xs - 0.5), 2)
# vs = np.zeros((size,))
vs[+0] = end_k
vs[-1] = end_k
V = np.matrix(np.diag(vs))

#  INFO: Laplacian

Lap = np.matrix(
    np.diag(np.ones((size - 1,)) * 1.0, k=+1)
    + np.diag(np.ones((size - 1,)) * 1.0, k=-1)
    - np.diag(np.ones((size,)) * 2.0, k=0)
) / (dx * dx)


def asAbs(x: NDArray):
    return np.sqrt(x.T * x)


#  INFO: Hamiltonian

H = V - Lap * -1.0 * hbar * hbar / 2.0 / mass

#  INFO: Eigenstates

E, Phi = la.eigh(H)

sort_inds = np.argsort(E)
E = E[sort_inds]
Phi = np.matrix(Phi[:, sort_inds])


def gauss(x: NDArray, x0: float, s: float):
    y = np.exp(-0.5 * np.power((x - x0) / s, 2))
    return y / la.norm(y)


Psi = np.matrix(gauss(xs, 0.65, 0.03)).T
coeffs = Phi.T * Psi


sort_inds = np.argsort(np.abs(coeffs.A1))[::-1]
temp_coeffs = coeffs[sort_inds]
for i in range(30):
    print(la.norm(temp_coeffs.A1[:i]))
