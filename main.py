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
print(vs)
V = np.diag(vs)

#  INFO: Laplacian

Lap = (
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

inds = np.argsort(E)
E = E[inds]
Phi = Phi[:, inds]

for i in range(10):
    plt.plot(xs, asAbs(Phi[:, i]), label=f"{i:3d}: {E[i]}")
plt.legend()
plt.show()
