import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy import linalg as la

# INFO: PARAMETERS
size = 1_000 + 1
end_k = 1e8
k = 1e5
mass = 1
hbar = 1

# INFO: BASIS
xs, dx = np.linspace(0, 1, size, retstep=True)

# INFO: Potential
vs = 0.5 * k * np.power((xs - 0.5), 2)
vs[0] = end_k
vs[-1] = end_k
V = np.matrix(np.diag(vs))

# INFO: Laplacian
Lap = np.matrix(
    np.diag(np.ones((size - 1,)) * 1.0, k=+1)
    + np.diag(np.ones((size - 1,)) * 1.0, k=-1)
    - np.diag(np.ones((size,)) * 2.0, k=0)
) / (dx * dx)

# INFO: Hamiltonian
H = V - Lap * hbar * hbar / (2.0 * mass)

# INFO: Eigenstates
E, Phi = la.eigh(H)
sort_inds = np.argsort(E)
E = E[sort_inds]
Phi = np.matrix(Phi[:, sort_inds])


def gauss(x: NDArray, x0: float, s: float):
    y = np.exp(-0.5 * np.power((x - x0) / s, 2))
    return y / la.norm(y)


H_new = np.diag(E)
Psi = np.matrix(gauss(xs, 0.65, 0.03)).H
coeffs = Phi.H * Psi

sort_inds = np.argsort(np.abs(coeffs.A1))[::-1]
temp_coeffs = coeffs[sort_inds]

ts, dt = np.linspace(0, 0.1, 600, retstep=True)

M = la.expm(-1j * H_new * dt / hbar)


def evolve(co: np.matrix, t: float):
    return la.expm(-1j * H_new * t / hbar) * co


# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(0, 1)  # Set x limits
ax.set_ylim(-1, 1)  # Set y limits for Re(Phi * coeffs)
ax.set_zlim(-1, 1)  # Set z limits for Im(Phi * coeffs)
ax.set_title("3D Animated Plot")
ax.set_xlabel("x")
ax.set_ylabel("Re(Phi * coeffs)")
ax.set_zlabel("Im(Phi * coeffs)")

# Initialize the line
(line,) = ax.plot([], [], [], color="b")


# Update function for animation
def update(frame):
    global coeffs
    coeffs = M * coeffs  # Update coeffs
    complex_values = Phi * coeffs  # Calculate complex values
    line.set_data(xs, np.real(complex_values.A1))  # Set x and y data (Re)
    line.set_3d_properties(np.imag(complex_values.A1))  # Set z data (Im)
    return (line,)


plt.tight_layout(pad=0.5)
# Create the animation
ani = FuncAnimation(fig, update, frames=len(ts), blit=True, interval=100)

# Show the animation
ani.save(
    "abs.gif",
    writer=PillowWriter(fps=30),
    savefig_kwargs={"pad_inches": 0},
)
print("SAVED")
