import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
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


for i in range(1, 21):
    ys = temp_coeffs[:i]
    n = ys.H * ys
    print(f"{i:2d} : {n[0,0]*100:.2f}")

# Shows that the top 10 eigenstates (by coeffs) make up >99% of the wavefunctions
CUTOFF = 11

states = Phi[:, :CUTOFF]
cfs = temp_coeffs[:CUTOFF]


LIMIT = 0.07

SPACE = 0.1

# Create a 3D plot
fig = plt.figure(figsize=(8, 6), dpi=150)
ax = fig.add_subplot(111, projection="3d")
ax.set_xlim(0, 1)  # Set x limits
ax.set_ylim(-LIMIT, LIMIT + SPACE)  # Set y limits for Re(Phi * coeffs)
ax.set_zlim(-LIMIT, LIMIT + SPACE)  # Set z limits for Im(Phi * coeffs)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$\Re(\Psi)$")
ax.set_zlabel(r"$\Im(\Psi)$")
ax.set_yticklabels([])
ax.set_zticklabels([])

# Initialize the lines
lines = [ax.plot([], [], [], color=f"C{i}")[0] for i in range(0, CUTOFF)]

# Create a square subplot for the 2D radial plot
ax2 = fig.add_axes((0.21, 0.6, 0.2, 0.2), polar=True)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2.set_ylim(0, np.max(np.abs(cfs)) * 1.1)

# Initialize radial lines and circles
radial_lines = [ax2.plot([], [], color=f"C{i}")[0] for i in range(CUTOFF)]
radial_circles = [ax2.plot([], [], "o", color=f"C{i}")[0] for i in range(CUTOFF)]


# Update function for animation
def update(_frame):
    global coeffs
    coeffs = M @ coeffs
    for i, line in enumerate(lines[1:]):
        c = np.zeros_like(coeffs)
        c[sort_inds[i]] = coeffs[sort_inds[i]]
        ys = Phi * c  # Calculate complex values for each line
        line.set_data(xs, np.real(ys.A1))  # Set x and y data (Re)
        line.set_3d_properties(np.imag(ys.A1))  # Set z data (Im)
    line = lines[0]
    ys = Phi * coeffs
    line.set_data(xs, np.real(ys.A1) + SPACE)  # Set x and y data (Re)
    line.set_3d_properties(np.imag(ys.A1) + SPACE)  # Set z data (Im)

    # Update the radial plot
    for i in range(CUTOFF):
        angle = np.angle(coeffs[sort_inds[i]][0, 0])
        magnitude = np.abs(coeffs[sort_inds[i]][0, 0])
        radial_lines[i].set_data([0, angle], [0, magnitude])
        radial_circles[i].set_data([angle], [magnitude])

    return lines + radial_lines + radial_circles


# Create the animation
ani = FuncAnimation(fig, update, frames=len(ts), blit=True, interval=100)

# Show the animation
ani.save(
    "states.gif",
    writer=PillowWriter(fps=30),
    savefig_kwargs={"pad_inches": 0},
)
