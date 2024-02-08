import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from matplotlib.animation import FuncAnimation

# Define parameters
Lx, Ly = 1.0, 1.0  # dimensions in meters
T0, T1, Ta = 293.15, 373.15, 293.15  # temperatures in Kelvin (initial, boundary, and ambient)
lambd = 401.0  # thermal conductivity of copper in W/(m·K)
rho = 8960.0  # density of copper in kg/m³
cp = 385.0  # thermal capacity of copper in J/(kg·K)
a = lambd / (rho * cp)  # thermal diffusivity of copper in m²/s
h = 10.0  # heat transfer coefficient for air in W/(m²·K)
Nx, Ny = 25, 25  # number of grid points
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)  # grid spacings
dt = 0.01  # time step
beta = h / lambd  # non-dimensional parameter

Fx = a * dt / (2 * dx**2)
Fy = a * dt / (2 * dy**2)

# Create grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
T = np.full((Nx, Ny), T0)

# Create a figure for the plot
fig, ax = plt.subplots()

# Initial plot
im = ax.imshow(T, origin='lower', extent=[0, Lx, 0, Ly], cmap='coolwarm')
fig.colorbar(im, cmap='coolwarm')

# Update function for the animation
def update(frame):
    # First half step
    T_half = np.copy(T)
    for j in range(Ny):
        A_x = np.zeros((3, Nx))
        A_x[0, 1:] = -Fx
        A_x[1, :] = 1 + Fx
        A_x[2, :-1] = -Fx

        b_x = Fy * T[:-1, j] + (1 - Fy) * T[:, j] + Fy * T[1:, j]
        b_x[0] += T1
        b_x[-1] -= h * dx / lambd * Ta

        T_half[:, j] = solve_banded((1, 1), A_x, b_x)

    # Second half step
    for i in range(Nx):
        A_y = np.zeros((3, Ny))
        A_y[0, 1:] = -Fy
        A_y[1, :] = 1 + Fy
        A_y[2, :-1] = -Fy

        b_y = Fx * T_half[i, :-1] + (1 - Fx) * T_half[i, :] + Fx * T_half[i, 1:]
        b_y[0] -= T1
        b_y[-1] += h * dx / lambd * Ta

        T[i, :] = solve_banded((1, 1), A_y, b_y)

    # Update the image
    im.set_array(T)

# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 10, dt), interval=41.67)

plt.show()