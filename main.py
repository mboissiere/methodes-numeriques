import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from matplotlib.animation import FuncAnimation

# Define parameters
Lx, Ly = 1.0, 1.0  # dimensions in meters
T0, T1, Ta = 293.15, 373.15, 293.15  # temperatures in Kelvin (initial, boundary, and ambient)
a = 1.11e-4  # thermal diffusivity of copper in m²/s
lambd = 401.0  # thermal conductivity of copper in W/(m·K)
h = 10.0  # heat transfer coefficient for air in W/(m²·K)
Nx, Ny = 25, 25  # number of grid points
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)  # grid spacings
dt = 0.01  # time step
beta = h / lambd  # non-dimensional parameter

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

    # Main loop
    for t in np.arange(0, 1, dt):
        # Solve in x-direction
        for j in range(Ny):
            # Set up tridiagonal matrix
            A_x = np.zeros((3, Nx))
            A_x[0, 1:] = -a * dt / (2 * dx**2)  # Upper diagonal
            A_x[1, :] = 1 + a * dt / (dx**2)  # Main diagonal
            A_x[2, :-1] = -a * dt / (2 * dx**2)  # Lower diagonal

            # Set up right-hand side
            b_x = T[:, j].copy()

            # Solve system
            T[:, j] = solve_banded((1, 1), A_x, b_x)

        # Solve in y-direction
        for i in range(Nx):
            # Set up tridiagonal matrix
            A_y = np.zeros((3, Ny))
            A_y[0, 1:] = -a * dt / (2 * dy**2)  # Upper diagonal
            A_y[1, :] = 1 + a * dt / (dy**2)  # Main diagonal
            A_y[2, :-1] = -a * dt / (2 * dy**2)  # Lower diagonal

            # Set up right-hand side
            b_y = T[i, :].copy()

            # Solve system
            T[i, :] = solve_banded((1, 1), A_y, b_y)

    # Update the image
    im.set_array(T)


# Create animation
ani = FuncAnimation(fig, update, frames=np.arange(0, 1, dt), interval=41.67)

plt.show()