import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
from matplotlib.animation import FuncAnimation

# Define parameters
# Lx, Ly = 50, 50  # dimensions
Lx, Ly = 1.0, 1.0  # dimensions (meters)
T0, T1, Ta = 273.15, 373.15, 293.15  # temperatures in Kelvin (initial, boundary, and ambient)
lambd = 401.0  # thermal conductivity of copper in W/(m·K)
rho = 8960.0  # density of copper in kg/m³
cp = 385.0  # thermal capacity of copper in J/(kg·K)
a = lambd / (rho * cp)  # thermal diffusivity of copper in m²/s
'''# Would actually be 1.11e-4, which is could be too small to make for meaningful animation...
# Diffusion is a very slow process ! Let's take a fictitious value instead.
a = 1.0
(Then again, we could imagine dt to be in hours ?)'''
h = 10.0  # heat transfer coefficient for air in W/(m²·K)
Nx, Ny = 101, 101  # number of grid points
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)  # grid spacings
# dt = 0.01 # time step ?
dt = 1  # time step
'''ATTENTION A VERIFIER MACHIN INCONDITIONNELLEMENT STABLE'''
animate = True  # set to True to animate the simulation, False to compute the end state
end_time = 500.0  # end time of the simulation

Fx = a * dt / (2 * dx**2)
Fy = a * dt / (2 * dy**2)

# Create grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
T = np.full((Nx, Ny), T0)

# Create a figure for the plot
fig, ax = plt.subplots()

# Initial plot
im = ax.imshow(T-273.15, origin='lower', extent=[0, Lx, 0, Ly], cmap='jet', vmin=min(T0, T1, Ta)-273.15, vmax=max(T0, T1, Ta)-273.15)
cbar = fig.colorbar(im, cmap='jet')
cbar.set_label('Temperature (°C)')

# Add a text box with the values of Ta and T1
text_box = ax.text(0.02, 0.95, 'Ta = {:.2f}\nT1 = {:.2f}'.format(Ta-273.15, T1-273.15), transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# Create an array of time values
times = np.arange(0, end_time, dt)

# Set the boundary condition at y=0
T[0, :] = T1

# Update function for the animation
def update(t):
    # First half step
    T_half = np.copy(T)
    for j in range(1, Ny):  # start from 1 to exclude the first row
        A_x = np.zeros((3, Nx - 1))
        A_x[0, 1:] = -Fx / 2        
        A_x[1, :] = 1 + Fx
        A_x[2, :-1] = -Fx / 2

        b_x = Fy / 2 * np.roll(T, 1, axis=0)[:-1, j] + (1 - Fy) * T[:-1, j] + Fy / 2 * np.roll(T, -1, axis=0)[1:, j]
        b_x[0] += T1  # add T1 term
        b_x[-1] -= h * dx / lambd * Ta  # subtract h * dx / lambd * Ta term

        T_half[:-1, j] = solve_banded((1, 1), A_x, b_x)

    # Second half step
    for i in range(1, Nx):  # start from 1 to exclude the first column
        A_y = np.zeros((3, Ny - 1))
        A_y[0, 1:] = -Fy / 2
        A_y[1, :] = 1 + Fy
        A_y[2, :-1] = -Fy / 2

        b_y = Fx / 2 * np.roll(T_half, 1, axis=1)[i, :-1] + (1 - Fx) * T_half[i, :-1] + Fx / 2 * np.roll(T_half, -1, axis=1)[i, 1:]
        b_y[0] -= T1  # subtract T1 term
        b_y[-1] += h * dx / lambd * Ta  # add h * dx / lambd * Ta term

        T[i, :-1] = solve_banded((1, 1), A_y, b_y)

    # Update the image
    im.set_array(T-273.15)

    # Update the title with current time
    plt.title(f"Temperature at t = {t:.3f} unit time")

if animate:
    # Create animation
    ani = FuncAnimation(fig, update, frames=times, interval=1, repeat=False)
    ani.save('heat_equation_solution.gif')
else:
    # Compute end state
    for t in np.arange(0, end_time, dt):
        update(t)

plt.show()