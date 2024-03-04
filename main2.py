import numpy as np
import sys

# Fonction pour résoudre un système tridiagonal à l'aide de l'algorithme TDMA
def tdma_solver(a, b, c, d):
    n = len(d)
    x = np.zeros(n)

    # Forward sweep
    # c_[0] = c[0] / b[0]
    # d_[0] = d[0] / b[0]
    for i in range(1, 3):
        temp = b[i] - a[i] * c[i - 1]
        if temp == 0:
            # Gestion du cas où le dénominateur est nul
            c[i] = c[i - 1]
            d[i] = d[i - 1]
        else:
            c[i] = c[i] / temp
            d[i] = (d[i] - a[i] * d[i - 1]) / temp
    # Backward substitution
    x[-1] = d[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d[i] - c[i] * x[i + 1]

    sys.exit()
    return x

# Paramètres physiques et conditions initiales
Lx = 1.0  # Longueur de la plaque en direction x
Ly = 1.0  # Longueur de la plaque en direction y
Nx = 50   # Nombre de points de discrétisation en direction x
Ny = 50   # Nombre de points de discrétisation en direction y
dx = Lx / (Nx - 1)  # Pas de discrétisation en direction x
dy = Ly / (Ny - 1)  # Pas de discrétisation en direction y
alpha = 0.01  # Coefficient de diffusion thermique
dt = 1    # Pas de temps
T1 = 373.15  # Température en y = 0
Ta = 293.15  # Température du fluide
h = 0.1      # Coefficient de transfert de chaleur

# Conditions initiales et conditions aux limites
T0 = np.zeros((Nx, Ny))  # Profil initial de température
T0[:, 0] = T1             # Condition de Dirichlet en bas

# Nombre total d'itérations dans chaque direction (splitting)
iterations = 10

# Fonction pour une itération dans la direction x
def step_x(T):
    T_new = np.zeros_like(T)
    for j in range(1, Ny - 1):
        a = np.zeros(Nx)
        b = np.zeros(Nx)
        c = np.zeros(Nx)
        d = np.zeros(Nx)

        for i in range(1, Nx - 1):
            a[i] = -alpha * dt / (2 * dx**2)
            b[i] = 1 + alpha * dt / dx**2
            c[i] = -alpha * dt / (2 * dx**2)
            d[i] = T[i, j] + alpha * dt / (2 * dx**2) * (T[i-1, j] - 2*T[i, j] + T[i+1, j])

        # Conditions aux limites
        d[0] += alpha * dt / dx**2 * T[0, j]  # Neumann condition à gauche
        d[-1] += alpha * dt / dx**2 * T[-1, j]  # Neumann condition à droite

        # Échange conducto-convectif en x = Lx
        d[-1] += h * dt / dx * (Ta - T[-1, j])
        print(a.shape)
        print(b.shape)
        print(c.shape)
        print(d.shape)
        T_new[:, j] = tdma_solver(a, b, c, d)

    return T_new

# Fonction pour une itération dans la direction y
def step_y(T):
    T_new = np.zeros_like(T)
    for i in range(1, Nx - 1):
        a = np.zeros(Ny)
        b = np.zeros(Ny)
        c = np.zeros(Ny)
        d = np.zeros(Ny)

        for j in range(1, Ny - 1):
            a[j] = -alpha * dt / (2 * dy**2)
            b[j] = 1 + alpha * dt / dy**2
            c[j] = -alpha * dt / (2 * dy**2)
            d[j] = T[i, j] + alpha * dt / (2 * dy**2) * (T[i, j-1] - 2*T[i, j] + T[i, j+1])

        # Conditions aux limites
        # Neumann condition en bas (déjà incluse dans les conditions initiales)
        d[-1] += alpha * dt / dy**2 * T[i, -1]  # Neumann condition en haut

        T_new[i, :] = tdma_solver(a, b, c, d)

    return T_new

# Boucle pour itérer alternativement entre les directions x et y
T = T0.copy()
for _ in range(iterations):
    T = step_x(T)
    T = step_y(T)

# Affichage de la température finale
print(T)
