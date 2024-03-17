# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

L_x, L_y = 10, 5 # Longueur et largeur de notre domaine d'étude
N_x, N_y = 100, 50 # Nombre de mailles selon x et selon y

dx = L_x/N_x # Taille d'une maille selon x
dy = L_y/N_y # Taille d'une maille selon y

dt = 1 # Choix du pas de temps de notre simulation
N_t = 1000 # Temps maximal de notre simulation

T_0 = 300  # Température initiale du corps étudié, en Kelvin (environ 27°C)
# Il est conseillé de garder 300 K : la densité et la capacité thermique d'un corps 
# sont thermosensibles, et plus loin on les prendra à 300 K.

T_1 = 323.15  # Température imposée au bord y=0, en Kelvin (50°C)
T_a = 273.15  # Température ambiante du fluide en x=Lx, en Kelvin (0°C)

class CorpsPhysique:
    def __init__(self, name, lambd, rho, cp):
        self.name = name  # Nom du corps physique
        self.lambd = lambd  # Conductivité thermique en W/(m·K)
        self.rho = rho  # Densité en kg/m³
        self.cp = cp  # Capacité thermique en J/(kg·K)
    
    def alpha(self):
        return self.lambd / (self.rho * self.cp)

air = CorpsPhysique("air", 0.0263, 1.1614, 1007)  # Air à 300 K
cuivre = CorpsPhysique("cuivre", 399, 8954, 383)  # Cuivre à 300 K

# Pour rajouter de nouveaux corps physiques, c'est ici.


# Si on veut changer le coefficient de convection extérieure, c'est ici.
h = 10 # en W/(m²·K)

# Si on veut changer le corps étudié, c'est ici.
corps = air

lambd, alpha = corps.lambd, corps.alpha()

# Calcul du temps caractéristique de conduction selon x
t_cond_x = L_x**2 / alpha
print(f"Temps caractéristique de conduction selon x: {t_cond_x:.2f}")

# Calcul du temps caractéristique de conduction selon y
t_cond_y = L_y**2 / alpha
print(f"Temps caractéristique de conduction selon y: {t_cond_y:.2f}")

print()

# Calcul du temps caractéristique de convection selon x
t_conv_x = t_cond_x / h
print(f"Temps caractéristique de convection selon x: {t_conv_x:.2f}")

# Calcul du temps caractéristique de convection selon y
t_conv_y = t_cond_y / h
print(f"Temps caractéristique de convection selon y: {t_conv_y:.2f}")

print()

# Calcul du nombre de Biot selon x
Bi_x = h * L_x / lambd
print(f"Nombre de Biot selon x: {Bi_x:.2f}")

# Calcul du nombre de Biot selon y
Bi_y = h * L_y / lambd
print(f"Nombre de Biot selon y: {Bi_y:.2f}")

print()

# Calcul du nombre de Fourier selon x
Fo_x = alpha * dt / dx**2
print(f"Nombre de Fourier selon x: {Fo_x:.5f}")

# Calcul du nombre de Fourier selon y
Fo_y = alpha * dt / dy**2
print(f"Nombre de Fourier selon y: {Fo_y:.5f}")


beta = h / lambd

def f(a):
    # Fonction dont on cherche les zéros - Ici la variable est nommée a
    # et non alpha, pour éviter toute confusion avec la diffusivité thermique 
    return a * np.tan(L_x * a) - beta

print("Résolution de l'équation a * tan(L_x * a) - h/lambd = 0 avec :")
print(f'L_x = {L_x}')
print(f'h = {h}')
print(f'lambd = {lambd}')

n = 10
print(f'Recherche des {n} premières solutions.')



def n_premieres_solutions(n):
    # Liste pour stocker les solutions
    solutions = []

    # Estimation initiale de la solution
    # On sait que tan est pi-périodique. 
    # Et on peut montrer que tan(L_x * a) enfin bref jsp c'est logique on va dire

    a0 = np.pi / (4 * L_x)
    k = 0

    while len(solutions) < n:
        tableau_solutions = fsolve(f, a0)
        print(f'Estimation initiale : {a0}')
        print(f'Solution trouvée: {tableau_solutions}')
        solution = fsolve(f, a0)[0]


        # Si la solution est déjà dans la liste, l'ignorer
        if solution in solutions:
            print("Solution déjà trouvée !")
            continue

        # Ajout de la solution à la liste
        solutions.append(solution)

        k += 1
        a0 = k * np.pi / L_x


    return solutions

print(n_premieres_solutions(n))
# TODO : Pour une raison étonnante, avec l'air, (qui a une très faible conductivité), ça pète un câble.

# Si on veut changer la valeur à partir de laquelle on s'arrête de calculer des termes de la série, c'est ici.
epsilon = 1e-6

def serie(x, y, _L_x = L_x, _L_y = L_y, _beta = beta):
    somme = 0
    k = 1
    while True:
        a_k = n_premieres_solutions(k)[-1]  # on récupère la k-ième solution
        terme_k = np.cos(a_k * y) * np.cosh(a_k * (_L_y - x)) / (((a_k**2 + _beta**2) * _L_x + _beta) * np.cos(a_k * _L_x) * np.cosh(a_k * _L_y))
        if abs(terme_k) < epsilon:
            print(f"Le {k}ème terme a été trouvé inférieur à epsilon, on décide d'arrêter le calcul")
            break
        somme += terme_k
        k += 1
    return somme

def T_analytique(x, y, _h = h, _lambd = lambd, _T_1 = T_1, _T_a = T_a):
    return _T_a + 2 * _h / _lambd * (_T_1 - _T_a) * serie(x, y)

'''# Define the colors for the colormap
colors = [(0, 0, 1), (1, 1, 0), (1, 0, 0)]  # Blue, Yellow, Red
positions = [0, 0.5, 1]  # Corresponding positions'''

# Define the colors for the colormap in HSL space
colors_hsl = [
    (0.5, 0.9, 0.5),   # Cyan (hue=0.5)
    (0.33, 0.9, 0.5),  # Green (hue=0.33)
    (0.17, 0.9, 0.5),  # Yellow (hue=0.17)
    (0.083, 0.9, 0.5), # Orange (hue=0.083)
    (0.0, 0.9, 0.5)    # Red (hue=0.0)
]

# Convert HSL to RGB
colors_rgb = [plt.cm.colors.hsv_to_rgb(color) for color in colors_hsl]

# Create the colormap
cmap = LinearSegmentedColormap.from_list('CyanToRed', colors_rgb)

def plot_temperature():
    x_values = np.linspace(0, L_x, N_x)
    y_values = np.linspace(0, L_y, N_y)
    X, Y = np.meshgrid(x_values, y_values)
    Z = np.vectorize(T_analytique)(X, Y)

    plt.figure()
    plt.imshow(Z, cmap=cmap, origin='lower', extent=[0, L_x, 0, L_y])
    plt.colorbar(label='T_analytique')
    plt.title('Profil de température analytique')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

plot_temperature()