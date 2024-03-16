# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import fsolve

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
corps = cuivre

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

    # Estimation initiale de la solution - plus petit flottant positif
    a0 = np.finfo(float).eps

    while len(solutions) < n:
        tableau_solutions = fsolve(f, a0)
        print(f'Estimation initiale : {a0}')
        print(f'Tableau de solutions trouvé : {tableau_solutions}')
        solution = fsolve(f, a0)[0]


        # Si la solution est déjà dans la liste, l'ignorer
        if solution in solutions:
            print("Solution déjà trouvée !")
            continue

        # Ajout de la solution à la liste
        solutions.append(solution)

        # Mise à jour de l'estimation initiale pour la prochaine itération
        a0 += np.pi / L_x


    return solutions

print(n_premieres_solutions(n))

