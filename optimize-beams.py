# I'm using a square beam.
# And aluminum 6061 - O -> I also have to evaluate the different tempers for cost

from gekko import GEKKO 
from numpy import pi 

m = GEKKO(remote=False)

# Constraints
Fx = 3000 # Transverse load (N)
Fy = 3000 # Axial load (N)
L = 0.3 # Beam length (m)

Rho = 2700 # Material density (kg/m^3)
S_y = 276e6 # Yield stress 
E = 68.9e9 # Young's modulus

SF_Buckling = 100
SF_Bending = 20


# Variables
B = m.Var(lb=250e-3)  # Beam width (m)
H = m.Var(lb=250e-3) # Beam height (m) 

b = m.Var(lb=200e-3)  # Beam width (m)
h = m.Var(lb=200e-3) # Beam height (m) 


weight = m.Var(lb=0.1) # Beam weight (kg) 


# (Compressive) Stresses due to forces
stress_x = m.Var(ub=S_y) 
stress_y = m.Var(ub=S_y)


# Bending Moment 
Mx = Fx * L
My = Fy * L


# Section modulus 
S = m.Intermediate(B * H ** 2 / 6 - b * h ** 3 / (6 * H))

# Inertia calculations
I_y = m.Intermediate(( B * H ** 3 - b * h ** 3) / 12)
I_x = m.Intermediate(( H * B ** 3 - h * b ** 3) / 12)

# Axial buckling load of proposed cross section  (Not considered)
K = 0.5 # (fixed fixed) linkage, could also be pinned pinned (1) or fixed free (2)
P_cr_x = m.Intermediate((pi ** 2) * E * I_x / ((K * L) ** 2))
P_cr_y = m.Intermediate((pi ** 2) * E * I_y / ((K * L) ** 2))


# Displacement 
delta_x = m.Intermediate((Fx * L**3) / (3 * E * I_y))
delta_y = m.Intermediate((Fy * L**3) / (3 * E * I_x))

m.Equations([
    weight == (B*H - b * h)* L * Rho,
    stress_y == My/S,
    stress_x == Mx/S,
    B-b >= 0.5e-3,
    H-h >= 0.5e-3,
    SF_Buckling * Fy <= P_cr_x,
    SF_Buckling * Fx <= P_cr_y,
    SF_Bending * stress_x <= S_y,  
    SF_Bending * stress_y <= S_y,  
    delta_x <= 0.1e-3,
    delta_y <= 0.1e-3
])


# Objective 
m.Minimize(weight)

m.options.SOLVER=3
m.solve()
print("Weight: ", weight[0])
print("Outer width (B): ", B[0] * 1e3, "mm")
print("Outer height (H): ", H[0] * 1e3, "mm")
print("Inner width (b): ", b[0] * 1e3, "mm")
print("Inner height (h): ", h[0] * 1e3, "mm")

print("Safety factor (bending):", S_y / max(stress_x[0], stress_y[0]))
print("Safety factor (buckling x):", P_cr_x[0] / Fy)
print("Safety factor (buckling y):", P_cr_y[0] / Fy)



