# Código completo (SymPy) para DERIVAR automáticamente las MATRICES DE MASA
# de los bosones gauge (doblemente cargados, simplemente cargados y neutros)
# a partir del término cinético de los campos escalares T, P, S_L, S_R y sus VEVs.
#
# Instrucciones:
# - Pega/ejecuta en un entorno Python con sympy instalado.
# - El script devuelve simbólicamente las 3 matrices de masa:
#     M_pp (2x2)   -> base (U^{++}_L, U^{++}_R)
#     M_p  (4x4)   -> base (W^-_L, V^+_L, W^-_R, V^+_R)
#     M_0  (5x5)   -> base (B, W3L, W8L, W3R, W8R)
# - Está basado en las expresiones que proporcionaste y en las definiciones
#   de la derivada covariante y M_{L,R}.
#
# Requisitos: sympy
#   pip install sympy
#
# NOTA: algunas constantes de carga X (g_X * X_phi) aparecen para P y S.
#       Si conoces los valores de X_P, X_SL, X_SR y gX, sustitúyelos en
#       la sección de parámetros numéricos al final.

import sympy as sp
I = sp.I
sqrt = sp.sqrt

# -------------------------
# 1) Símbolos y parámetros
# -------------------------
g, gX = sp.symbols('g gX', real=True)
# cargas U(1)_X de cada escalar (símbolos por si no conoces numéricos)
X_T, X_P, X_SL, X_SR = 0, 1 ,0 ,0
#X_T, X_P, X_SL, X_SR = sp.symbols('X_T X_P X_SL X_SR', real=True)

# VEVs proporcionados por el usuario:
vd, vu, vj = sp.symbols('vd vu vj', real=True)                  # T
wu, wd = sp.symbols('wu wd', real=True)                         # P
vL1, vL2, vR1, vR2 = sp.symbols('vL1 vL2 vR1 vR2', real=True)   # S_L, S_R

# Definición de los VEV matrices (según lo diste)
T_vev = sp.Matrix([[vd, 0, 0],
                   [0, vu, 0],
                   [0, 0, vj]])
P_vev = sp.Matrix([[0, wd, 0],
                   [wu, 0, 0],
                   [0, 0, 0]])
SL_vev = sp.Matrix([[vL1, 0, 0],
                    [0, 0, vL2/sp.sqrt(2)],
                    [0, vL2/sp.sqrt(2), 0]])
SR_vev = sp.Matrix([[vR1, 0, 0],
                    [0, 0, vR2/sp.sqrt(2)],
                    [0, vR2/sp.sqrt(2), 0]])

# -------------------------
# 2) Campos gauge (componentes reales)
# -------------------------
# Campos gauge SU(3)_L y SU(3)_R, índices 1..8
Wl = sp.symbols('Wl1:9', real=True)  # Wl1..Wl8
Wr = sp.symbols('Wr1:9', real=True)  # Wr1..Wr8
# Campo B (U(1)_X)
B = sp.symbols('B', real=True)

# Crear matrices M_L and M_R in fundamental representation:
# M = sum_j W^j * lambda_j  (we use Gell-Mann matrices)
# For practicality, we build directly the 3x3 matrix as per your formula:
Wl1,Wl2,Wl3,Wl4,Wl5,Wl6,Wl7,Wl8 = Wl
Wr1,Wr2,Wr3,Wr4,Wr5,Wr6,Wr7,Wr8 = Wr

# M_L matrix (symbolic)
M_L = sp.Matrix([
    [ Wl8/sqrt(3) + Wl3,          Wl1 - I*Wl2,            Wl4 - I*Wl5],
    [ Wl1 + I*Wl2,                Wl8/sqrt(3) - Wl3,      Wl6 - I*Wl7],
    [ Wl4 + I*Wl5,                Wl6 + I*Wl7,           -2*Wl8/sqrt(3)]
]) / 2

M_R = sp.Matrix([
    [ Wr8/sqrt(3) + Wr3,          Wr1 - I*Wr2,            Wr4 - I*Wr5],
    [ Wr1 + I*Wr2,                Wr8/sqrt(3) - Wr3,      Wr6 - I*Wr7],
    [ Wr4 + I*Wr5,                Wr6 + I*Wr7,           -2*Wr8/sqrt(3)]
]) / 2

# ---------------------------------------
# 3) Definir los términos a_mu phi según tu nota
#    a_mu T = - i g ( M_L T - T M_R )
#    a_mu P = - i g ( M_L P + P M_R^T ) - i gX X_P B P
#    a_mu S_L = - i g ( M_L S_L + S_L M_L^T )
#    a_mu S_R = - i g ( M_R S_R + S_R M_R^T )
# ---------------------------------------
aT = -I * g * (M_L * T_vev - T_vev * M_R)
aP = -I * g * (M_L * P_vev + P_vev * M_R.T) - I * gX * X_P * B * P_vev
aSL = -I * g * (M_L * SL_vev + SL_vev * M_L.T)
aSR = -I * g * (M_R * SR_vev + SR_vev * M_R.T)

# ---------------------------------------
# 4) Construir el término de masa: sum_phi Tr[(a_phi)^\dagger (a_phi)]
#    Para matrices 3x3 calculamos la suma elemento a elemento:
#    L_mass = sum_{entries} conj(a_phi_ij) * a_phi_ij
#    (trabajamos con símbolos reales para VEVs; conj no cambia)
# ---------------------------------------
def sum_entry_sq(M):
    # suma de |M_ij|^2 (como símbolo)
    s = 0
    for i in range(M.rows):
        for j in range(M.cols):
            s += sp.simplify(sp.conjugate(M[i,j]) * M[i,j])
    return sp.simplify(s)

L_T = sum_entry_sq(aT)
L_P = sum_entry_sq(aP)
L_SL = sum_entry_sq(aSL)
L_SR = sum_entry_sq(aSR)

L_mass_total = sp.simplify(L_T + L_P + L_SL + L_SR)

# ---------------------------------------
# 5) Ahora expresamos L_mass_total como forma bilineal en los campos gauge
#    Queremos extraer matrices de coeficientes para las bases específicas:
#      - doblemente cargados: U^{++}_L,R (provenientes de W6 ± i W7)
#      - simplemente cargados: W^-_L, V^+_L, W^-_R, V^+_R
#      - neutros: B, W3L, W8L, W3R, W8R
#
#  Estrategia:
#   - Definir variables complejas para combinaciones de componentes reales.
#   - Reemplazar en L_mass_total las expresiones W6,W7,etc en términos de las combinaciones.
#   - Extraer coeficientes de los términos bilineales para formar las matrices.
# ---------------------------------------

# Definir combinaciones complejas (según tus definiciones)
# U^{++} = (W6 + i W7)/sqrt(2)
UppL, UmmL, UppR, UmmR = sp.symbols('UppL UmmL UppR UmmR')  # Upp, Umm left and right (complex)
# Solve: W6 = (Upp+Umm)/sqrt(2), W7 = (Upp-Umm)/(I*sqrt(2))
subs_Wl6_7 = {
    Wl6: (UppL + UmmL)/sqrt(2),
    Wl7: (UppL - UmmL)/(I*sqrt(2))
}
subs_Wr6_7 = {
    Wr6: (UppR + UmmR)/sqrt(2),
    Wr7: (UppR - UmmR)/(I*sqrt(2))
}

# Definir simplemente cargados:
# W^+ = (W1 - i W2)/sqrt(2), W^- = (W1 + i W2)/sqrt(2)
# V^+ = (W4 - i W5)/sqrt(2), V^- = (W4 + i W5)/sqrt(2)
WplusL, WminusL, VplusL, VminusL = sp.symbols('WplusL WminusL VplusL VminusL')
WplusR, WminusR, VplusR, VminusR = sp.symbols('WplusR WminusR VplusR VminusR')

subs_Wl1_2 = {
    Wl1: (WplusL + WminusL)/sqrt(2),
    Wl2: (WplusL - WminusL)/( -I*sqrt(2) )   # inverted from W1 = (W+ + W-)/sqrt2 etc.
}
# Better derive invertible mapping:
# Let Wplus = (W1 - i W2)/sqrt2 ; Wminus = (W1 + i W2)/sqrt2
# Solve for W1,W2:
# W1 = (Wplus + Wminus)/sqrt2
# W2 = I*(Wplus - Wminus)/sqrt2
subs_Wl1_2 = { Wl1: (WplusL + WminusL)/sqrt(2),
               Wl2: I*(WplusL - WminusL)/sqrt(2) }

subs_Wl4_5 = { Wl4: (VplusL + VminusL)/sqrt(2),
               Wl5: I*(VplusL - VminusL)/sqrt(2) }

subs_Wr1_2 = { Wr1: (WplusR + WminusR)/sqrt(2),
               Wr2: I*(WplusR - WminusR)/sqrt(2) }
subs_Wr4_5 = { Wr4: (VplusR + VminusR)/sqrt(2),
               Wr5: I*(VplusR - VminusR)/sqrt(2) }

# Construir el reemplazo global para pasar a variables cargadas
subs_global = {}
subs_global.update(subs_Wl6_7)
subs_global.update(subs_Wr6_7)
subs_global.update(subs_Wl1_2)
subs_global.update(subs_Wl4_5)
subs_global.update(subs_Wr1_2)
subs_global.update(subs_Wr4_5)

# Además mantener los neutrales B, W3L, W8L, W3R, W8R tal como están.

# Reemplazar en L_mass_total:
L_mass_charged = sp.simplify( sp.expand(L_mass_total.subs(subs_global)) )

# ---------------------------------------
# 6) Construir matrices extrayendo coeficientes bilineales
#    Función auxiliar: dada una lista de campos (orden de base) y la Lagr. L_mass,
#    construye la matriz M_ij tal que L_mass contains sum_{i,j} phi_i^* M_ij phi_j
#    Nota: trabajamos con campos complejos donde la estructura esperada es
#          suma phi_i^* * M_ij * phi_j  (phi_i^* es conjugado).
# ---------------------------------------
def build_mass_matrix(L, basis, conj_suffix='conj'):
    """
    L: expression (sum of bilinears)
    basis: list of sympy symbols representing complex fields [phi1, phi2, ...]
    returns: Matrix M such that L contains sum conj(phi_i)*M[i,j]*phi_j
    """
    n = len(basis)
    M = sp.zeros(n)
    # For each pair, extract coefficient of conj(phi_i)*phi_j
    for i, ph_i in enumerate(basis):
        ph_i_conj = sp.symbols(f'{str(ph_i)}_c')  # temporary symbol name for conjugate
        # Replace conjugate occurrences sp.conjugate(ph_i) by ph_i_conj to simplify pattern matching
        L_temp = L.xreplace({sp.conjugate(ph_i): ph_i_conj})
        for j, ph_j in enumerate(basis):
            # Coefficient of ph_i_conj * ph_j
            coeff = sp.simplify(sp.expand(L_temp).coeff(ph_i_conj*ph_j, 1))
            M[i,j] = sp.simplify(coeff)
    return sp.simplify(M)

# Define bases:
# Doblemente cargados: basis_Upp = [UppL, UppR]  (usamos Upp and its conjugate)
basis_Upp = [UppL, UppR]
# Simplemente cargados: choose basis order (W^-_L, V^+_L, W^-_R, V^+_R)
# Note: our symbols WminusL, VplusL, WminusR, VplusR represent the complex fields
basis_simple = [WminusL, VplusL, WminusR, VplusR]
# Neutros: basis_neutral = [B, W3L, W8L, W3R, W8R]  (real fields; for bilinear treat as real)
basis_neutral = [B, Wl3, Wl8, Wr3, Wr8] = [B, Wl3, Wl8, Wr3, Wr8]  # use previously defined names

# For charged sectors we need to use conjugates in L_mass_charged
# But our L_mass_charged expression contains conjugates via complex algebra already.
# Replace sp.conjugate(symbol) occurrences for matching:
# Build mass matrices for double and simple charge sectors:
M_Upp = build_mass_matrix(L_mass_charged, basis_Upp)
M_simple = build_mass_matrix(L_mass_charged, basis_simple)

# For neutrals we don't need to remap; just extract coefficients of B, W3L, W8L, W3R, W8R
# But L_mass_total is currently in terms of Wl3, Wl8, Wr3, Wr8 and B; extract quadratic form
neut_basis = [B, Wl3, Wl8, Wr3, Wr8]
# For neutral real fields, the bilinear is simply (1/2) N^T M N in your notation.
# Our L_mass_total is sum coef * field_i * field_j (no conjugates). Extract symmetric matrix.
def build_symmetric_mass_matrix(L, basis):
    n = len(basis)
    M = sp.zeros(n)
    L_exp = sp.expand(L)
    for i in range(n):
        for j in range(n):
            # coefficient of basis[i]*basis[j]
            coeff_ij = sp.simplify(L_exp.coeff(basis[i]*basis[j], 1))
            # Because L contains both i,j and j,i, we average
            M[i,j] = sp.simplify(coeff_ij)
    # Ensure symmetry
    M = sp.simplify((M + M.T)/2)
    return M

M_neutral = build_symmetric_mass_matrix(L_mass_total, neut_basis)

# ---------------------------------------
# 7) Limpiar resultados: factorizar y simplificar
# ---------------------------------------
M_Upp_simpl = sp.simplify(sp.factor(M_Upp))
M_simple_simpl = sp.simplify(sp.factor(M_simple))
M_neutral_simpl = sp.simplify(sp.factor(M_neutral))

# Mostrar resultados simbólicos
sp.pp = sp.pprint
print("\n=== MATRIZ DOBLEMENTE CARGADA M_(±±)  (base [UppL, UppR]) ===\n")
sp.pprint(M_Upp_simpl)

print("\n=== MATRIZ SIMPLEMENTE CARGADA M_(±)  (base [W^-_L, V^+_L, W^-_R, V^+_R]) ===\n")
sp.pprint(M_simple_simpl)

print("\n=== MATRIZ NEUTRA M_(0)  (base [B, W3L, W8L, W3R, W8R]) ===\n")
sp.pprint(M_neutral_simpl)

# ---------------------------------------
# 8) Opcional: sustituir wd=wu, vR2=vR1 y gx/g -> t (o valores numéricos)
#    Si quieres aplicar esas simplificaciones simbólicas ahora:
# ---------------------------------------
subs_simpl = {wd: wu, vR2: vR1}
# sustituir gx = t*g
t = sp.symbols('t', real=True)
subs_simpl[gX] = gX  # keep
# apply subs
M_Upp_final = sp.simplify(M_Upp_simpl.subs(subs_simpl))
M_simple_final = sp.simplify(M_simple_simpl.subs(subs_simpl))
M_neutral_final = sp.simplify(M_neutral_simpl.subs(subs_simpl))

# Factor final
M_Upp_final = sp.factor(M_Upp_final)
M_simple_final = sp.factor(M_simple_final)
M_neutral_final = sp.factor(M_neutral_final)

# Imprimir versión final
print("\n--- Versión final (tras sustituciones wd=wu, vR2=vR1) ---\n")
sp.pprint(M_Upp_final)
sp.pprint(M_simple_final)
sp.pprint(M_neutral_final)

# -----------------------
# 9) Sugerencia:
# Si las matrices salen con términos similares, puedes sustituir valores numéricos
# para inspeccionar numéricamente (por ejemplo g=0.65, gX=0.35, etc.)
# y luego diagonalizarlas con numpy o sympy.eigenvects.
# -----------------------
