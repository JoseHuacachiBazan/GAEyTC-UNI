# derive_mass_matrices_331LR_full.py
# Script completo y autocontenido (SymPy) para derivar las matrices de masa
# de los bosones gauge (±±, ±, 0) a partir del término cinético de los escalares
# Incluye parche para SymPy: trata conj(φ) como variables independientes.
#
# Requisitos:
#   pip install sympy numpy
#
# Guarda como derive_mass_matrices_331LR_full.py y ejecútalo con python3.

import sympy as sp
import numpy as np

sp.init_printing(use_unicode=True)
I = sp.I
sqrt = sp.sqrt

# -----------------------
# 1) Símbolos y parámetros
# -----------------------
# acoplamientos
g, gX, t = sp.symbols('g gX t', real=True)

# cargas U(1)_X (si no conoces, mantenlas simbólicas)
X_T, X_P, X_SL, X_SR = 0, 1, 0, 0
# VEVs
vd, vu, vj = sp.symbols('vd vu vj', real=True)                  # T
wu, wd = sp.symbols('wu wd', real=True)                         # P
vL1, vL2, vR1, vR2 = sp.symbols('vL1 vL2 vR1 vR2', real=True)   # S_L, S_R

# -----------------------
# 2) VEV matrices (según lo proporcionado)
# -----------------------
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

# -----------------------
# 3) Campos gauge (componentes reales)
# -----------------------
Wl1, Wl2, Wl3, Wl4, Wl5, Wl6, Wl7, Wl8 = sp.symbols('Wl1 Wl2 Wl3 Wl4 Wl5 Wl6 Wl7 Wl8', real=True)
Wr1, Wr2, Wr3, Wr4, Wr5, Wr6, Wr7, Wr8 = sp.symbols('Wr1 Wr2 Wr3 Wr4 Wr5 Wr6 Wr7 Wr8', real=True)
B = sp.symbols('B', real=True)

# -----------------------
# 4) M_L and M_R (fundamental representation)
# -----------------------
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

# -----------------------
# 5) a_mu phi (evaluado en VEVs)
# -----------------------
aT = -I * g * (M_L * T_vev - T_vev * M_R)
aP = -I * g * (M_L * P_vev + P_vev * M_R.T) - I * gX * X_P * B * P_vev
aSL = -I * g * (M_L * SL_vev + SL_vev * M_L.T)
aSR = -I * g * (M_R * SR_vev + SR_vev * M_R.T)

# -----------------------
# 6) L_mass = sum_{entries} conj(a_ij) * a_ij  (Tr[a^† a])
# -----------------------
def sum_entry_sq(M):
    s = 0
    for i in range(M.rows):
        for j in range(M.cols):
            s += sp.simplify(sp.conjugate(M[i,j]) * M[i,j])
    return sp.simplify(s)

L_T = sum_entry_sq(aT)
L_P = sum_entry_sq(aP)
L_SL = sum_entry_sq(aSL)
L_SR = sum_entry_sq(aSR)

L_mass_total = sp.simplify(L_T + L_P + L_SL + L_SR)   # forma completa simbólica

# -----------------------
# 7) Reescribir componentes reales en campos cargados (complejos)
#    Definiciones (inversas usadas para sustituir Wl*,Wr* en función de campos complejos):
#      W1 = (W+ + W-)/sqrt2
#      W2 = I*(W+ - W-)/sqrt2
#      W4 = (V+ + V-)/sqrt2
#      W5 = I*(V+ - V-)/sqrt2
#      W6 = (Upp + Umm)/sqrt2
#      W7 = -I*(Upp - Umm)/sqrt2
# -----------------------
# Campos complejos (y sus "nombres" conjugados distintos más abajo)
WplusL, WminusL, VplusL, VminusL = sp.symbols('WplusL WminusL VplusL VminusL', complex=True)
WplusR, WminusR, VplusR, VminusR = sp.symbols('WplusR WminusR VplusR VminusR', complex=True)
UppL, UmmL, UppR, UmmR = sp.symbols('UppL UmmL UppR UmmR', complex=True)

# Mapeo inverso (real -> complejo)
subs_map = {
    # SU(3)_L
    Wl1: (WplusL + WminusL)/sqrt(2),
    Wl2: I*(WplusL - WminusL)/sqrt(2),
    Wl4: (VplusL + VminusL)/sqrt(2),
    Wl5: I*(VplusL - VminusL)/sqrt(2),
    Wl6: (UppL + UmmL)/sqrt(2),
    Wl7: -I*(UppL - UmmL)/sqrt(2),
    # SU(3)_R
    Wr1: (WplusR + WminusR)/sqrt(2),
    Wr2: I*(WplusR - WminusR)/sqrt(2),
    Wr4: (VplusR + VminusR)/sqrt(2),
    Wr5: I*(VplusR - VminusR)/sqrt(2),
    Wr6: (UppR + UmmR)/sqrt(2),
    Wr7: -I*(UppR - UmmR)/sqrt(2)
}

L_mass_charged = sp.simplify(sp.expand(L_mass_total.subs(subs_map)))

# -----------------------
# 8) Parche SymPy: crear símbolos independientes para los conjugados
#    (en lugar de usar sp.conjugate(field) en las derivadas)
# -----------------------
# conjugados como variables independientes
UmmL, UmmR = sp.symbols('UmmL UmmR', complex=True)
WplusL, VminusL, WplusR, VminusR = sp.symbols('WplusL VminusL WplusR VminusR', complex=True)

# Sustituir conj(...) en L_mass_charged por esos símbolos independientes
conj_replacements = {
    sp.conjugate(UppL): UmmL,
    sp.conjugate(UppR): UmmR,
    sp.conjugate(WminusL): WplusL,
    sp.conjugate(VplusL): VminusL,
    sp.conjugate(WminusR): WplusR,
    sp.conjugate(VplusR): VminusR,
    sp.conjugate(WplusL): sp.symbols('WminusL', complex=True),  # si aparecen Wplus conj
    sp.conjugate(WplusR): sp.symbols('WminusR', complex=True),
    sp.conjugate(VminusL): sp.symbols('VplusL', complex=True),
    sp.conjugate(VminusR): sp.symbols('VplusR', complex=True),
    sp.conjugate(UmmL): sp.symbols('UppL', complex=True),
    sp.conjugate(UmmR): sp.symbols('UppR', complex=True)
}

L_mass_charged = L_mass_charged.xreplace(conj_replacements)
L_mass_total = L_mass_total.xreplace(conj_replacements)  # también sustituimos en total por seguridad

# -----------------------
# 9) Definir funciones robustas para extraer matrices por derivadas
# -----------------------
def mass_matrix_complex(L, basis, basis_conj):
    """
    L: expresión simbólica (conjugados reemplazados por variables independientes)
    basis: lista de símbolos complejos [phi1, phi2, ...]
    basis_conj: lista de símbolos que representan conj(phi1), conj(phi2), ...
    Devuelve M tal que L contiene sum basis_conj[i] * M[i,j] * basis[j]
    """
    n = len(basis)
    M = sp.zeros(n)
    Lexp = sp.expand(L)
    for i in range(n):
        for j in range(n):
            M[i, j] = sp.simplify(sp.diff(sp.diff(Lexp, basis_conj[i]), basis[j]))
    return sp.simplify(M)

def mass_matrix_real(L, basis):
    """
    Para campos reales: L debe ser expresión en basis elements (bilineal).
    Extrae M simétrica tal que L = 1/2 * N^T M N (o similar).
    Implementa M[i,j] = d^2 L / d phi_i d phi_j  y luego symmetriza.
    """
    n = len(basis)
    M = sp.zeros(n)
    Lexp = sp.expand(L)
    for i in range(n):
        for j in range(n):
            M[i, j] = sp.simplify(sp.diff(sp.diff(Lexp, basis[i]), basis[j]))
    M = sp.simplify((M + M.T) / 2)
    return sp.simplify(M)

# -----------------------
# 10) Bases (coherentes con sustituciones hechas)
# -----------------------
# Doblemente cargados: Upp (usamos UppL, UppR)  - sus conjugados UppLc, UppRc
basis_double = [UppL, UppR]
basis_double_conj = [UmmL, UmmR]

# Simplemente cargados: elegimos base (W^-_L, V^+_L, W^-_R, V^+_R)
### Correción ####

basis_simple = [WminusL, VplusL, WminusR, VplusR]
basis_simple_conj = [WplusL, VminusL, WplusR, VminusR]
#basis_simple = [WplusL, VplusL, WplusR, VplusR]
#basis_simple_conj = [WminusL, VminusL, WminusR, VminusR]

# Neutros (reales): (B, Wl3, Wl8, Wr3, Wr8)
basis_neutral = [B, Wl3, Wl8, Wr3, Wr8]

# -----------------------
# 11) Construir matrices de masa
# -----------------------
M_double = mass_matrix_complex(L_mass_charged, basis_double, basis_double_conj)
M_simple = mass_matrix_complex(L_mass_charged, basis_simple, basis_simple_conj)
M_neutral = mass_matrix_real(L_mass_total, basis_neutral)

# -----------------------
# 12) Simplificar / factorizar
# -----------------------
M_double_s = sp.simplify(sp.factor(sp.expand(M_double)))
M_simple_s = sp.simplify(sp.factor(sp.expand(M_simple)))
M_neutral_s = sp.simplify(sp.factor(sp.expand(M_neutral)))


# -----------------------
# 14) Mostrar resultados simbólicos (bien presentados)
# -----------------------
print("\n=== MATRIZ DOBLEMENTE CARGADA M_(±±)  (base [U^{++}_L, U^{++}_R]) ===\n")
sp.pprint(M_double_s, use_unicode=True)

print("\n=== MATRIZ SIMPLEMENTE CARGADA M_(±)  (base [W^-_L, V^+_L, W^-_R, V^+_R]) ===\n")
sp.pprint(M_simple_s, use_unicode=True)

print("\n=== MATRIZ NEUTRA M_(0)  (base [B, W^3_L, W^8_L, W^3_R, W^8_R]) ===\n")
sp.pprint(M_neutral_s, use_unicode=True)

export_latex = False
if export_latex:
    with open("M_double.tex", "w") as f:
        f.write(sp.latex(M_double_s))
    with open("M_simple.tex", "w") as f:
        f.write(sp.latex(M_simple_s))
    with open("M_neutral.tex", "w") as f:
        f.write(sp.latex(M_neutral_s))
    print("Matrices exportadas a LaTeX: M_double.tex, M_simple.tex, M_neutral.tex")

# ==================================================
# AUTOVALORES SOLO SIMBÓLICOS
# ==================================================

def print_symbolic_eigenvalues(name, M):
    print("\n===========================================")
    print(f"Autovalores simbólicos de: {name}")
    print("===========================================")

    try:
        eigvals = M.eigenvals()
        for ev, mult in eigvals.items():
            print(f"  multiplicidad {mult}:  {sp.simplify(ev)}")
    except Exception as e:
        print("Error al obtener autovalores simbólicos:")
        print(e)


# Ejecución para las tres matrices
print_symbolic_eigenvalues("M_double", M_double_s)
print_symbolic_eigenvalues("M_simple", M_simple_s)
print_symbolic_eigenvalues("M_neutral", M_neutral_s)

