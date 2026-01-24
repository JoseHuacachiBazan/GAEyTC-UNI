import sympy as sp
import re
import os

# ============================================================
# 1. Conversión segura LaTeX → SymPy
# ============================================================
def latex_to_sympy(expr):
    """Convierte expresiones LaTeX simples a formato entendible por SymPy."""

    expr = expr.replace(r"\left", "")
    expr = expr.replace(r"\right", "")
    expr = expr.replace(r"\,", "")
    expr = expr.replace(r"\!", "")

    # Convertir \frac{A}{B} a (A)/(B)
    expr = re.sub(r"\\frac\s*\{(.+?)\}\{(.+?)\}", r"(\1)/(\2)", expr)

    # ^{n} → **n
    expr = re.sub(r"\^\{([^}]+)\}", r"**(\1)", expr)

    # ^n → **n
    expr = re.sub(r"\^([0-9])", r"**\1", expr)

    # multiplicaciones implícitas → insertar *
    expr = re.sub(r"([a-zA-Z0-9])\s+([a-zA-Z])", r"\1*\2", expr)

    # reemplazar \cdot
    expr = expr.replace(r"\cdot", "*")

    # paréntesis LaTeX
    expr = expr.replace(r"\(", "(").replace(r"\)", ")")

    return expr


# ============================================================
# 2. Cargar matriz desde archivo .tex
# ============================================================
def load_matrix_from_tex(filename):
    with open(filename, "r") as f:
        tex = f.read()

    tex = tex.replace("\\left[", "").replace("\\right]", "")

    match = re.search(r"\\begin\{matrix\}(.+?)\\end\{matrix\}", tex, re.DOTALL)
    if not match:
        raise ValueError(f"No se encontró matriz en: {filename}")

    body = match.group(1).strip()
    rows_tex = body.split(r"\\")
    rows = []

    for row in rows_tex:
        raw_cols = row.split("&")
        cols = []
        for col in raw_cols:
            cleaned = latex_to_sympy(col.strip())
            cols.append(sp.sympify(cleaned))
        rows.append(cols)

    return sp.Matrix(rows)


# ============================================================
# 3. Autovalores y guardado
# ============================================================
def compute_and_save_eigenvalues(matrix, output_filename):
    eigs = matrix.eigenvals()

    with open(output_filename, "w") as f:
        f.write("Autovalores:\n")
        for val, mult in eigs.items():
            f.write(f"{sp.simplify(val)}   multiplicidad = {mult}\n")

    print(f"Autovalores guardados en: {output_filename}")


# ============================================================
# 4. PROCESAR TODAS LAS MATRICES
# ============================================================
matrices_tex = {
    "M_double.tex":  "eigs_double.txt",
    "M_simple.tex":  "eigs_simple.txt",
    "M_neutral.tex": "eigs_neutral.txt"
}

for tex_file, out_file in matrices_tex.items():

    if not os.path.exists(tex_file):
        print(f"⚠ No existe {tex_file}")
        continue

    print(f"\n=== Procesando {tex_file} ===")

    M = load_matrix_from_tex(tex_file)
    print("Matriz cargada:")
    sp.pprint(M)

    compute_and_save_eigenvalues(M, out_file)

print("\n✔ Finalizado sin errores.")
