# High Energy Physics Models (hep_models)

Este repositorio contiene un paquete de Python desarrollado en el contexto de física teórica y física de partículas elementales. Su objetivo es modelar, calcular y visualizar aspectos clave del **Modelo Estándar** y sus **Extensiones (BSM)**, tales como el sector de Higgs, derivadas covariantes y matrices de masa.

El proyecto está estructurado como un paquete instalable de Python, lo que permite desarrollar los modelos analíticos o numéricos en el repositorio e importarlos de manera global en cualquier script local del sistema.

---

## 📂 Estructura del Repositorio

```text
high-energy-physics/
├── README.md               # Presentación e instrucciones de uso
├── setup.py                # Configuración para la instalación editable del paquete
├── hep_models/             # PAQUETE PRINCIPAL (Módulos de física)
│   ├── __init__.py         # Inicializador del paquete
│   ├── plotting.py         # Gráficas avanzadas (potenciales, espacio de fases, etc.)
│   ├── covariant.py        # Cálculos de derivadas covariantes y tensores de norma
│   ├── higgs.py            # Estudio del sector de Higgs e invariancia de calibre
│   └── masses.py           # Diagonalización de matrices de masa (Yukawa, CKM, PMNS)
└── examples/               # Scripts de ejemplo para pruebas locales
```

---

## 🔧 Requisitos Previos

Antes de instalar, asegúrate de contar con Python 3.8+ y un gestor de paquetes actualizado. Las dependencias principales que utiliza el paquete son:
* `numpy` (Álgebra lineal para matrices de masa)
* `scipy` (Optimización y cálculo numérico complejo)
* `matplotlib` (Visualización y gráficas avanzadas)
* `sympy` (Cálculo simbólico algebraico para tensores y derivadas)

---

## 🚀 Instalación (Modo Desarrollo / Editable)

Para poder clonar este repositorio y utilizar sus módulos en cualquier script de Python de tu computadora sin necesidad de mover archivos, debes realizar una **instalación editable**.

1. **Clona el repositorio** en tu computadora:
   ```bash
   git clone https://github.com
   cd TU_REPOSITORIO
   ```

2. **Instala el paquete en modo editable (`-e`)**:
   ```bash
   pip install -e .
   ```

> 💡 **¿Por qué modo editable?** Esto creará un enlace simbólico en tu entorno de Python. Cualquier cambio o nueva fórmula matemática que agregues dentro de la carpeta `hep_models/` se reflejará instantáneamente en todo tu sistema, sin necesidad de reinstalar el paquete.

---

## 🧪 Ejemplo de Uso

Una vez instalado, abre cualquier terminal o entorno de desarrollo (Jupyter, VS Code, etc.) en **cualquier carpeta de tu computadora** y escribe:

```python
# Importar tus propios módulos desde cualquier parte del sistema
from hep_models.masses import diagonalize_mass_matrix
from hep_models.plotting import plot_higgs_potential

# 1. Ejemplo: Diagonalización del sector de masa (unidades en GeV)
M_quarks = [[0.002, 0.010, 0.0], 
            [0.010, 1.270, 0.0], 
            [0.0,   0.0,   172.7]]

masas_fisicas = diagonalize_mass_matrix(M_quarks)
print(f"Masas físicas obtenidas: {masas_fisicas} GeV")

# 2. Ejemplo: Graficar el potencial de Higgs (Ruptura Espontánea de la Simetría)
plot_higgs_potential(mu=125, lmbda=0.13)
```

---

## 📐 Convenciones Teóricas
* **Unidades Naturales:** $\hbar = c = 1$.
* **Métrica del Espacio-Tiempo:** Convención de West Coast $\eta_{\mu\nu} = \text{diag}(+1, -1, -1, -1)$.
* **Grupo de Norma del ME:** $SU(3)_C \times SU(2)_L \times U(1)_Y$.
