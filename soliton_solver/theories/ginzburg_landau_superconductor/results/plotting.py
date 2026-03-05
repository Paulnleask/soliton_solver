# =========================
# plotting.py
# =========================
"""
To run: python -m soliton_solver.theories.ginzburg_landau_superconductor.results.plotting
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from pathlib import Path
HERE = Path(__file__).resolve().parent

# Font setting
import matplotlib.font_manager as font_manager
matplotlib.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=matplotlib.get_data_path() + '/fonts/ttf/cmr10.ttf')
matplotlib.rcParams['font.serif']=cmfont.get_name()
matplotlib.rcParams['mathtext.fontset']='cm'
matplotlib.rcParams['axes.unicode_minus']=False

# Data loader
def load_dat(path):
    return np.loadtxt(HERE / path)

# Bicubic interpolation using scipy
def bicubic_upscale(arr, factor=2):
    return zoom(arr, zoom=factor, order=4)

# Creates top-down surface plot panel using pcolormesh 
def draw_panel(ax, X, Y, Z, title, xlen, ylen, xlim=None, ylim=None, cmap="jet"):
    # (works with nonuniform grids X,Y shaped like Z)
    pc = ax.pcolormesh(X, Y, Z, shading="auto", cmap=cmap)
    ax.set_aspect(xlen / ylen)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_title(title)
    ax.set_facecolor("none")
    plt.colorbar(pc, ax=ax)
    ax.grid(False)
    plt.axes(ax)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return pc

# Load lattice
latticePoints = load_dat("LatticePoints.dat")
latticeVectors = load_dat("LatticeVectors.dat")

xSize = int(latticePoints[0])
ySize = int(latticePoints[1])
X1 = np.array([latticeVectors[0], latticeVectors[1]])
X2 = np.array([latticeVectors[2], latticeVectors[3]])

# Load grid
xGrid = load_dat("yGrid.dat")
yGrid = load_dat("xGrid.dat")

xMin, xMax = np.min(xGrid), np.max(xGrid)
yMin, yMax = np.min(yGrid), np.max(yGrid)
xLength = xMax - xMin
yLength = yMax - yMin

# Spacing estimates (not strictly needed after interpolation)
xSpacing = np.abs(xGrid[0, 0] - xGrid[1, 0])
ySpacing = np.abs(yGrid[0, 0] - yGrid[0, 1])

# Load fields
Field1 = load_dat("HiggsField1.dat")
Field2 = load_dat("HiggsField2.dat")

Supercurrent1 = load_dat("Supercurrent1.dat")
Supercurrent2 = load_dat("Supercurrent2.dat")

GaugeField1 = load_dat("GaugeField1.dat")
GaugeField2 = load_dat("GaugeField2.dat")

EnergyDensity = load_dat("EnergyDensity.dat")
MagneticFluxDensityXY = load_dat("VortexDensity.dat")

# 2D bicubic interpolations
xGrid = bicubic_upscale(xGrid, 2)
yGrid = bicubic_upscale(yGrid, 2)

Field1 = bicubic_upscale(Field1, 2)
Field2 = bicubic_upscale(Field2, 2)

Supercurrent1 = bicubic_upscale(Supercurrent1, 2)
Supercurrent2 = bicubic_upscale(Supercurrent2, 2)

GaugeField1 = bicubic_upscale(GaugeField1, 2)
GaugeField2 = bicubic_upscale(GaugeField2, 2)

EnergyDensity = bicubic_upscale(EnergyDensity, 2)
MagneticFluxDensityXY = bicubic_upscale(MagneticFluxDensityXY, 2)

# Define modulus of Higgs field
psi1 = Field1 * Field1 + Field2 * Field2

# ---------- Figure 1: densities ----------
fig1 = plt.figure(figsize=(12, 6), dpi=500)
gs = fig1.add_gridspec(2, 3, wspace=0.25, hspace=0.35)

# Energy density
ax11 = fig1.add_subplot(gs[0, 0])
draw_panel(ax11, xGrid, yGrid, EnergyDensity, r"$\mathcal{E}(\vec{x})$", xLength, yLength, xlim=(xMin,xMax), ylim=(yMin,yMax))

# |psi|^2 density
ax12 = fig1.add_subplot(gs[0, 1])
draw_panel(ax12, xGrid, yGrid, psi1, r"$|\psi(\vec{x})|^2$", xLength, yLength, xlim=(xMin,xMax), ylim=(yMin,yMax))

# Magnetic flux density (with -/(2π))
ax13 = fig1.add_subplot(gs[0, 2])
draw_panel(ax13, xGrid, yGrid, MagneticFluxDensityXY, r"$\Phi(\vec{x})$", xLength, yLength, xlim=(xMin,xMax), ylim=(yMin,yMax), cmap="bone")

# Plots and labels the figure
fig1.savefig(HERE / "Plot_Densities.png", dpi=500, bbox_inches="tight")