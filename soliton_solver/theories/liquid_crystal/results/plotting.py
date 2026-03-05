# =========================
# plotting.py
# =========================
"""
To run: python -m soliton_solver.theories.liquid_crystal.results.plotting
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
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
EnergyDensity = load_dat("EnergyDensity.dat")
ChargeDensity = load_dat("SkyrmionChargeDensity.dat")
MagneticChargeDensity = load_dat("ElectricChargeDensity.dat")
MagnetField1 = load_dat("MagnetField1.dat")
MagnetField2 = load_dat("MagnetField2.dat")
MagnetField3 = load_dat("MagnetField3.dat")

# 2D bicubic interpolations
EnergyDensity = bicubic_upscale(EnergyDensity, 2)
ChargeDensity = bicubic_upscale(ChargeDensity, 2)
MagneticChargeDensity = bicubic_upscale(MagneticChargeDensity, 2)
MagnetField1 = bicubic_upscale(MagnetField1, 2)
MagnetField2 = bicubic_upscale(MagnetField2, 2)
MagnetField3 = bicubic_upscale(MagnetField3, 2)

xGrid = bicubic_upscale(xGrid, 2)
yGrid = bicubic_upscale(yGrid, 2)

# Runge color construction in HSV
H = 0.5 + (1.0 / (2.0 * np.pi)) * np.arctan2(MagnetField1, MagnetField2)  # hue
S = 0.5 - 0.5 * np.tanh(3.0 * (MagnetField3 - 0.5))                       # saturation
V = MagnetField3 + 1.0                                                    # value
# Stack HSV -> RGB
rgbOutput = np.clip(hsv_to_rgb(np.stack([H, S, V], axis=-1)), 0, 1)

# ---------- Figure 1: densities ----------
fig1 = plt.figure(figsize=(12, 6), dpi=500)
gs = fig1.add_gridspec(2, 3, wspace=0.25, hspace=0.35)

# Energy density
ax11 = fig1.add_subplot(gs[0, 0])
draw_panel(ax11, xGrid, yGrid, EnergyDensity, r"$\mathcal{E}(\vec{x})$", xLength, yLength, xlim=(xMin,xMax), ylim=(yMin,yMax))

# Magnetic charge density
ax12 = fig1.add_subplot(gs[0, 1])
draw_panel(ax12, xGrid, yGrid, MagneticChargeDensity, r"$\rho(\vec{x})=-\vec{\nabla}\cdot\vec{P}[\vec{n}]$", xLength, yLength, xlim=(xMin,xMax), ylim=(yMin,yMax),cmap="bone")

# Runge coloring (use the RGB image directly)
ax13 = fig1.add_subplot(gs[0, 2])
im = ax13.imshow(rgbOutput, origin="lower", extent=[xMin, xMax, yMin, yMax], interpolation="nearest")
ax13.set_aspect(xLength / yLength)
ax13.set_xlim(xMin, xMax)
ax13.set_ylim(yMin, yMax)
ax13.set_title(r"$\vec{n}(\vec{x})$")
ax13.set_facecolor("none")
for spine in ax13.spines.values():
    spine.set_visible(False)

# Plots the colorbar for the Runge sphere coloring
inset = fig1.add_axes([0.89, 0.59, 0.025, 0.32])
ps_n = 1000
ps_theta = np.linspace(-np.pi, np.pi, ps_n)
ps_sigma = np.linspace(-1.0, 1.0, ps_n // 2)
ps_h = 0.5 + (1.0 / (2.0 * np.pi)) * ps_theta
ps_s = 0.5 - 0.5 * np.tanh(3.0 * (ps_sigma - 0.5))
ps_v = ps_sigma + 1.0
ps_S, ps_H = np.meshgrid(ps_s, ps_h)
ps_V, _ = np.meshgrid(ps_v, ps_h)
ps_rgb = np.clip(hsv_to_rgb(np.stack([ps_H, ps_S, ps_V], axis=-1)), 0, 1)
# Display with axes
inset.imshow(ps_rgb, origin="lower", extent=[ps_sigma.min(), ps_sigma.max(), ps_theta.min(), ps_theta.max()], interpolation="nearest")
inset.set_xlabel(r"$m_z$")
inset.set_ylabel(r"$\arg(m_x+im_y)$")
inset.yaxis.set_label_position("right")
inset.yaxis.tick_right()
inset.set_yticks([-np.pi, np.pi])
inset.set_yticklabels(["0", r"$2\pi$"])
inset.set_xticks([-1, 1])
for spine in inset.spines.values():
    spine.set_visible(True)

fig1.savefig(HERE / "Plot_Densities.png", dpi=500, bbox_inches="tight")