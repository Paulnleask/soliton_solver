# =====================================================================================
# soliton_solver/core/colormaps.py
# =====================================================================================
"""
CUDA colormap and rendering kernels for writing RGBA images into a mapped PBO.

Purpose:
    Provides simple CUDA kernels that map scalar densities or vector magnetization
    fields to 8-bit RGBA pixels suitable for CUDA-OpenGL interop rendering.

Usage:
    - Allocate/map an output buffer as a uint8 RGBA array: pbo_rgba[y, x, 0..3].
    - Launch a 2D CUDA grid covering (xlen, ylen).
    - Provide either:
        - density_flat with idx = y + x * ylen, or
        - Field containing stacked magnetization planes for HSV rendering.

Outputs:
    - Writes RGBA pixels into pbo_rgba with alpha set to 255.
"""

# ---------------- Imports ----------------
import math
from numba import cuda

# ---------------- Clipping ----------------
@cuda.jit(device=True, inline=True)
def clip_u8(x):
    """
    Clamp a float to [0, 255] and convert to an integer suitable for uint8 storage.

    Usage:
        pbo_rgba[y, x, 0] = clip_u8(r * 255.0)

    Parameters:
        x: Float value to clamp.

    Outputs:
        - Returns an int in [0, 255].
    """
    if x < 0.0:
        return 0
    if x > 255.0:
        return 255
    return int(x)

# ---------------- HSV2RGB ----------------
@cuda.jit(device=True, inline=True)
def _hsv_to_rgb(h, s, v):
    """
    Convert HSV values to RGB.

    Usage:
        R, G, B = _hsv_to_rgb(hue_deg, saturation, value)

    Parameters:
        h: Hue in degrees (wraps values >= 360 to 0).
        s: Saturation in [0, 1].
        v: Value/brightness in [0, 1+] (may exceed 1 for brightening).

    Outputs:
        - Returns (R, G, B) as floats (typically in [0, 1+]).
    """
    if h >= 360.0:
        h = 0.0
    h = h / 60.0
    i = int(h)
    ff = h - i
    p = v * (1.0 - s)
    r = v * (1.0 - (s * ff))
    t = v * (1.0 - (s * (1.0 - ff)))

    if i == 0:
        R, G, B = v, t, p
    elif i == 1:
        R, G, B = r, v, p
    elif i == 2:
        R, G, B = p, v, t
    elif i == 3:
        R, G, B = p, r, v
    elif i == 4:
        R, G, B = t, p, v
    else:
        R, G, B = v, p, r
    return R, G, B

# ---------------- Jet colormap ----------------
@cuda.jit
def render_jet_density_to_rgba(pbo_rgba, density_flat, xlen, ylen, vmin, vmax):
    """
    Render a scalar density field to RGBA using a jet-style piecewise-linear colormap.

    Usage:
        render_jet_density_to_rgba[grid2d, block2d](pbo_rgba, density_flat, xlen, ylen, vmin, vmax)

    Parameters:
        pbo_rgba: (ylen, xlen, 4) uint8 device array written in-place.
        density_flat: (xlen*ylen,) float device array, indexed as idx = y + x * ylen.
        xlen, ylen: Image dimensions.
        vmin, vmax: Normalization range for density values.

    Outputs:
        - Writes (R, G, B, A) into pbo_rgba for each pixel.
        - Sets A = 255.
    """
    x, y = cuda.grid(2)
    if x >= xlen or y >= ylen:
        return

    idx = y + x * ylen
    val = density_flat[idx]
    hue = (val - vmin) / (vmax - vmin + 1e-30)
    if hue < 0.0:
        hue = 0.0
    if hue > 1.0:
        hue = 1.0

    if hue <= 1.0 / 8.0:
        r = 0.0
        g = 0.0
        b = (4.0 * hue + 0.5) * 255.0
    elif hue <= 3.0 / 8.0:
        r = 0.0
        g = (4.0 * hue - 0.5) * 255.0
        b = 255.0
    elif hue <= 5.0 / 8.0:
        r = (4.0 * hue - 1.5) * 255.0
        g = 255.0
        b = (-4.0 * hue + 2.5) * 255.0
    elif hue <= 7.0 / 8.0:
        r = 255.0
        g = (-4.0 * hue + 3.5) * 255.0
        b = 0.0
    else:
        r = (-4.0 * hue + 4.5) * 255.0
        g = 0.0
        b = 0.0

    pbo_rgba[y, x, 0] = clip_u8(r)
    pbo_rgba[y, x, 1] = clip_u8(g)
    pbo_rgba[y, x, 2] = clip_u8(b)
    pbo_rgba[y, x, 3] = 255

# ---------------- Grayscale colormap ----------------
@cuda.jit
def render_gray_density_to_rgba(pbo_rgba, density_flat, xlen, ylen, vmin, vmax):
    """
    Render a scalar density field to RGBA as grayscale.

    Usage:
        render_gray_density_to_rgba[grid2d, block2d](pbo_rgba, density_flat, xlen, ylen, vmin, vmax)

    Parameters:
        pbo_rgba: (ylen, xlen, 4) uint8 device array written in-place.
        density_flat: (xlen*ylen,) float device array, indexed as idx = y + x * ylen.
        xlen, ylen: Image dimensions.
        vmin, vmax: Normalization range for density values.

    Outputs:
        - Writes R=G=B=normalized*255 and A=255 into pbo_rgba for each pixel.
    """
    x, y = cuda.grid(2)
    if x >= xlen or y >= ylen:
        return
    idx = y + x * ylen
    val = density_flat[idx]
    hue = (val - vmin) / (vmax - vmin + 1e-30)
    if hue < 0.0:
        hue = 0.0
    if hue > 1.0:
        hue = 1.0
    c = hue * 255.0
    uc = clip_u8(c)
    pbo_rgba[y, x, 0] = uc
    pbo_rgba[y, x, 1] = uc
    pbo_rgba[y, x, 2] = uc
    pbo_rgba[y, x, 3] = 255

# ---------------- Runge colormap ----------------
@cuda.jit
def render_magnetization_to_rgba(pbo_rgba, Field, xlen, ylen, p_i):
    """
    Render a 3-component magnetization field to RGBA using an HSV color wheel mapping.

    Usage:
        render_magnetization_to_rgba[grid2d, block2d](pbo_rgba, Field, xlen, ylen, p_i)

    Parameters:
        pbo_rgba: (ylen, xlen, 4) uint8 device array written in-place.
        Field: Flattened float device array storing magnetization planes (m1, m2, m3).
        xlen, ylen: Image dimensions.
        p_i: Device int array used to compute plane stride:
            - p_i[0]: x-dimension used in plane stride
            - p_i[1]: y-dimension used in base index

    Outputs:
        - Interprets Field as three stacked planes:
            base = y + x * ylen
            plane = xlen * ylen  (computed from p_i)
            m1 = Field[base + 0*plane], m2 = Field[base + 1*plane], m3 = Field[base + 2*plane]
        - Computes hue from atan2(m1, m2); shapes saturation/value using m3.
        - Writes (R, G, B, A) into pbo_rgba with A = 255.
    """
    x, y = cuda.grid(2)
    if x >= xlen or y >= ylen:
        return

    ylen_i = int(p_i[1])
    plane = int(p_i[0]) * ylen_i
    base = y + x * ylen_i

    m1 = Field[base + 0 * plane]
    m2 = Field[base + 1 * plane]
    m3 = Field[base + 2 * plane]

    # HSV
    hue = (0.5 + (1.0 / (2.0 * math.pi)) * math.atan2(m1, m2)) * 360.0
    saturation = 0.5 - 0.5 * math.tanh(3.0 * (m3 - 0.5))
    value = 1.0 + m3

    # HSV2RGB
    R, G, B = _hsv_to_rgb(hue, saturation, value)

    pbo_rgba[y, x, 0] = clip_u8(R * 255.0)
    pbo_rgba[y, x, 1] = clip_u8(G * 255.0)
    pbo_rgba[y, x, 2] = clip_u8(B * 255.0)
    pbo_rgba[y, x, 3] = 255