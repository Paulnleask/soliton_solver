"""
CUDA colormap and rendering kernels for writing RGBA images into a mapped PBO.

Examples
--------
Use ``render_jet_density_to_rgba`` to map a scalar density field to RGBA pixels.
Use ``render_gray_density_to_rgba`` to render a scalar density field in grayscale.
Use ``render_magnetization_to_rgba`` to map a magnetization field to RGBA pixels.
"""

import math
from numba import cuda

@cuda.jit(device=True, inline=True)
def clip_u8(x):
    """
    Clamp a float to the interval [0, 255] and convert it to an integer.

    Parameters
    ----------
    x : float
        Value to clamp.

    Returns
    -------
    int
        Clamped integer value in the interval [0, 255].

    Raises
    ------
    None.

    Examples
    --------
    Use ``clip_u8(r * 255.0)`` before writing a channel value into an RGBA buffer.
    """
    if x < 0.0:
        return 0
    if x > 255.0:
        return 255
    return int(x)

@cuda.jit(device=True, inline=True)
def _hsv_to_rgb(h, s, v):
    """
    Convert HSV values to RGB values.

    Parameters
    ----------
    h : float
        Hue in degrees.
    s : float
        Saturation value.
    v : float
        Value component.

    Returns
    -------
    tuple of float
        Red, green, and blue channel values.

    Raises
    ------
    None.

    Examples
    --------
    Use ``_hsv_to_rgb(hue_deg, saturation, value)`` to convert HSV values before writing RGB output.
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

@cuda.jit
def render_jet_density_to_rgba(pbo_rgba, density_flat, xlen, ylen, vmin, vmax):
    """
    Render a scalar density field to RGBA using a jet-style colormap.

    Parameters
    ----------
    pbo_rgba : device array
        Output RGBA buffer with shape ``(ylen, xlen, 4)``.
    density_flat : device array
        Flattened scalar field indexed as ``y + x * ylen``.
    xlen : int
        Number of grid points along the x direction.
    ylen : int
        Number of grid points along the y direction.
    vmin : float
        Lower bound of the normalization range.
    vmax : float
        Upper bound of the normalization range.

    Returns
    -------
    None
        The RGBA values are written into ``pbo_rgba`` in place.

    Raises
    ------
    None.

    Examples
    --------
    Launch ``render_jet_density_to_rgba[grid2d, block2d](pbo_rgba, density_flat, xlen, ylen, vmin, vmax)`` to render the density field.
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

@cuda.jit
def render_gray_density_to_rgba(pbo_rgba, density_flat, xlen, ylen, vmin, vmax):
    """
    Render a scalar density field to RGBA in grayscale.

    Parameters
    ----------
    pbo_rgba : device array
        Output RGBA buffer with shape ``(ylen, xlen, 4)``.
    density_flat : device array
        Flattened scalar field indexed as ``y + x * ylen``.
    xlen : int
        Number of grid points along the x direction.
    ylen : int
        Number of grid points along the y direction.
    vmin : float
        Lower bound of the normalization range.
    vmax : float
        Upper bound of the normalization range.

    Returns
    -------
    None
        The RGBA values are written into ``pbo_rgba`` in place.

    Raises
    ------
    None.

    Examples
    --------
    Launch ``render_gray_density_to_rgba[grid2d, block2d](pbo_rgba, density_flat, xlen, ylen, vmin, vmax)`` to render the density field.
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

@cuda.jit
def render_magnetization_to_rgba(pbo_rgba, Field, xlen, ylen, p_i):
    """
    Render a three-component magnetization field to RGBA using an HSV mapping.

    Parameters
    ----------
    pbo_rgba : device array
        Output RGBA buffer with shape ``(ylen, xlen, 4)``.
    Field : device array
        Flattened array containing three stacked magnetization planes.
    xlen : int
        Number of grid points along the x direction.
    ylen : int
        Number of grid points along the y direction.
    p_i : device array
        Integer array used to determine the plane stride and base indexing.

    Returns
    -------
    None
        The RGBA values are written into ``pbo_rgba`` in place.

    Raises
    ------
    None.

    Examples
    --------
    Launch ``render_magnetization_to_rgba[grid2d, block2d](pbo_rgba, Field, xlen, ylen, p_i)`` to render the magnetization field.
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