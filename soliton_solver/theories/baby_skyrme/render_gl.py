"""
Baby Skyrme specific CUDA OpenGL renderer logic.

Examples
--------
Use ``GLRenderer`` to render the Baby Skyrme simulation interactively.
Use ``advance_solver`` to step the simulation between rendered frames.
Use ``run_viewer`` to start the interactive viewer loop.
"""

from __future__ import annotations
import math
import numpy as np
import glfw
from numba import cuda
from soliton_solver.visualization.gl_backend import GLBackend, cuda_array_from_ptr
from soliton_solver.core.utils import compute_min, compute_max
from soliton_solver.core.utils import launch_2d
from soliton_solver.theories.baby_skyrme.initial_config import create_skyrmion_kernel
from soliton_solver.core.colormaps import clip_u8, _hsv_to_rgb, render_jet_density_to_rgba, render_gray_density_to_rgba, render_magnetization_to_rgba

DISPLAY_ENERGY = 1
DISPLAY_MAGNETIZATION = 2

DISPLAY_TITLES = {
    DISPLAY_ENERGY: "En",
    DISPLAY_MAGNETIZATION: "Sk",
}

@cuda.jit
def overlay_skyrmion_preview_kernel(pbo_rgba, grid, pxi, pxj, rotation_angle, ansatz_bloch, ansatz_neel, ansatz_anti, p_i, p_f):
    """
    Draw a translucent skyrmion preview into the RGBA pixel buffer.

    Parameters
    ----------
    pbo_rgba : device array
        Mapped RGBA pixel buffer.
    grid : device array
        Flattened coordinate grid.
    pxi : int
        x index of the preview center.
    pxj : int
        y index of the preview center.
    rotation_angle : float
        Rotation angle applied to the preview.
    ansatz_bloch : bool
        Flag selecting the Bloch ansatz.
    ansatz_neel : bool
        Flag selecting the Neel ansatz.
    ansatz_anti : bool
        Flag selecting the anti-skyrmion ansatz.
    p_i : device array
        Integer device parameter array.
    p_f : device array
        Float device parameter array.

    Returns
    -------
    None
        The preview colors are blended into ``pbo_rgba`` in place.

    Examples
    --------
    Launch ``overlay_skyrmion_preview_kernel[grid2d, block2d](pbo_rgba, grid, pxi, pxj, rotation_angle, ansatz_bloch, ansatz_neel, ansatz_anti, p_i, p_f)`` to draw the preview overlay.
    """
    x, y = cuda.grid(2)

    xlen = int(p_i[0])
    ylen = int(p_i[1])
    if x >= xlen or y >= ylen:
        return

    if pxi < 0 or pxi >= xlen or pxj < 0 or pxj >= ylen:
        return

    xsize = p_f[0]
    ysize = p_f[1]
    skN = p_f[8]
    rmax = xsize if xsize > ysize else ysize

    plane = xlen * ylen

    base_c = int(pxj) + int(pxi) * ylen
    xcent = grid[base_c + 0 * plane]
    ycent = grid[base_c + 1 * plane]

    base = y + x * ylen
    gx = grid[base + 0 * plane] - xcent
    gy = grid[base + 1 * plane] - ycent
    r1 = math.sqrt(gx * gx + gy * gy)

    theta = -rotation_angle
    if r1 != 0.0:
        ang = math.atan2(gy, gx)
        if ang < 0.0:
            ang += 2.0 * math.pi
        theta += ang
    if skN == 0:
        max_r = rmax / 10.0 * 2.0
    else:
        max_r = rmax / 10.0 * skN
    if r1 > max_r:
        fm = 0.0
    else:
        t = (2.0 * r1 / max_r)
        fm = math.pi * math.exp(-(t * t))

    if skN == 0:
        if ansatz_bloch:
            mx = -math.sin(2.0 * fm) * math.sin(theta)
            my =  math.sin(2.0 * fm) * math.cos(theta)
            mz =  math.cos(2.0 * fm)
        elif ansatz_neel:
            mx =  math.sin(2.0 * fm) * math.cos(theta)
            my =  math.sin(2.0 * fm) * math.sin(theta)
            mz =  math.cos(2.0 * fm)
        elif ansatz_anti:
            mx = -math.sin(2.0 * fm) * math.sin(theta)
            my = -math.sin(2.0 * fm) * math.cos(theta)
            mz =  math.cos(2.0 * fm)
        else:
            mx = 0.0
            my = 0.0
            mz = 1.0
    else:
        if ansatz_bloch:
            mx = -math.sin(fm) * math.sin(skN * theta)
            my =  math.sin(fm) * math.cos(skN * theta)
            mz =  math.cos(fm)
        elif ansatz_neel:
            mx =  math.sin(fm) * math.cos(skN * theta)
            my =  math.sin(fm) * math.sin(skN * theta)
            mz =  math.cos(fm)
        elif ansatz_anti:
            mx = -math.sin(fm) * math.sin(skN * theta)
            my = -math.sin(fm) * math.cos(skN * theta)
            mz =  math.cos(fm)
        else:
            mx = 0.0
            my = 0.0
            mz = 1.0

    hue = (0.5 + (1.0 / (2.0 * math.pi)) * math.atan2(mx, my)) * 360.0
    saturation = 0.5 - 0.5 * math.tanh(3.0 * (mz - 0.5))
    value = mz + 1.0
    R, G, B = _hsv_to_rgb(hue, saturation, value)

    if skN == 0:
        sigma = 0.10 * rmax * 2.0
    else:
        sigma = 0.10 * rmax * skN
    if sigma <= 0.0:
        return
    fade = math.exp(-(r1 * r1) / (2.0 * sigma * sigma))
    a = 0.55 * fade

    br = float(pbo_rgba[y, x, 0])
    bg = float(pbo_rgba[y, x, 1])
    bb = float(pbo_rgba[y, x, 2])

    pbo_rgba[y, x, 0] = clip_u8((1.0 - a) * br + a * (R * 255.0))
    pbo_rgba[y, x, 1] = clip_u8((1.0 - a) * bg + a * (G * 255.0))
    pbo_rgba[y, x, 2] = clip_u8((1.0 - a) * bb + a * (B * 255.0))
    pbo_rgba[y, x, 3] = 255

class GLRenderer:
    """
    Interactive renderer for the Baby Skyrme simulation.

    Examples
    --------
    Use ``renderer = GLRenderer(width=xlen, height=ylen)`` to create the renderer.
    Use ``renderer.bind_sim(sim)`` to attach a simulation.
    Use ``renderer.render(...)`` inside the display loop to draw a frame.
    """

    def __init__(self, width: int, height: int, title: str = "superferro (CUDA-OpenGL)"):
        """
        Create the renderer and its GLFW OpenGL backend.

        Parameters
        ----------
        width : int
            Viewport width in pixels.
        height : int
            Viewport height in pixels.
        title : str, optional
            Window title.

        Returns
        -------
        None
            The renderer and backend are initialized.

        Examples
        --------
        Use ``renderer = GLRenderer(params.xlen, params.ylen)`` to create the viewer.
        """
        self.backend = GLBackend(width, height, title=title)

        self.width = int(width)
        self.height = int(height)

        self.display_mode = DISPLAY_ENERGY

        self.hud_top_text = ""
        self.hud_bottom_text = ""

        self._sim = None

        self.soliton_mode = "none"

        self.preview_active = False
        self.preview_pxi = 0
        self.preview_pxj = 0
        self.iso_rotating = False
        self.iso_start_mouse_x = 0.0
        self.iso_start_angle = 0.0
        self.skyrmion_rotation = 0.0

        self.vortex_type = 0
        self.request_save_output = False

        glfw.set_key_callback(self.backend.window, self._on_key)
        glfw.set_cursor_pos_callback(self.backend.window, self._on_cursor_pos)
        glfw.set_mouse_button_callback(self.backend.window, self._on_mouse_button)

        self._update_window_title()

    def bind_sim(self, sim):
        """
        Attach a simulation instance to the renderer.

        Parameters
        ----------
        sim
            Simulation instance providing fields and parameter arrays.

        Returns
        -------
        None
            The simulation reference is stored by the renderer.

        Examples
        --------
        Use ``renderer.bind_sim(sim)`` to enable preview and skyrmion insertion.
        """
        self._sim = sim
        try:
            self.skyrmion_rotation = float(sim.p_f_h[9])
        except Exception:
            self.skyrmion_rotation = 0.0

    def _window_to_sim(self, x, y):
        """
        Convert a window position to simulation lattice indices.

        Parameters
        ----------
        x : float
            Window x coordinate.
        y : float
            Window y coordinate.

        Returns
        -------
        tuple
            Pair ``(sim_x, sim_y)`` of lattice indices.

        Examples
        --------
        Use ``sim_x, sim_y = self._window_to_sim(x, y)`` to map the cursor to the simulation grid.
        """
        sim = self._sim
        if sim is None:
            return 0, 0

        win_w, win_h = glfw.get_window_size(self.backend.window)
        if win_w < 1: win_w = 1
        if win_h < 1: win_h = 1

        xlen = int(sim.p_i_h[0])
        ylen = int(sim.p_i_h[1])

        nx = x / win_w
        ny = y / win_h
        ny = 1.0 - ny

        sim_x = int(nx * xlen)
        sim_y = int(ny * ylen)

        sim_x = max(0, min(xlen - 1, sim_x))
        sim_y = max(0, min(ylen - 1, sim_y))
        return sim_x, sim_y

    @staticmethod
    def _wrap2pi(a: float) -> float:
        """
        Wrap an angle to the interval ``[0, 2π)``.

        Parameters
        ----------
        a : float
            Input angle.

        Returns
        -------
        float
            Wrapped angle.

        Examples
        --------
        Use ``a = self._wrap2pi(a)`` to keep the rotation angle in range.
        """
        twopi = 2.0 * math.pi
        a = math.fmod(a, twopi)
        if a < 0.0:
            a += twopi
        return a

    def _on_key(self, window, key, scancode, action, mods):
        """
        Handle GLFW key press events.

        Parameters
        ----------
        window
            GLFW window handle.
        key : int
            GLFW key code.
        scancode : int
            Platform specific scan code.
        action : int
            GLFW action code.
        mods : int
            GLFW modifier flags.

        Returns
        -------
        None
            The renderer state is updated in response to the key press.

        Examples
        --------
        Use ``glfw.set_key_callback(self.backend.window, self._on_key)`` to register the key handler.
        """
        if action != glfw.PRESS:
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(self.backend.window, True)
            return

        if key == glfw.KEY_F1:
            self.set_display_mode(DISPLAY_ENERGY)
        elif key == glfw.KEY_F2:
            self.set_display_mode(DISPLAY_MAGNETIZATION)

        elif key == glfw.KEY_S:
            self.soliton_mode = "skyrmion"
            self.preview_active = True
            self.iso_rotating = False

        elif glfw.KEY_0 <= key <= glfw.KEY_9:
            n = float(key - glfw.KEY_0)
            self._set_topological_number(n)

        elif key == glfw.KEY_N and self._sim is not None:
            self._sim.set_params(self._sim.p.with_(newtonflow=not self._sim.p.newtonflow))

        elif key == glfw.KEY_K and self._sim is not None:
            self._sim.set_params(self._sim.p.with_(killkinen=not self._sim.p.killkinen))

        elif key == glfw.KEY_Q:
            self.soliton_mode = "none"
            self.preview_active = False
            self.iso_rotating = False

        elif key == glfw.KEY_O:
            self.request_save_output = True

    def _set_topological_number(self, n: float):
        """
        Update the skyrmion number in the bound simulation parameters.

        Parameters
        ----------
        n : float
            New skyrmion number.

        Returns
        -------
        None
            The host and device parameter arrays are updated.

        Examples
        --------
        Use ``self._set_topological_number(n)`` to update the skyrmion number from a key press.
        """
        sim = getattr(self, "_sim", None)
        if sim is None:
            return

        sim.p_f_h[8] = float(n)

        try:
            sim.p_f_d.copy_to_device(sim.p_f_h)
        except Exception:
            sim.p_f_d = cuda.to_device(sim.p_f_h)

    def _on_cursor_pos(self, window, xpos, ypos):
        """
        Handle GLFW mouse motion events.

        Parameters
        ----------
        window
            GLFW window handle.
        xpos : float
            Cursor x position.
        ypos : float
            Cursor y position.

        Returns
        -------
        None
            The preview position or rotation state is updated.

        Examples
        --------
        Use ``glfw.set_cursor_pos_callback(self.backend.window, self._on_cursor_pos)`` to register the cursor handler.
        """
        if self._sim is None:
            return

        if self.soliton_mode != "skyrmion":
            self.preview_active = False
            return
        self.preview_active = True

        if self.iso_rotating:
            fb_w, _fb_h = glfw.get_framebuffer_size(self.backend.window)
            if fb_w < 1:
                fb_w = 1
            ndx = (float(xpos) - float(self.iso_start_mouse_x)) / float(fb_w)
            self.skyrmion_rotation = self._wrap2pi(self.iso_start_angle + (2.0 * math.pi) * ndx)
            return

        simX, simY = self._window_to_sim(xpos, ypos)
        self.preview_pxi = int(simX)
        self.preview_pxj = int(simY)

    def _on_mouse_button(self, window, button, action, mods):
        """
        Handle GLFW mouse button events.

        Parameters
        ----------
        window
            GLFW window handle.
        button : int
            GLFW mouse button code.
        action : int
            GLFW action code.
        mods : int
            GLFW modifier flags.

        Returns
        -------
        None
            The preview state or simulation field is updated in response to the mouse action.

        Examples
        --------
        Use ``glfw.set_mouse_button_callback(self.backend.window, self._on_mouse_button)`` to register the mouse handler.
        """
        if self._sim is None:
            return

        if button != glfw.MOUSE_BUTTON_LEFT:
            return

        sim = self._sim

        if self.soliton_mode == "skyrmion":
            if action == glfw.PRESS:
                self.preview_active = True
                self.iso_rotating = True
                self.iso_start_mouse_x = float(glfw.get_cursor_pos(self.backend.window)[0])
                self.iso_start_angle = float(self.skyrmion_rotation)

                x, y = glfw.get_cursor_pos(self.backend.window)
                simX, simY = self._window_to_sim(x, y)
                self.preview_pxi = int(simX)
                self.preview_pxj = int(simY)
                return

            if action == glfw.RELEASE:
                self.iso_rotating = False

                p_f = sim.p_f_h
                ans_bloch = bool(p_f[10] > 0.5)
                ans_neel = bool(p_f[11] > 0.5)
                ans_anti = bool(p_f[12] > 0.5)

                grid2d, block2d = launch_2d(sim.p_i_h, threads=(8, 8))
                create_skyrmion_kernel[grid2d, block2d](
                    sim.Field, sim.grid,
                    int(self.preview_pxi), int(self.preview_pxj),
                    float(self.skyrmion_rotation),
                    ans_bloch, ans_neel, ans_anti,
                    sim.p_i_d, sim.p_f_d
                )
                cuda.synchronize()
                return

    def set_display_mode(self, mode: int):
        """
        Set the active display mode.

        Parameters
        ----------
        mode : int
            Display mode identifier.

        Returns
        -------
        None
            The renderer display mode is updated.

        Examples
        --------
        Use ``renderer.set_display_mode(DISPLAY_MAGNETIZATION)`` to switch the display mode.
        """
        if mode not in DISPLAY_TITLES:
            return
        self.display_mode = mode
        self._update_window_title()

    def _update_window_title(self):
        """
        Update the window title to match the active display mode.

        Returns
        -------
        None
            The backend window title is updated.

        Examples
        --------
        Use ``self._update_window_title()`` after changing the display mode.
        """
        self.backend.set_window_title(DISPLAY_TITLES.get(self.display_mode, "superferro"))

    def close(self):
        """
        Destroy the OpenGL resources and close the window.

        Returns
        -------
        None
            The renderer backend is closed.

        Examples
        --------
        Use ``renderer.close()`` when the viewer loop exits.
        """
        self.backend.close()

    def should_close(self) -> bool:
        """
        Return whether the window should close.

        Returns
        -------
        bool
            ``True`` if the window should close.

        Examples
        --------
        Use ``while not renderer.should_close():`` as the main viewer loop condition.
        """
        return self.backend.should_close()

    def begin_frame(self):
        """
        Poll events at the beginning of a frame.

        Returns
        -------
        None
            Frame setup is delegated to the backend.

        Examples
        --------
        Use ``renderer.begin_frame()`` at the start of each frame.
        """
        self.backend.begin_frame()

    def end_frame(self):
        """
        Swap buffers at the end of a frame.

        Returns
        -------
        None
            Frame presentation is delegated to the backend.

        Examples
        --------
        Use ``renderer.end_frame()`` after rendering each frame.
        """
        self.backend.end_frame()

    def set_hud_text(self, *, top: str = "", bottom: str = ""):
        """
        Set the HUD text displayed by the renderer.

        Parameters
        ----------
        top : str, optional
            Top HUD text line.
        bottom : str, optional
            Bottom HUD text line.

        Returns
        -------
        None
            The HUD text is updated.

        Examples
        --------
        Use ``renderer.set_hud_text(top="...", bottom="...")`` to update the HUD.
        """
        self.backend.set_hud_text(top=top, bottom=bottom)

    def render(self, *, Field, density_flat, xlen: int, ylen: int, p_i, vmin: float, vmax: float, use_jet: bool):
        """
        Render one frame to the OpenGL window.

        Parameters
        ----------
        Field : device array
            Simulation field used for magnetization rendering.
        density_flat : device array
            Scalar density used for scalar display modes.
        xlen : int
            Number of lattice points along the x direction.
        ylen : int
            Number of lattice points along the y direction.
        p_i : device array
            Integer device parameter array.
        vmin : float
            Lower color normalization bound.
        vmax : float
            Upper color normalization bound.
        use_jet : bool
            Flag selecting the jet colormap for scalar rendering.

        Returns
        -------
        None
            The current frame is rendered to the window.

        Examples
        --------
        Use ``renderer.render(Field=sim.Field, density_flat=sim.en, xlen=params.xlen, ylen=params.ylen, p_i=sim.p_i_d, vmin=vmin, vmax=vmax, use_jet=True)`` to draw a frame.
        """
        mapped = self.backend.map_pbo()
        try:
            pbo_view = cuda_array_from_ptr(mapped.ptr, (self.height, self.width, 4), np.uint8)

            grid2d, block2d = launch_2d(p_i, threads=(8, 8))

            if self.display_mode == DISPLAY_MAGNETIZATION:
                render_magnetization_to_rgba[grid2d, block2d](pbo_view, Field, xlen, ylen, p_i)
            else:
                if use_jet:
                    render_jet_density_to_rgba[grid2d, block2d](pbo_view, density_flat, xlen, ylen, float(vmin), float(vmax))
                else:
                    render_gray_density_to_rgba[grid2d, block2d](pbo_view, density_flat, xlen, ylen, float(vmin), float(vmax))

            if self.preview_active and self.soliton_mode == "skyrmion" and self._sim is not None:
                p_f = self._sim.p_f_h
                ans_bloch = bool(p_f[10] > 0.5)
                ans_neel = bool(p_f[11] > 0.5)
                ans_anti = bool(p_f[12] > 0.5)
                overlay_skyrmion_preview_kernel[grid2d, block2d](
                    pbo_view, self._sim.grid,
                    int(self.preview_pxi), int(self.preview_pxj),
                    float(self.skyrmion_rotation),
                    ans_bloch, ans_neel, ans_anti,
                    self._sim.p_i_d, self._sim.p_f_d
                )

            cuda.synchronize()

        finally:
            self.backend.unmap_pbo()

        self.backend.upload_and_draw()

def advance_solver(sim, steps_per_frame: int, state: dict):
    """
    Advance the simulation by a fixed number of solver steps.

    Parameters
    ----------
    sim
        Simulation instance.
    steps_per_frame : int
        Number of solver steps per rendered frame.
    state : dict
        Mutable state dictionary storing energy, error, and epoch count.

    Returns
    -------
    None
        The state dictionary is updated in place.

    Examples
    --------
    Use ``advance_solver(sim, steps_per_frame, state)`` once per rendered frame.
    """
    if state.get("energy") is None:
        obs = sim.observables()
        state["energy"] = float(obs["energy"])
    if sim.rp.newtonflow:
        for _ in range(steps_per_frame):
            state["energy"], state["error"] = sim.step(state["energy"])
            state["epochs"] += 1

def _compute_density_for_mode(sim, mode: int):
    """
    Compute the scalar density required by the current display mode.

    Parameters
    ----------
    sim
        Simulation instance.
    mode : int
        Active display mode.

    Returns
    -------
    None
        The density buffer is updated in place when needed.

    Examples
    --------
    Use ``_compute_density_for_mode(sim, renderer.display_mode)`` before rendering a frame.
    """
    if mode == DISPLAY_ENERGY:
        if hasattr(sim, "compute_energy_density"):
            sim.compute_energy_density()
            return

def run_viewer(sim, params, *, steps_per_frame: int = 5, fps_print_every: int = 120):
    """
    Run the interactive CUDA OpenGL viewer loop.

    Parameters
    ----------
    sim
        Simulation instance.
    params
        Parameter object providing the lattice dimensions.
    steps_per_frame : int, optional
        Number of solver steps taken per rendered frame.
    fps_print_every : int, optional
        Number of frames between FPS printouts.

    Returns
    -------
    None
        The interactive viewer runs until the window is closed.

    Examples
    --------
    Use ``run_viewer(sim, sim.rp, steps_per_frame=5)`` to start the interactive viewer.
    """
    renderer = GLRenderer(params.xlen, params.ylen)
    renderer.bind_sim(sim)

    state = {"energy": None, "error": None, "epochs": 0}

    n = sim.en.size
    tpb = 1024
    blocks = (n + tpb - 1) // tpb
    state["min_partial"] = cuda.device_array(blocks, dtype=np.float64)
    state["max_partial"] = cuda.device_array(blocks, dtype=np.float64)

    try:
        while not renderer.should_close():
            renderer.begin_frame()

            if renderer.request_save_output:
                renderer.request_save_output = False
                if hasattr(sim, "save_output"):
                    sim.save_output(precision=32)
                else:
                    print("Warning: sim has no save_output(...) method.")

            advance_solver(sim, steps_per_frame, state)

            try:
                obs = sim.observables()
                state["energy"] = float(obs["energy"])
                if "skyrmion_number" in obs:
                    state["sk"] = float(obs["skyrmion_number"])
            except Exception:
                pass

            _compute_density_for_mode(sim, renderer.display_mode)

            vmin = compute_min(sim.en, state["min_partial"], sim.en.size)
            vmax = compute_max(sim.en, state["max_partial"], sim.en.size)
            if vmax <= vmin:
                vmax = vmin + 1e-30

            if renderer.display_mode==DISPLAY_ENERGY:
                use_jet = True
            else:
                use_jet = False

            dt = float(sim.p_f_h[5])
            epochs = int(state.get("epochs", 0))
            t = epochs * dt
            err = state.get("error", None)

            top_text = f"t={t:.1f} ({epochs} epochs)"
            if err is not None:
                top_text += f": err={float(err):.3e}"

            bottom_parts = []
            if state.get("energy") is not None:
                bottom_parts.append(f"E={float(state['energy']):.4f}")
            if state.get("sk") is not None:
                bottom_parts.append(f"Sk={float(state['sk']):.4f}")
            bottom_text = ", ".join(bottom_parts)

            renderer.set_hud_text(top=top_text, bottom=bottom_text)
            renderer.render(Field=sim.Field, density_flat=sim.en, xlen=params.xlen, ylen=params.ylen, p_i=sim.p_i_d, vmin=vmin, vmax=vmax, use_jet=use_jet)
            renderer.end_frame()

    finally:
        renderer.close()