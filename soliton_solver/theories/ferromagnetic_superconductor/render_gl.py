"""
CUDA-OpenGL rendering for the ferromagnetic superconductor theory.

Examples
--------
Use ``run_viewer`` to launch the interactive viewer.
"""
from __future__ import annotations
import math
import numpy as np
import glfw
from numba import cuda
from soliton_solver.visualization.gl_backend import GLBackend, cuda_array_from_ptr
from soliton_solver.core.utils import compute_min, compute_max
from soliton_solver.core.utils import launch_2d
from soliton_solver.theories.ferromagnetic_superconductor.initial_config import create_skyrmion_kernel, create_vortex_kernel
from soliton_solver.core.colormaps import clip_u8, _hsv_to_rgb, render_jet_density_to_rgba, render_gray_density_to_rgba, render_magnetization_to_rgba

DISPLAY_ENERGY = 1
DISPLAY_MAGNETIZATION = 2
DISPLAY_HIGGS = 3
DISPLAY_MAGNETIC_FLUX = 4

DISPLAY_TITLES = {
    DISPLAY_ENERGY: "En",
    DISPLAY_MAGNETIZATION: "Sk",
    DISPLAY_HIGGS: "|psi|^2",
    DISPLAY_MAGNETIC_FLUX: "B",
}

@cuda.jit
def overlay_skyrmion_preview_kernel(pbo_rgba, grid, pxi, pxj, rotation_angle, ansatz_bloch, ansatz_neel, ansatz_anti, p_i, p_f):
    """
    Blend a skyrmion preview into the RGBA output buffer.

    Parameters
    ----------
    pbo_rgba : device array
        Mapped RGBA pixel buffer.
    grid : device array
        Coordinate grid.
    pxi : int
        Preview center index along the x direction.
    pxj : int
        Preview center index along the y direction.
    rotation_angle : float
        Rotation angle for the preview.
    ansatz_bloch : bool
        Flag selecting the Bloch ansatz.
    ansatz_neel : bool
        Flag selecting the Néel ansatz.
    ansatz_anti : bool
        Flag selecting the anti-skyrmion ansatz.
    p_i : device array
        Integer parameter array.
    p_f : device array
        Float parameter array.

    Returns
    -------
    None
        The preview is blended into ``pbo_rgba`` in place.

    Examples
    --------
    Launch ``overlay_skyrmion_preview_kernel[grid2d, block2d](...)`` to draw the preview overlay.
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
    skN = p_f[18]
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
    Interactive OpenGL renderer for the ferromagnetic superconductor theory.

    Parameters
    ----------
    width : int
        Window and viewport width in pixels.
    height : int
        Window and viewport height in pixels.
    title : str, optional
        Window title.

    Examples
    --------
    Use ``renderer = GLRenderer(width, height)`` to create the renderer.
    """

    def __init__(self, width: int, height: int, title: str = "ferromagnetic_superconductor (CUDA-OpenGL)"):
        """
        Create the renderer and its backend.

        Parameters
        ----------
        width : int
            Window and viewport width in pixels.
        height : int
            Window and viewport height in pixels.
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
        Bind a simulation instance to the renderer.

        Parameters
        ----------
        sim : object
            Simulation instance providing fields, parameters, and buffers.

        Returns
        -------
        None
            The simulation is attached to the renderer.

        Examples
        --------
        Use ``renderer.bind_sim(sim)`` to enable interactive editing and previews.
        """
        self._sim = sim
        try:
            self.skyrmion_rotation = float(sim.p_f_h[19])
        except Exception:
            self.skyrmion_rotation = 0.0

    def _window_to_sim(self, x, y):
        """
        Convert window coordinates to simulation indices.

        Parameters
        ----------
        x : float
            Window x coordinate.
        y : float
            Window y coordinate.

        Returns
        -------
        tuple
            Pair ``(sim_x, sim_y)`` of simulation indices.

        Examples
        --------
        Use ``sim_x, sim_y = self._window_to_sim(x, y)`` in mouse handlers.
        """
        sim = self._sim
        if sim is None:
            return 0, 0

        win_w, win_h = glfw.get_window_size(self.backend.window)
        if win_w < 1:
            win_w = 1
        if win_h < 1:
            win_h = 1

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
            Input angle in radians.

        Returns
        -------
        float
            Wrapped angle.

        Examples
        --------
        Use ``a = self._wrap2pi(a)`` to keep interactive rotations bounded.
        """
        twopi = 2.0 * math.pi
        a = math.fmod(a, twopi)
        if a < 0.0:
            a += twopi
        return a

    def _on_key(self, window, key, scancode, action, mods):
        """
        Handle keyboard input.

        Parameters
        ----------
        window : object
            GLFW window handle.
        key : int
            GLFW key code.
        scancode : int
            Platform scancode.
        action : int
            GLFW key action.
        mods : int
            Modifier mask.

        Returns
        -------
        None
            Renderer state is updated in response to the key event.

        Examples
        --------
        This method is registered as the GLFW key callback.
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
        elif key == glfw.KEY_F3:
            self.set_display_mode(DISPLAY_HIGGS)
        elif key == glfw.KEY_F4:
            self.set_display_mode(DISPLAY_MAGNETIC_FLUX)

        elif key == glfw.KEY_S:
            self.soliton_mode = "skyrmion"
            self.preview_active = True
            self.iso_rotating = False
        elif key == glfw.KEY_V:
            self.soliton_mode = "vortex"
            self.preview_active = False
            self.iso_rotating = False

        elif glfw.KEY_0 <= key <= glfw.KEY_9:
            n = float(key - glfw.KEY_0)
            self._set_topological_number(n)

        elif key == glfw.KEY_A:
            self.vortex_type = 1 - int(self.vortex_type)

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
        Update the topological numbers in the simulation parameters.

        Parameters
        ----------
        n : float
            New topological number.

        Returns
        -------
        None
            The device parameter array is updated in place.

        Examples
        --------
        Use ``self._set_topological_number(2.0)`` to update the skyrmion and vortex numbers.
        """
        sim = getattr(self, "_sim", None)
        if sim is None:
            return

        sim.p_f_h[18] = float(n)
        sim.p_f_h[12] = float(n)

        try:
            sim.p_f_d.copy_to_device(sim.p_f_h)
        except Exception:
            sim.p_f_d = cuda.to_device(sim.p_f_h)

    def _on_cursor_pos(self, window, xpos, ypos):
        """
        Handle mouse movement for previews and skyrmion rotation.

        Parameters
        ----------
        window : object
            GLFW window handle.
        xpos : float
            Cursor x coordinate.
        ypos : float
            Cursor y coordinate.

        Returns
        -------
        None
            Preview state or rotation state is updated.

        Examples
        --------
        This method is registered as the GLFW cursor callback.
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
        Handle mouse clicks for skyrmion and vortex insertion.

        Parameters
        ----------
        window : object
            GLFW window handle.
        button : int
            GLFW mouse button code.
        action : int
            GLFW button action.
        mods : int
            Modifier mask.

        Returns
        -------
        None
            The simulation field may be modified in response to the mouse event.

        Examples
        --------
        This method is registered as the GLFW mouse button callback.
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
                ans_bloch = bool(p_f[20] > 0.5)
                ans_neel = bool(p_f[21] > 0.5)
                ans_anti = bool(p_f[22] > 0.5)

                grid2d, block2d = launch_2d(sim.p_i_h, threads=(16, 32))
                create_skyrmion_kernel[grid2d, block2d](
                    sim.Field, sim.grid,
                    int(self.preview_pxi), int(self.preview_pxj),
                    float(self.skyrmion_rotation),
                    ans_bloch, ans_neel, ans_anti,
                    sim.p_i_d, sim.p_f_d
                )
                cuda.synchronize()
                return

        if self.soliton_mode == "vortex" and action == glfw.PRESS:
            x, y = glfw.get_cursor_pos(self.backend.window)
            simX, simY = self._window_to_sim(x, y)
            grid2d, block2d = launch_2d(sim.p_i_h, threads=(16, 32))
            create_vortex_kernel[grid2d, block2d](
                sim.Field, sim.grid, int(simX), int(simY), int(self.vortex_type), sim.p_i_d, sim.p_f_d
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
            The display mode and window title are updated.

        Examples
        --------
        Use ``renderer.set_display_mode(DISPLAY_MAGNETIZATION)`` to change the view.
        """
        if mode not in DISPLAY_TITLES:
            return
        self.display_mode = mode
        self._update_window_title()

    def _update_window_title(self):
        """
        Update the window title for the active display mode.

        Returns
        -------
        None
            The backend window title is updated.

        Examples
        --------
        This method is called internally after display mode changes.
        """
        self.backend.set_window_title(DISPLAY_TITLES.get(self.display_mode, "ferromagnetic_superconductor"))

    def close(self):
        """
        Close the renderer and release backend resources.

        Returns
        -------
        None
            Window and OpenGL resources are released.

        Examples
        --------
        Use ``renderer.close()`` when shutting down the viewer.
        """
        self.backend.close()

    def should_close(self) -> bool:
        """
        Check whether the window should close.

        Returns
        -------
        bool
            True if the window has been marked for closing.

        Examples
        --------
        Use ``while not renderer.should_close():`` in the viewer loop.
        """
        return self.backend.should_close()

    def begin_frame(self):
        """
        Begin a new frame.

        Returns
        -------
        None
            Input events are polled for the current frame.

        Examples
        --------
        Use ``renderer.begin_frame()`` at the start of each viewer frame.
        """
        self.backend.begin_frame()

    def end_frame(self):
        """
        End the current frame.

        Returns
        -------
        None
            The frame is presented to the screen.

        Examples
        --------
        Use ``renderer.end_frame()`` at the end of each viewer frame.
        """
        self.backend.end_frame()

    def set_hud_text(self, *, top: str = "", bottom: str = ""):
        """
        Set the HUD text.

        Parameters
        ----------
        top : str, optional
            Top HUD line.
        bottom : str, optional
            Bottom HUD line.

        Returns
        -------
        None
            The HUD text is stored in the backend.

        Examples
        --------
        Use ``renderer.set_hud_text(top="...", bottom="...")`` to update the HUD.
        """
        self.backend.set_hud_text(top=top, bottom=bottom)

    def render(self, *, Field, density_flat, xlen: int, ylen: int, p_i, vmin: float, vmax: float, use_jet: bool):
        """
        Render one frame.

        Parameters
        ----------
        Field : device array
            Simulation field array.
        density_flat : device array
            Scalar density buffer used for scalar display modes.
        xlen : int
            Number of lattice points along the x direction.
        ylen : int
            Number of lattice points along the y direction.
        p_i : device array
            Integer parameter array.
        vmin : float
            Lower colormap bound.
        vmax : float
            Upper colormap bound.
        use_jet : bool
            Whether to use the jet colormap for scalar rendering.

        Returns
        -------
        None
            The rendered frame is drawn to the window.

        Examples
        --------
        Use ``renderer.render(Field=sim.Field, density_flat=sim.en, xlen=params.xlen, ylen=params.ylen, p_i=sim.p_i_d, vmin=vmin, vmax=vmax, use_jet=True)`` to draw a frame.
        """
        mapped = self.backend.map_pbo()
        try:
            pbo_view = cuda_array_from_ptr(mapped.ptr, (self.height, self.width, 4), np.uint8)

            grid2d, block2d = launch_2d(p_i, threads=(16, 32))

            if self.display_mode == DISPLAY_MAGNETIZATION:
                render_magnetization_to_rgba[grid2d, block2d](pbo_view, Field, xlen, ylen, p_i)
            else:
                if use_jet:
                    render_jet_density_to_rgba[grid2d, block2d](pbo_view, density_flat, xlen, ylen, float(vmin), float(vmax))
                else:
                    render_gray_density_to_rgba[grid2d, block2d](pbo_view, density_flat, xlen, ylen, float(vmin), float(vmax))

            if self.preview_active and self.soliton_mode == "skyrmion" and self._sim is not None:
                p_f = self._sim.p_f_h
                ans_bloch = bool(p_f[20] > 0.5)
                ans_neel = bool(p_f[21] > 0.5)
                ans_anti = bool(p_f[22] > 0.5)
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
    Advance the simulation by a fixed number of steps.

    Parameters
    ----------
    sim : object
        Simulation instance.
    steps_per_frame : int
        Number of solver steps per rendered frame.
    state : dict
        Mutable state dictionary holding energy, error, and epoch counters.

    Returns
    -------
    None
        The simulation and state dictionary are updated in place.

    Examples
    --------
    Use ``advance_solver(sim, steps_per_frame, state)`` once per frame in the viewer loop.
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
    Compute the active scalar density for the selected display mode.

    Parameters
    ----------
    sim : object
        Simulation instance.
    mode : int
        Display mode identifier.

    Returns
    -------
    None
        The active scalar density is written into ``sim.en`` when needed.

    Examples
    --------
    Use ``_compute_density_for_mode(sim, mode)`` before rendering a scalar display.
    """
    if mode == DISPLAY_ENERGY:
        if hasattr(sim, "compute_energy_density"):
            sim.compute_energy_density()
            return
    if mode == DISPLAY_HIGGS:
        if hasattr(sim, "compute_higgs_density"):
            sim.compute_higgs_density()
            return
    if mode == DISPLAY_MAGNETIC_FLUX:
        if hasattr(sim, "compute_magnetic_flux_density"):
            sim.compute_magnetic_flux_density(which=2)
            return

def run_viewer(sim, params, *, steps_per_frame: int = 5, fps_print_every: int = 120):
    """
    Run the interactive viewer loop.

    Parameters
    ----------
    sim : object
        Simulation instance.
    params : object
        Parameter object providing the lattice dimensions.
    steps_per_frame : int, optional
        Number of solver steps per rendered frame.
    fps_print_every : int, optional
        Frame interval used for FPS reporting.

    Returns
    -------
    None
        The interactive viewer runs until the window closes.

    Examples
    --------
    Use ``run_viewer(sim, sim.rp, steps_per_frame=5)`` to launch the viewer.
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
                if "vortex_number" in obs:
                    state["vort"] = float(obs["vortex_number"])
            except Exception:
                pass

            _compute_density_for_mode(sim, renderer.display_mode)

            vmin = compute_min(sim.en, state["min_partial"], sim.en.size)
            vmax = compute_max(sim.en, state["max_partial"], sim.en.size)
            if vmax <= vmin:
                vmax = vmin + 1e-30

            if renderer.display_mode in (DISPLAY_ENERGY, DISPLAY_HIGGS):
                use_jet = True
            elif renderer.display_mode == DISPLAY_MAGNETIC_FLUX:
                use_jet = False
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
            if state.get("vort") is not None:
                bottom_parts.append(f"V={float(state['vort']):.4f}")
            bottom_text = ", ".join(bottom_parts)

            renderer.set_hud_text(top=top_text, bottom=bottom_text)
            renderer.render(Field=sim.Field, density_flat=sim.en, xlen=params.xlen, ylen=params.ylen, p_i=sim.p_i_d, vmin=vmin, vmax=vmax, use_jet=use_jet)
            renderer.end_frame()

    finally:
        renderer.close()