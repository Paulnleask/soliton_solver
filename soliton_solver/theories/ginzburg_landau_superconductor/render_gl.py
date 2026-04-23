"""
Render the Ginzburg-Landau superconductor simulation with the OpenGL viewer.

Examples
--------
Use ``run_viewer(sim, params)`` to start the interactive viewer.
"""
from __future__ import annotations
import numpy as np
import glfw
from numba import cuda
from soliton_solver.visualization.gl_backend import GLBackend, cuda_array_from_ptr
from soliton_solver.core.utils import compute_min, compute_max
from soliton_solver.core.utils import launch_2d
from soliton_solver.theories.ginzburg_landau_superconductor.initial_config import create_vortex_kernel
from soliton_solver.core.colormaps import render_jet_density_to_rgba, render_gray_density_to_rgba

DISPLAY_ENERGY = 1
DISPLAY_HIGGS = 2
DISPLAY_MAGNETIC_FLUX = 3

DISPLAY_TITLES = {
    DISPLAY_ENERGY: "En",
    DISPLAY_HIGGS: "|psi|^2",
    DISPLAY_MAGNETIC_FLUX: "B"
}

class GLRenderer:
    """
    Manage interactive rendering for the Ginzburg-Landau superconductor simulation.

    Examples
    --------
    Use ``renderer = GLRenderer(width, height)`` to create the viewer.
    """

    def __init__(self, width: int, height: int, title: str = "ginzburg_landau_superconductor (CUDA-OpenGL)"):
        """
        Create the renderer and its OpenGL backend.

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
            The renderer is initialized in place.

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

        self.vortex_type = 0
        self.request_save_output = False

        glfw.set_key_callback(self.backend.window, self._on_key)
        glfw.set_mouse_button_callback(self.backend.window, self._on_mouse_button)

        self._update_window_title()

    def bind_sim(self, sim):
        """
        Bind a simulation instance to the renderer.

        Parameters
        ----------
        sim
            Simulation instance used for rendering and interaction.

        Returns
        -------
        None
            The simulation reference is stored on the renderer.

        Examples
        --------
        Use ``renderer.bind_sim(sim)`` to attach a simulation to the viewer.
        """
        self._sim = sim

    def _window_to_sim(self, x, y):
        """
        Convert window coordinates to simulation lattice indices.

        Parameters
        ----------
        x : float
            Window x coordinate in pixels.
        y : float
            Window y coordinate in pixels.

        Returns
        -------
        tuple
            Pair ``(sim_x, sim_y)`` of lattice indices.

        Examples
        --------
        Use ``sim_x, sim_y = self._window_to_sim(x, y)`` to map cursor position to the lattice.
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

    def _on_key(self, window, key, scancode, action, mods):
        """
        Handle keyboard input.

        Parameters
        ----------
        window
            GLFW window handle.
        key : int
            GLFW key code.
        scancode : int
            Platform-specific scan code.
        action : int
            GLFW key action.
        mods : int
            GLFW modifier flags.

        Returns
        -------
        None
            Renderer state is updated in place.

        Examples
        --------
        This method is used as the GLFW key callback.
        """
        if action != glfw.PRESS:
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(self.backend.window, True)
            return

        if key == glfw.KEY_F1:
            self.set_display_mode(DISPLAY_ENERGY)
        elif key == glfw.KEY_F2:
            self.set_display_mode(DISPLAY_HIGGS)
        elif key == glfw.KEY_F3:
            self.set_display_mode(DISPLAY_MAGNETIC_FLUX)

        elif key == glfw.KEY_V:
            self.soliton_mode = "vortex"

        elif glfw.KEY_1 <= key <= glfw.KEY_9:
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

        elif key == glfw.KEY_O:
            self.request_save_output = True

    def _set_topological_number(self, n: float):
        """
        Update the vortex number in the simulation parameters.

        Parameters
        ----------
        n : float
            New vortex number.

        Returns
        -------
        None
            The device parameter array is updated in place.

        Examples
        --------
        Use ``self._set_topological_number(2.0)`` to set the vortex number to two.
        """
        sim = getattr(self, "_sim", None)
        if sim is None:
            return

        sim.p_f_h[9] = float(n)

        try:
            sim.p_f_d.copy_to_device(sim.p_f_h)
        except Exception:
            sim.p_f_d = cuda.to_device(sim.p_f_h)

    def _on_mouse_button(self, window, button, action, mods):
        """
        Handle mouse button input.

        Parameters
        ----------
        window
            GLFW window handle.
        button : int
            GLFW mouse button code.
        action : int
            GLFW mouse action.
        mods : int
            GLFW modifier flags.

        Returns
        -------
        None
            Simulation fields may be updated in place.

        Examples
        --------
        This method is used as the GLFW mouse callback.
        """
        if self._sim is None:
            return

        if button != glfw.MOUSE_BUTTON_LEFT:
            return

        sim = self._sim

        if self.soliton_mode == "vortex" and action == glfw.PRESS:
            x, y = glfw.get_cursor_pos(self.backend.window)
            simX, simY = self._window_to_sim(x, y)
            grid2d, block2d = launch_2d(sim.p_i_h, threads=(8, 8))
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
            The display mode and window title are updated in place.

        Examples
        --------
        Use ``renderer.set_display_mode(DISPLAY_HIGGS)`` to show the Higgs density.
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
            The window title is updated in place.

        Examples
        --------
        Use ``self._update_window_title()`` after changing the display mode.
        """
        self.backend.set_window_title(DISPLAY_TITLES.get(self.display_mode, "ginzburg_landau_superconductor"))

    def close(self):
        """
        Close the renderer and release OpenGL resources.

        Returns
        -------
        None
            The backend is closed.

        Examples
        --------
        Use ``renderer.close()`` to shut down the viewer.
        """
        self.backend.close()

    def should_close(self) -> bool:
        """
        Check whether the window should close.

        Returns
        -------
        bool
            True if the window should close.

        Examples
        --------
        Use ``while not renderer.should_close():`` to run the viewer loop.
        """
        return self.backend.should_close()

    def begin_frame(self):
        """
        Begin a render frame.

        Returns
        -------
        None
            Window events are polled.

        Examples
        --------
        Use ``renderer.begin_frame()`` at the start of each frame.
        """
        self.backend.begin_frame()

    def end_frame(self):
        """
        End a render frame.

        Returns
        -------
        None
            Buffers are swapped for the current frame.

        Examples
        --------
        Use ``renderer.end_frame()`` at the end of each frame.
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
            The HUD text is updated in place.

        Examples
        --------
        Use ``renderer.set_hud_text(top="t=1.0", bottom="E=0.1")`` to update the HUD.
        """
        self.backend.set_hud_text(top=top, bottom=bottom)

    def render(self, *, Field, density_flat, xlen: int, ylen: int, p_i, vmin: float, vmax: float, use_jet: bool):
        """
        Render one frame.

        Parameters
        ----------
        Field
            Simulation field array.
        density_flat
            Flattened scalar field to visualize.
        xlen : int
            Number of lattice points in the x direction.
        ylen : int
            Number of lattice points in the y direction.
        p_i
            Device integer parameter array.
        vmin : float
            Lower bound of the scalar color range.
        vmax : float
            Upper bound of the scalar color range.
        use_jet : bool
            Flag selecting the jet colormap instead of grayscale.

        Returns
        -------
        None
            The frame is rendered to the current OpenGL window.

        Examples
        --------
        Use ``renderer.render(Field=sim.Field, density_flat=sim.en, xlen=params.xlen, ylen=params.ylen, p_i=sim.p_i_d, vmin=vmin, vmax=vmax, use_jet=True)`` to draw a frame.
        """
        mapped = self.backend.map_pbo()
        try:
            pbo_view = cuda_array_from_ptr(mapped.ptr, (self.height, self.width, 4), np.uint8)

            grid2d, block2d = launch_2d(p_i, threads=(8, 8))

            if use_jet:
                render_jet_density_to_rgba[grid2d, block2d](pbo_view, density_flat, xlen, ylen, float(vmin), float(vmax))
            else:
                render_gray_density_to_rgba[grid2d, block2d](pbo_view, density_flat, xlen, ylen, float(vmin), float(vmax))

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
        Number of solver steps to take.
    state : dict
        Mutable state dictionary storing energy, error, and epoch counters.

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
    Compute the scalar density for the active display mode.

    Parameters
    ----------
    sim
        Simulation instance.
    mode : int
        Display mode identifier.

    Returns
    -------
    None
        The density field is written into ``sim.en``.

    Examples
    --------
    Use ``_compute_density_for_mode(sim, renderer.display_mode)`` before rendering a frame.
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
    sim
        Simulation instance.
    params
        Parameter object containing the lattice dimensions.
    steps_per_frame : int, optional
        Number of solver steps per rendered frame.
    fps_print_every : int, optional
        Number of frames between FPS prints.

    Returns
    -------
    None
        The viewer loop runs until the window is closed.

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
            if state.get("vort") is not None:
                bottom_parts.append(f"V={float(state['vort']):.4f}")
            if state.get("rho") is not None:
                bottom_parts.append(f"rho={float(state['rho']):.4f}")
            bottom_text = ", ".join(bottom_parts)

            renderer.set_hud_text(top=top_text, bottom=bottom_text)
            renderer.render(Field=sim.Field, density_flat=sim.en, xlen=params.xlen, ylen=params.ylen, p_i=sim.p_i_d, vmin=vmin, vmax=vmax, use_jet=use_jet)
            renderer.end_frame()

    finally:
        renderer.close()