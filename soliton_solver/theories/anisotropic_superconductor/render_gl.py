"""
CUDA OpenGL rendering for the anisotropic s+id superconductor theory.

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
from soliton_solver.theories.anisotropic_superconductor.initial_config import create_vortex_kernel
from soliton_solver.core.colormaps import render_jet_density_to_rgba, render_gray_density_to_rgba

DISPLAY_ENERGY = 1
DISPLAY_HIGGS1 = 2
DISPLAY_HIGGS2 = 3
DISPLAY_MAGNETIC_FLUX = 4
DISPLAY_PHASE_DIFF = 5

DISPLAY_TITLES = {
    DISPLAY_ENERGY: "En",
    DISPLAY_HIGGS1: "|psi1|^2",
    DISPLAY_HIGGS2: "|psi2|^2",
    DISPLAY_MAGNETIC_FLUX: "B",
    DISPLAY_PHASE_DIFF: "cos(theta12)",
}

class GLRenderer:
    """
    Interactive OpenGL renderer for the theory viewer.

    Examples
    --------
    Use ``renderer = GLRenderer(width, height)`` to create the viewer.
    """

    def __init__(self, width: int, height: int, title: str = "superferro (CUDA-OpenGL)"):
        """
        Create the renderer and the OpenGL backend.

        Parameters
        ----------
        width : int
            Viewport width.
        height : int
            Viewport height.
        title : str, optional
            Window title.

        Examples
        --------
        Use ``renderer = GLRenderer(params.xlen, params.ylen)`` to create the renderer.
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
        Attach a simulation instance to the renderer.

        Parameters
        ----------
        sim : object
            Simulation instance.

        Examples
        --------
        Use ``renderer.bind_sim(sim)`` to enable injection and preview.
        """
        self._sim = sim

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
            Simulation indices ``(pxi, pxj)``.

        Examples
        --------
        Use ``pxi, pxj = self._window_to_sim(x, y)`` to map the cursor to the lattice.
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
        window : object
            GLFW window handle.
        key : int
            GLFW key code.
        scancode : int
            GLFW scan code.
        action : int
            GLFW action code.
        mods : int
            GLFW modifier flags.

        Examples
        --------
        This callback is registered automatically during renderer initialization.
        """
        if action != glfw.PRESS:
            return

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(self.backend.window, True)
            return

        if key == glfw.KEY_F1:
            self.set_display_mode(DISPLAY_ENERGY)
        elif key == glfw.KEY_F2:
            self.set_display_mode(DISPLAY_HIGGS1)
        elif key == glfw.KEY_F3:
            self.set_display_mode(DISPLAY_HIGGS2)
        elif key == glfw.KEY_F4:
            self.set_display_mode(DISPLAY_MAGNETIC_FLUX)
        elif key == glfw.KEY_F5:
            self.set_display_mode(DISPLAY_PHASE_DIFF)

        elif key == glfw.KEY_V:
            self.soliton_mode = "vortex"

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

        elif key == glfw.KEY_O:
            self.request_save_output = True

    def _set_topological_number(self, n: float):
        """
        Update the skyrmion and vortex numbers in the simulation parameters.

        Parameters
        ----------
        n : float
            New topological number.

        Examples
        --------
        Use ``self._set_topological_number(n)`` to update the simulation parameter buffer.
        """
        sim = getattr(self, "_sim", None)
        if sim is None:
            return

        sim.p_f_h[21] = float(n)
        sim.p_f_h[22] = -sim.p_f_h[21] / sim.p_f_h[7]

        try:
            sim.p_f_d.copy_to_device(sim.p_f_h)
        except Exception:
            sim.p_f_d = cuda.to_device(sim.p_f_h)

    def _on_mouse_button(self, window, button, action, mods):
        """
        Handle mouse button input for soliton insertion.

        Parameters
        ----------
        window : object
            GLFW window handle.
        button : int
            GLFW mouse button code.
        action : int
            GLFW action code.
        mods : int
            GLFW modifier flags.

        Examples
        --------
        This callback is registered automatically during renderer initialization.
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
            create_vortex_kernel[grid2d, block2d](sim.Field, sim.grid, int(simX), int(simY), int(self.vortex_type), sim.p_i_d, sim.p_f_d)
            cuda.synchronize()
            return

    def set_display_mode(self, mode: int):
        """
        Set the active display mode.

        Parameters
        ----------
        mode : int
            Display mode constant.

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
        Update the window title for the active display mode.

        Examples
        --------
        This method is called automatically when the display mode changes.
        """
        self.backend.set_window_title(DISPLAY_TITLES.get(self.display_mode, "superferro"))

    def close(self):
        """
        Close the renderer and release the OpenGL resources.

        Examples
        --------
        Use ``renderer.close()`` to destroy the renderer.
        """
        self.backend.close()

    def should_close(self) -> bool:
        """
        Check whether the window should close.

        Returns
        -------
        bool
            ``True`` if the window should close.

        Examples
        --------
        Use ``renderer.should_close()`` in the viewer loop condition.
        """
        return self.backend.should_close()

    def begin_frame(self):
        """
        Begin a frame.

        Examples
        --------
        Use ``renderer.begin_frame()`` at the start of each viewer iteration.
        """
        self.backend.begin_frame()

    def end_frame(self):
        """
        End a frame.

        Examples
        --------
        Use ``renderer.end_frame()`` at the end of each viewer iteration.
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
            Simulation field buffer.
        density_flat : device array
            Scalar density buffer.
        xlen : int
            Lattice size along the x direction.
        ylen : int
            Lattice size along the y direction.
        p_i : device array
            Integer parameter array.
        vmin : float
            Lower colormap bound.
        vmax : float
            Upper colormap bound.
        use_jet : bool
            Whether to use the jet colormap for scalar data.

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
    Advance the solver for one rendered frame.

    Parameters
    ----------
    sim : object
        Simulation instance.
    steps_per_frame : int
        Number of solver steps per frame.
    state : dict
        Mutable state dictionary.

    Examples
    --------
    Use ``advance_solver(sim, steps_per_frame, state)`` inside the viewer loop.
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
    sim : object
        Simulation instance.
    mode : int
        Display mode constant.

    Examples
    --------
    Use ``_compute_density_for_mode(sim, mode)`` before rendering a scalar field.
    """
    if mode == DISPLAY_ENERGY:
        if hasattr(sim, "compute_energy_density"):
            sim.compute_energy_density()
            return
    if mode == DISPLAY_HIGGS1:
        if hasattr(sim, "compute_higgs1_density"):
            sim.compute_higgs1_density()
            return
    if mode == DISPLAY_HIGGS2:
        if hasattr(sim, "compute_higgs2_density"):
            sim.compute_higgs2_density()
            return
    if mode == DISPLAY_MAGNETIC_FLUX:
        if hasattr(sim, "compute_magnetic_flux_density"):
            sim.compute_magnetic_flux_density(which=2)
            return
    if mode == DISPLAY_PHASE_DIFF:
        if hasattr(sim, "compute_phase_difference_density"):
            sim.compute_phase_difference_density()
            return

def run_viewer(sim, params, *, steps_per_frame: int = 5, fps_print_every: int = 120):
    """
    Run the interactive viewer loop.

    Parameters
    ----------
    sim : object
        Simulation instance.
    params : object
        Parameter object containing the lattice dimensions.
    steps_per_frame : int, optional
        Number of solver steps per frame.
    fps_print_every : int, optional
        Number of frames between FPS prints.

    Examples
    --------
    Use ``run_viewer(sim, params, steps_per_frame=5)`` to start the interactive viewer.
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

            if renderer.display_mode in (DISPLAY_ENERGY, DISPLAY_HIGGS1, DISPLAY_HIGGS2, DISPLAY_PHASE_DIFF):
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
            bottom_text = ", ".join(bottom_parts)

            renderer.set_hud_text(top=top_text, bottom=bottom_text)
            renderer.render(Field=sim.Field, density_flat=sim.en, xlen=params.xlen, ylen=params.ylen, p_i=sim.p_i_d, vmin=vmin, vmax=vmax, use_jet=use_jet)
            renderer.end_frame()

    finally:
        renderer.close()