# =========================
# soliton_solver/theories/maxwell_chern_simons_higgs/render_gl.py
# =========================
"""
SuperFerro-specific CUDA-OpenGL renderer logic.

Usage:
- This module provides the interactive viewer layer on top of the generic GL backend:
    * chooses which scalar density to compute (energy / |psi|^2 / flux) per display mode
    * renders either a scalar density (jet/gray colormaps) or magnetization (HSV)
    * supports interactive soliton injection:
        - skyrmion placement + rotation via mouse (preview overlay + commit on release)
        - vortex placement via mouse click
    * draws a small HUD with time/epochs/err and observables

Public API:
- GLRenderer: interactive renderer/controller
- run_viewer(sim, params, ...): convenience main loop for interactive viewing

Notes:
- GLBackend handles: window creation, PBO/texture setup, CUDA-GL mapping, and HUD drawing.
- This file handles: simulation-specific content, callbacks, and which kernels to run.
"""

# ---------------- Imports ----------------
from __future__ import annotations
import math
import time
import numpy as np
import glfw
from numba import cuda
from soliton_solver.visualization.gl_backend import GLBackend, cuda_array_from_ptr
from soliton_solver.core.utils import compute_min, compute_max
from soliton_solver.core.utils import launch_2d
from soliton_solver.theories.maxwell_chern_simons_higgs.initial_config import create_vortex_kernel
from soliton_solver.core.colormaps import render_jet_density_to_rgba, render_gray_density_to_rgba

# ---------------- Display modes ----------------
DISPLAY_ENERGY = 1
DISPLAY_HIGGS = 2
DISPLAY_MAGNETIC_FLUX = 3
DISPLAY_ELECTRIC_CHARGE = 4
DISPLAY_NOETHER_CHARGE = 5

DISPLAY_TITLES = {
    DISPLAY_ENERGY: "En",
    DISPLAY_HIGGS: "|psi|^2",
    DISPLAY_MAGNETIC_FLUX: "B",
    DISPLAY_ELECTRIC_CHARGE: "Rho_e",
    DISPLAY_NOETHER_CHARGE: "Rho_m"
}

# ---------------- OpenGL renderer ----------------
class GLRenderer:
    """
    SuperFerro viewer front-end.

    Usage:
    - renderer = GLRenderer(width=xlen, height=ylen)
    - renderer.bind_sim(sim)   # enables injection + preview based on sim params
    - In a loop:
        renderer.begin_frame()
        ... compute density into sim.en ...
        renderer.set_hud_text(...)
        renderer.render(Field=sim.Field, density_flat=sim.en, ...)
        renderer.end_frame()

    Responsibilities:
    - Manages display mode, UI state, and input callbacks.
    - Delegates all windowing and CUDA-GL interop to GLBackend.
    - Launches colormap kernels into the mapped PBO.
    - Optionally overlays a skyrmion preview and/or injects solitons into the sim on clicks.
    """

    def __init__(self, width: int, height: int, title: str = "maxwell_chern_simons_higgs (CUDA-OpenGL)"):
        """
        Create a renderer and its underlying GLFW/OpenGL backend.

        Usage:
        - renderer = GLRenderer(params.xlen, params.ylen)
        - width/height are used both for:
            * the GLFW window size / OpenGL texture resolution
            * the CUDA render kernel launch bounds (xlen/ylen)

        Parameters:
        - width, height: viewport dimensions in pixels
        - title: window title
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

        # callbacks
        glfw.set_key_callback(self.backend.window, self._on_key)
        glfw.set_mouse_button_callback(self.backend.window, self._on_mouse_button)

        self._update_window_title()

    def bind_sim(self, sim):
        self._sim = sim

    # ---------------- Window → simulation pixel mapping ----------------
    def _window_to_sim(self, x, y):
        """
        Convert a GLFW window pixel coordinate into a simulation lattice index.

        Usage:
        - Called by mouse handlers to map the cursor location to (pxi, pxj) in the sim grid.

        Notes:
        - Uses current window size (not framebuffer size) and flips y so that window origin
          matches the simulation’s expected orientation.
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

    # ---------------- Keyboard callback ----------------
    def _on_key(self, window, key, scancode, action, mods):
        """
        GLFW key handler.

        Usage / bindings:
        - ESC: close window
        - F1..F4: switch display mode (energy / magnetization / |psi|^2 / flux)
        - S: enter skyrmion placement mode (shows preview; click/drag rotates)
        - V: enter vortex placement mode (click to insert)
        - 1..9: set skyrmion_number and vortex_number (writes to sim.p_f)
        - A: toggle vortex type/sign (0/1) for insertion
        - N: toggle newtonflow flag in the simulation params
        - K: toggle killkinen flag in the simulation params
        - Q: exit soliton mode (no preview/insertion)
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
        elif key == glfw.KEY_F4:
            self.set_display_mode(DISPLAY_ELECTRIC_CHARGE)
        elif key == glfw.KEY_F5:
            self.set_display_mode(DISPLAY_NOETHER_CHARGE)

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
        Update skyrmion_number and vortex_number in the bound Simulation (in-place).

        Usage:
        - Called by key handler when pressing number keys 1..9.
        - Writes:
            sim.p_f_h[10] = n  (vortex_number)
          and uploads p_f to the device.

        Note:
        - This updates only the device parameter array; it does not rebuild Simulation buffers.
        """
        sim = getattr(self, "_sim", None)
        if sim is None:
            return

        sim.p_f_h[10] = float(n)  # vortex_number

        try:
            sim.p_f_d.copy_to_device(sim.p_f_h)
        except Exception:
            sim.p_f_d = cuda.to_device(sim.p_f_h)

    def _on_mouse_button(self, window, button, action, mods):
        """
        GLFW mouse-button handler for committing soliton insertions.

        Usage:
        - Left click behavior depends on soliton_mode:
            * "skyrmion":
                - PRESS: start rotation + lock preview center
                - RELEASE: launch create_skyrmion_kernel to write into sim.Field
            * "vortex":
                - PRESS: launch create_vortex_kernel at cursor position
        """
        if self._sim is None:
            return

        if button != glfw.MOUSE_BUTTON_LEFT:
            return

        sim = self._sim

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
        Set the current display mode and update the window title.

        Usage:
        - renderer.set_display_mode(DISPLAY_MAGNETIZATION)
        """
        if mode not in DISPLAY_TITLES:
            return
        self.display_mode = mode
        self._update_window_title()

    def _update_window_title(self):
        """
        Update the GLFW window title to match the active display mode.
        """
        self.backend.set_window_title(DISPLAY_TITLES.get(self.display_mode, "maxwell_chern_simons_higgs"))

    def close(self):
        """
        Destroy GL resources and close the window.
        """
        self.backend.close()

    def should_close(self) -> bool:
        """
        Convenience passthrough to GLFW window_should_close.
        """
        return self.backend.should_close()

    def begin_frame(self):
        """
        Poll events at the beginning of a frame.
        """
        self.backend.begin_frame()

    def end_frame(self):
        """
        Swap buffers at the end of a frame.
        """
        self.backend.end_frame()

    def set_hud_text(self, *, top: str = "", bottom: str = ""):
        """
        Set HUD text lines (top/bottom).

        Usage:
        - renderer.set_hud_text(top="...", bottom="...")
        """
        self.backend.set_hud_text(top=top, bottom=bottom)

    def render(self, *, Field, density_flat, xlen: int, ylen: int, p_i, vmin: float, vmax: float, use_jet: bool):
        """
        Render one frame into the OpenGL window.

        Usage:
        - Call after you have computed the appropriate scalar field into density_flat (typically sim.en):
            renderer.render(Field=sim.Field, density_flat=sim.en, xlen=..., ylen=..., p_i=sim.p_i_d, vmin=..., vmax=..., use_jet=True)

        Rendering pipeline:
        1) Map the OpenGL PBO into CUDA address space (backend.map_pbo)
        2) Wrap the device pointer as a uint8(H,W,4) array view
        3) Launch one of:
            - render_magnetization_to_rgba (HSV)
            - render_jet_density_to_rgba or render_gray_density_to_rgba (scalar densities)
        4) Optionally overlay a skyrmion preview into the same PBO
        5) Unmap PBO and have backend upload it to a texture + draw fullscreen quad

        Parameters:
        - Field: simulation Field (needed for magnetization HSV mode)
        - density_flat: scalar density to visualize for non-magnetization modes
        - xlen, ylen: simulation lattice size (bounds for CUDA kernels)
        - p_i: device parameter array (passed to magnetization renderer)
        - vmin, vmax: scalar range used to normalize density -> colormap
        - use_jet: selects jet vs grayscale for scalar displays
        """
        mapped = self.backend.map_pbo()
        try:
            pbo_view = cuda_array_from_ptr(mapped.ptr, (self.height, self.width, 4), np.uint8)

            grid2d, block2d = launch_2d(p_i, threads=(16, 32))

            if use_jet:
                    render_jet_density_to_rgba[grid2d, block2d](pbo_view, density_flat, xlen, ylen, float(vmin), float(vmax))
            else:
                render_gray_density_to_rgba[grid2d, block2d](pbo_view, density_flat, xlen, ylen, float(vmin), float(vmax))

            cuda.synchronize()

        finally:
            self.backend.unmap_pbo()

        self.backend.upload_and_draw()

# ---------------- Simulation stepping adapter ----------------
def advance_solver(sim, steps_per_frame: int, state: dict):
    """
    Advance the simulation by a fixed number of solver steps per rendered frame.

    Usage:
    - Called once per frame from run_viewer().
    - Updates the mutable `state` dict in-place:
        state["energy"], state["error"], state["epochs"]

    Parameters:
    - sim: Simulation instance
    - steps_per_frame: number of sim.step(...) calls per frame
    - state: dict holding running values (energy/error/epochs)
    """
    if state.get("energy") is None:
        obs = sim.observables()
        state["energy"] = float(obs["energy"])
    if sim.rp.newtonflow:
        for _ in range(steps_per_frame):
            state["energy"], state["error"] = sim.step(state["energy"])
            state["epochs"] += 1

# ---------------- Multi-display density selection ----------------
def _compute_density_for_mode(sim, mode: int):
    """
    Ensure sim.en contains the correct scalar density for the current display mode.

    Usage:
    - Called each frame by run_viewer() before computing vmin/vmax and rendering.

    Behavior:
    - DISPLAY_ENERGY: sim.compute_energy_density() -> sim.en
    - DISPLAY_HIGGS: sim.compute_higgs_density() -> sim.en
    - DISPLAY_MAGNETIC_FLUX: sim.compute_magnetic_flux_density(which=2) -> sim.en
    - DISPLAY_MAGNETIZATION: does nothing here (render uses Field directly)
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
    if mode == DISPLAY_ELECTRIC_CHARGE:
        if hasattr(sim, "compute_electric_charge_density"):
            sim.compute_electric_charge_density()
            return
    if mode == DISPLAY_NOETHER_CHARGE:
        if hasattr(sim, "compute_noether_charge_density"):
            sim.compute_noether_charge_density()
            return

# ---------------- Convenience viewer loop ----------------
def run_viewer(sim, params, *, steps_per_frame: int = 5, fps_print_every: int = 120):
    """
    Run an interactive CUDA-OpenGL viewer loop for a Simulation.

    Usage:
    - run_viewer(sim, sim.rp, steps_per_frame=5)
      (`params` is used mainly for xlen/ylen and window sizing; passing ResolvedParams is typical.)

    What happens per frame:
    1) poll input events
    2) advance the solver by `steps_per_frame` (if newtonflow enabled)
    3) compute observables for HUD (energy/skyrmion/vortex)
    4) compute the active display density into sim.en (if needed)
    5) compute vmin/vmax via device reductions (compute_min/compute_max)
    6) render into CUDA-mapped PBO and draw to screen
    7) optionally print FPS every `fps_print_every` frames

    Keyboard shortcuts are handled by GLRenderer._on_key (see that docstring).
    """
    renderer = GLRenderer(params.xlen, params.ylen)
    renderer.bind_sim(sim)

    state = {"energy": None, "error": None, "epochs": 0}
    last_time = time.time()
    frames = 0

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
                if "electric_charge" in obs:
                    state["rho"] = float(obs["electric_charge"])
            except Exception:
                pass

            _compute_density_for_mode(sim, renderer.display_mode)

            vmin = compute_min(sim.en, state["min_partial"], sim.en.size)
            vmax = compute_max(sim.en, state["max_partial"], sim.en.size)
            if vmax <= vmin:
                vmax = vmin + 1e-30

            if renderer.display_mode in (DISPLAY_ENERGY, DISPLAY_HIGGS):
                use_jet = True
            elif renderer.display_mode in (DISPLAY_MAGNETIC_FLUX, DISPLAY_ELECTRIC_CHARGE, DISPLAY_NOETHER_CHARGE):
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

            frames += 1
            if fps_print_every and frames % fps_print_every == 0:
                now = time.time()
                dt0 = now - last_time
                fps = frames / dt0 if dt0 > 0 else 0.0
                print(f"FPS ~ {fps:.1f} (steps/frame={steps_per_frame})")
                frames = 0
                last_time = now

    finally:
        renderer.close()