# =========================
# soliton_solver/theories/baby_skyrme/render_gl.py
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
from soliton_solver.theories.baby_skyrme.initial_config import create_skyrmion_kernel
from soliton_solver.core.colormaps import clip_u8, _hsv_to_rgb, render_jet_density_to_rgba, render_gray_density_to_rgba, render_magnetization_to_rgba

# ---------------- Display modes ----------------
DISPLAY_ENERGY = 1
DISPLAY_MAGNETIZATION = 2

DISPLAY_TITLES = {
    DISPLAY_ENERGY: "En",
    DISPLAY_MAGNETIZATION: "Sk",
}

# ---------------- Skyrmion overlay preview ----------------
@cuda.jit
def overlay_skyrmion_preview_kernel(pbo_rgba, grid, pxi, pxj, rotation_angle, ansatz_bloch, ansatz_neel, ansatz_anti, p_i, p_f):
    """
    Draw a translucent "preview" skyrmion overlay into the already-rendered RGBA PBO.

    Usage:
    - Launched after the main render kernel (jet/gray/magnetization) when:
        preview_active == True and soliton_mode == "skyrmion"
    - Does not modify the simulation Field; it only blends colors into pbo_rgba.

    Parameters:
    - pbo_rgba: uint8 (H, W, 4) device view of the mapped OpenGL PBO
    - grid: device coordinate grid (flattened), used to convert pixels to physical coords
    - pxi, pxj: integer pixel indices in simulation grid that define the skyrmion center
    - rotation_angle: extra angle offset applied to the skyrmion azimuth (interactive rotation)
    - ansatz_*: which skyrmion ansatz to preview (bloch/neel/anti)
    - p_i, p_f: device parameter arrays (expects xlen/ylen and skyrmion_number etc.)

    Notes:
    - The overlay uses HSV coloring consistent with magnetization rendering.
    - A Gaussian fade (sigma ~ 0.1 rmax) blends the preview into the current frame.
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

    max_r = rmax / 10.0 * skN
    if r1 > max_r:
        fm = 0.0
    else:
        t = (2.0 * r1 / max_r)
        fm = math.pi * math.exp(-(t * t))

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

    def __init__(self, width: int, height: int, title: str = "superferro (CUDA-OpenGL)"):
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

        self.preview_active = False
        self.preview_pxi = 0
        self.preview_pxj = 0
        self.iso_rotating = False
        self.iso_start_mouse_x = 0.0
        self.iso_start_angle = 0.0
        self.skyrmion_rotation = 0.0

        self.vortex_type = 0
        self.request_save_output = False

        # callbacks
        glfw.set_key_callback(self.backend.window, self._on_key)
        glfw.set_cursor_pos_callback(self.backend.window, self._on_cursor_pos)
        glfw.set_mouse_button_callback(self.backend.window, self._on_mouse_button)

        self._update_window_title()

    # ---------------- Bind a Simulation instance for interactive injection + overlay ----------------
    def bind_sim(self, sim):
        """
        Attach a Simulation instance to enable injection and to read current parameter flags.

        Usage:
        - renderer.bind_sim(sim)

        Effects:
        - Enables skyrmion/vortex insertion kernels to write into sim.Field.
        - Reads initial skyrmion_rotation from sim.p_f_h[9] if present.
        """
        self._sim = sim
        try:
            self.skyrmion_rotation = float(sim.p_f_h[9])
        except Exception:
            self.skyrmion_rotation = 0.0

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

    @staticmethod
    def _wrap2pi(a: float) -> float:
        """
        Wrap an angle to [0, 2π).

        Usage:
        - Used to keep interactive skyrmion_rotation bounded.
        """
        twopi = 2.0 * math.pi
        a = math.fmod(a, twopi)
        if a < 0.0:
            a += twopi
        return a

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
            self.set_display_mode(DISPLAY_MAGNETIZATION)

        elif key == glfw.KEY_S:
            self.soliton_mode = "skyrmion"
            self.preview_active = True
            self.iso_rotating = False

        elif glfw.KEY_1 <= key <= glfw.KEY_9:
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
        Update skyrmion_number and vortex_number in the bound Simulation (in-place).

        Usage:
        - Called by key handler when pressing number keys 1..9.
        - Writes:
            sim.p_f_h[8] = n  (skyrmion_number)
          and uploads p_f to the device.

        Note:
        - This updates only the device parameter array; it does not rebuild Simulation buffers.
        """
        sim = getattr(self, "_sim", None)
        if sim is None:
            return

        sim.p_f_h[8] = float(n)  # skyrmion_number

        try:
            sim.p_f_d.copy_to_device(sim.p_f_h)
        except Exception:
            sim.p_f_d = cuda.to_device(sim.p_f_h)

    def _on_cursor_pos(self, window, xpos, ypos):
        """
        GLFW mouse-move handler for skyrmion preview and rotation.

        Usage:
        - If not in skyrmion mode: disables preview.
        - If rotating (mouse held down): updates skyrmion_rotation based on horizontal mouse delta.
        - Otherwise: updates preview center (pxi, pxj) from current cursor position.
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
        self.backend.set_window_title(DISPLAY_TITLES.get(self.display_mode, "superferro"))

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
    - DISPLAY_MAGNETIZATION: does nothing here (render uses Field directly)
    """
    if mode == DISPLAY_ENERGY:
        if hasattr(sim, "compute_energy_density"):
            sim.compute_energy_density()
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