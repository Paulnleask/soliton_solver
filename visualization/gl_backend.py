# =====================================================================================
# soliton_solver/visualization/gl_backend.py
# =====================================================================================
"""
CUDA-OpenGL interoperability backend for soliton_solver.

Usage overview:
- Create a window + GL resources:
    backend = GLBackend(width, height, title="...")
- Each frame:
    backend.begin_frame()
    mapped = backend.map_pbo()
    # Wrap mapped.ptr as a Numba device array and run CUDA kernels that write RGBA
    # (typically width*height*4 bytes) into the PBO memory.
    backend.unmap_pbo()
    backend.upload_and_draw()
    backend.end_frame()
- Optional: backend.set_hud_text(top="...", bottom="...") to draw simple HUD bars.

This module is theory-agnostic:
- GLFW window + OpenGL texture rendering
- PBO management + CUDA graphics interop
- HUD drawing utilities (legacy fixed-function; disabled by default on macOS core profile)
- Shader compile/link helpers
- Device pointer -> Numba device array view (cuda_array_interface)

Theory logic (what to render, how to compute densities, user interactions)
lives elsewhere.
"""

# ---------------- Imports ----------------
from __future__ import annotations
import ctypes
import platform
import time
from dataclasses import dataclass
import numpy as np
import glfw
from OpenGL import GL
from OpenGL import GLUT
from numba import cuda
try:
    from cuda.bindings import runtime as cudart  # cuda-python / cuda-bindings >= 13
except Exception:
    from cuda import cudart  # older trampoline style

# ---------------- CUDA err to int ----------------
def _cuda_err_to_int(err) -> int:
    """
    Normalize a CUDA error value to an integer error code.

    Usage:
        code = _cuda_err_to_int(err)

    Parameters:
        err: CUDA error value (enum/int/ctypes-like), as returned by cudart bindings.

    Outputs:
        - Returns an int suitable for comparison against the cudaSuccess value.
    """
    try:
        return int(err)
    except Exception:
        return 1

# ---------------- CUDA success value ----------------
def _cuda_success_value() -> int:
    """
    Get the integer value corresponding to cudaSuccess for the active cudart binding.

    Usage:
        ok = (_cuda_err_to_int(err) == _cuda_success_value())

    Parameters:
        None

    Outputs:
        - Returns the cudaSuccess integer code (best-effort across cudart binding variants).
    """
    if hasattr(cudart, "cudaError_t") and hasattr(cudart.cudaError_t, "cudaSuccess"):
        try:
            return int(cudart.cudaError_t.cudaSuccess)
        except Exception:
            return 0
    return 0

# ---------------- CUDA checker ----------------
def _cuda_check(err, where: str = "CUDA call"):
    """
    Raise RuntimeError if a CUDA runtime call did not return success.

    Usage:
        out = cudart.someCall(...)
        _cuda_check(out, "someCall")

    Parameters:
        err: CUDA return value or (err, ...) tuple from cudart.
        where: Context string included in the raised error.

    Outputs:
        - Raises RuntimeError on failure.
        - Returns None on success.
    """
    if isinstance(err, tuple) and len(err) >= 1:
        err = err[0]
    if _cuda_err_to_int(err) != _cuda_success_value():
        raise RuntimeError(f"{where} failed with error={err!r}")

# ---------------- CUDA GL write, discard, flag ----------------
def _cuda_gl_write_discard_flag() -> int:
    """
    Get the enum value for registering a GL buffer with CUDA as write-discard.

    Usage:
        flags = _cuda_gl_write_discard_flag()
        res = _cuda_register_gl_buffer(pbo_id, flags)

    Parameters:
        None

    Outputs:
        - Returns an int flag value used with cudaGraphicsGLRegisterBuffer.
    """
    if hasattr(cudart, "cudaGraphicsRegisterFlagsWriteDiscard"):
        try:
            return int(getattr(cudart, "cudaGraphicsRegisterFlagsWriteDiscard"))
        except Exception:
            pass
    if hasattr(cudart, "cudaGraphicsRegisterFlags"):
        enum = getattr(cudart, "cudaGraphicsRegisterFlags")
        for name in ("cudaGraphicsRegisterFlagsWriteDiscard", "WriteDiscard"):
            if hasattr(enum, name):
                try:
                    return int(getattr(enum, name))
                except Exception:
                    pass
    return 2

# ---------------- Check if is ctype instance ----------------
def _is_ctypes_instance(x) -> bool:
    """
    Check whether a value is a ctypes scalar/structure instance.

    Usage:
        if _is_ctypes_instance(resource):
            ...

    Parameters:
        x: Object to test.

    Outputs:
        - Returns True if x is a ctypes scalar or ctypes.Structure instance, else False.
    """
    return isinstance(x, (ctypes._SimpleCData, ctypes.Structure))

# ---------------- Register GL buffers ----------------
def _cuda_register_gl_buffer(pbo_id: int, flags: int):
    """
    Register an OpenGL buffer object (PBO) with CUDA and return a graphics resource handle.

    Usage:
        resource = _cuda_register_gl_buffer(pbo_id, flags)

    Parameters:
        pbo_id: OpenGL buffer object id (GLuint).
        flags: CUDA registration flags (e.g. write-discard).

    Outputs:
        - Registers the GL buffer with CUDA.
        - Returns a CUDA graphics resource handle usable with map/unmap calls.
    """
    # runtime-style first: (err, resource) = cudaGraphicsGLRegisterBuffer(pbo, flags)
    try:
        out = cudart.cudaGraphicsGLRegisterBuffer(int(pbo_id), int(flags))
        if isinstance(out, tuple) and len(out) >= 2:
            err, resource = out[0], out[1]
            _cuda_check(err, "cudaGraphicsGLRegisterBuffer")
            return resource
    except TypeError:
        pass

    # ctypes out-param fallback
    resource = ctypes.c_void_p()
    out = cudart.cudaGraphicsGLRegisterBuffer(ctypes.byref(resource), int(pbo_id), int(flags))
    _cuda_check(out, "cudaGraphicsGLRegisterBuffer")
    return resource

# ---------------- CUDA unregister resource ----------------
def _cuda_unregister_resource(resource):
    """
    Unregister a CUDA graphics resource created from an OpenGL buffer.

    Usage:
        _cuda_unregister_resource(resource)

    Parameters:
        resource: CUDA graphics resource handle returned by _cuda_register_gl_buffer.

    Outputs:
        - Releases the CUDA-side registration for the GL resource.
    """
    out = cudart.cudaGraphicsUnregisterResource(resource)
    _cuda_check(out, "cudaGraphicsUnregisterResource")

# ---------------- CUDA map resource ----------------
def _cuda_map_resource(resource):
    """
    Map a CUDA graphics resource and return (device_ptr, nbytes) for the mapped region.

    Usage:
        ptr, nbytes = _cuda_map_resource(resource)

    Parameters:
        resource: CUDA graphics resource handle registered from a GL buffer.

    Outputs:
        - Maps the resource for CUDA access.
        - Returns:
            ptr: integer device pointer to the mapped storage.
            nbytes: size of the mapped storage in bytes.
        - Raises TypeError if all binding-specific call signatures fail.
    """
    map_attempts = [
        lambda: cudart.cudaGraphicsMapResources(1, [resource], 0),
        lambda: cudart.cudaGraphicsMapResources(1, (resource,), 0),
        lambda: cudart.cudaGraphicsMapResources(1, resource, 0),
    ]

    for attempt in map_attempts:
        try:
            out = attempt()
            _cuda_check(out, "cudaGraphicsMapResources")

            out2 = cudart.cudaGraphicsResourceGetMappedPointer(resource)
            if isinstance(out2, tuple) and len(out2) >= 3:
                err, ptr, size = out2[0], out2[1], out2[2]
                _cuda_check(err, "cudaGraphicsResourceGetMappedPointer")
                return int(ptr), int(size)
        except TypeError:
            continue

    # ctypes fallback
    if not _is_ctypes_instance(resource):
        raise TypeError(
            "CUDA-OpenGL interop map(): runtime-style cudaGraphicsMapResources calls all failed for a "
            "non-ctypes resource handle. This indicates a bindings signature mismatch."
        )

    out = cudart.cudaGraphicsMapResources(1, ctypes.byref(resource), 0)
    _cuda_check(out, "cudaGraphicsMapResources")

    dev_ptr = ctypes.c_void_p()
    size = ctypes.c_size_t()
    out2 = cudart.cudaGraphicsResourceGetMappedPointer(ctypes.byref(dev_ptr), ctypes.byref(size), resource)
    _cuda_check(out2, "cudaGraphicsResourceGetMappedPointer")
    return int(dev_ptr.value), int(size.value)

# ---------------- CUDA unmap resource ----------------
def _cuda_unmap_resource(resource):
    """
    Unmap a previously mapped CUDA graphics resource.

    Usage:
        _cuda_unmap_resource(resource)

    Parameters:
        resource: CUDA graphics resource handle that is currently mapped.

    Outputs:
        - Unmaps the resource, allowing OpenGL to access the buffer again.
        - Raises TypeError if binding-specific unmap signatures fail for non-ctypes handles.
    """
    unmap_attempts = [
        lambda: cudart.cudaGraphicsUnmapResources(1, [resource], 0),
        lambda: cudart.cudaGraphicsUnmapResources(1, (resource,), 0),
        lambda: cudart.cudaGraphicsUnmapResources(1, resource, 0),
    ]

    for attempt in unmap_attempts:
        try:
            out = attempt()
            _cuda_check(out, "cudaGraphicsUnmapResources")
            return
        except TypeError:
            continue

    if not _is_ctypes_instance(resource):
        raise TypeError(
            "CUDA-OpenGL interop unmap(): runtime-style cudaGraphicsUnmapResources calls all failed for a "
            "non-ctypes resource handle."
        )

    out = cudart.cudaGraphicsUnmapResources(1, ctypes.byref(resource), 0)
    _cuda_check(out, "cudaGraphicsUnmapResources")

# ---------------- OpenGL shader compiler ----------------
def _compile_shader(src: str, shader_type):
    """
    Compile a single GLSL shader stage.

    Usage:
        shader = _compile_shader(src, GL.GL_VERTEX_SHADER)

    Parameters:
        src: GLSL source code string.
        shader_type: OpenGL shader type enum (e.g. GL.GL_VERTEX_SHADER).

    Outputs:
        - Returns the compiled shader object id.
        - Raises RuntimeError with the compiler log on compilation failure.
    """
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, src)
    GL.glCompileShader(shader)

    ok = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
    if not ok:
        log = GL.glGetShaderInfoLog(shader).decode("utf-8", errors="ignore")
        raise RuntimeError(f"Shader compile failed:\n{log}")
    return shader

# ---------------- Link program to OpenGL ----------------
def _link_program(vs_src: str, fs_src: str):
    """
    Compile vertex/fragment shaders and link them into an OpenGL program.

    Usage:
        prog = _link_program(vs_src, fs_src)

    Parameters:
        vs_src: Vertex shader GLSL source.
        fs_src: Fragment shader GLSL source.

    Outputs:
        - Returns the linked OpenGL program id.
        - Deletes intermediate shader objects after linking.
        - Raises RuntimeError with the linker log on link failure.
    """
    vs = _compile_shader(vs_src, GL.GL_VERTEX_SHADER)
    fs = _compile_shader(fs_src, GL.GL_FRAGMENT_SHADER)

    prog = GL.glCreateProgram()
    GL.glAttachShader(prog, vs)
    GL.glAttachShader(prog, fs)
    GL.glLinkProgram(prog)

    ok = GL.glGetProgramiv(prog, GL.GL_LINK_STATUS)
    if not ok:
        log = GL.glGetProgramInfoLog(prog).decode("utf-8", errors="ignore")
        raise RuntimeError(f"Program link failed:\n{log}")

    GL.glDeleteShader(vs)
    GL.glDeleteShader(fs)
    return prog

# ---------------- Wrap pointer as Numba array on device ----------------
def cuda_array_from_ptr(ptr_int: int, shape, dtype):
    """
    Wrap an existing device pointer as a Numba CUDA device array view (no allocation, no copy).

    Usage:
        mapped = backend.map_pbo()
        rgba = cuda_array_from_ptr(mapped.ptr, (H, W, 4), np.uint8)

    Parameters:
        ptr_int: Integer device pointer address.
        shape: Array shape tuple to interpret the raw buffer.
        dtype: Numpy dtype for the view (e.g. np.uint8).

    Outputs:
        - Returns a Numba device array view over the given pointer using __cuda_array_interface__.
    """
    iface = {
        "data": (int(ptr_int), False),
        "shape": tuple(shape),
        "strides": None,
        "typestr": np.dtype(dtype).newbyteorder("|").str,
        "version": 2,
    }

    class _Wrapper:
        __slots__ = ("__cuda_array_interface__",)

        def __init__(self, iface_dict):
            self.__cuda_array_interface__ = iface_dict

    return cuda.as_cuda_array(_Wrapper(iface))

@dataclass
class MappedPBO:
    """
    Container describing a mapped CUDA-registered PBO region.

    Usage:
        mapped = backend.map_pbo()
        ptr = mapped.ptr
        nbytes = mapped.nbytes

    Parameters:
        ptr: Integer device pointer to the mapped buffer storage.
        nbytes: Size of the mapped region in bytes.

    Outputs:
        - Provides a simple typed return object for map_pbo().
    """
    ptr: int
    nbytes: int

class GLBackend:
    """
    GLFW window + OpenGL texture quad renderer backed by a CUDA-mapped PBO.

    Usage:
        backend = GLBackend(width, height, title="CUDA-OpenGL")
        while not backend.should_close():
            backend.begin_frame()
            mapped = backend.map_pbo()
            rgba = cuda_array_from_ptr(mapped.ptr, (backend.height, backend.width, 4), np.uint8)
            launch CUDA kernels writing RGBA into `rgba`
            backend.unmap_pbo()
            backend.upload_and_draw()
            backend.end_frame()
        backend.close()

    Parameters:
        width, height: Window and framebuffer dimensions (pixels).
        title: Initial window title.

    Outputs:
        - Creates a window, GL texture, fullscreen quad, and a PBO registered with CUDA.
        - Provides per-frame map/unmap and draw calls for CUDA-driven rendering.
    """
    def __init__(self, width: int, height: int, title: str = "CUDA-OpenGL"):
        """
        Create the window, allocate GL resources, and register the PBO with CUDA.

        Usage:
            backend = GLBackend(width, height, title="...")

        Parameters:
            width, height: Window dimensions in pixels.
            title: Initial window title.

        Outputs:
            - Initializes GLFW and creates an OpenGL context.
            - Allocates:
                - PBO sized width*height*4 bytes (RGBA8).
                - 2D texture for display.
                - Shader program and fullscreen quad VAO/VBO/EBO.
            - Registers the PBO with CUDA and stores the CUDA graphics resource handle.
            - Enables/disables HUD drawing depending on platform (disabled by default on macOS core profile).
        """
        self.width = int(width)
        self.height = int(height)

        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)

        # On macOS, glfw only exposes a core profile; legacy HUD requires compat profile.
        if platform.system() == "Darwin":
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        else:
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_COMPAT_PROFILE)

        self.window = glfw.create_window(self.width, self.height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("GLFW window creation failed")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        # GLUT only for bitmap-font HUD text (no GLUT window)
        try:
            GLUT.glutInit()
        except Exception:
            try:
                GLUT.glutInit(["cuda_gl"])
            except Exception:
                pass

        self.hud_enabled = (platform.system() != "Darwin")
        self.hud_top_text = ""
        self.hud_bottom_text = ""

        # PBO
        self.pbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, self.width * self.height * 4, None, GL.GL_STREAM_DRAW)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

        # Texture
        self.tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, self.width, self.height, 0,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        # Shader + fullscreen quad
        vs_src = r"""
        #version 330 core
        layout (location = 0) in vec2 pos;
        layout (location = 1) in vec2 uv;
        out vec2 vUV;
        void main() {
            vUV = uv;
            gl_Position = vec4(pos, 0.0, 1.0);
        }
        """

        fs_src = r"""
        #version 330 core
        in vec2 vUV;
        out vec4 FragColor;
        uniform sampler2D tex;
        void main() {
            FragColor = texture(tex, vUV);
        }
        """

        self.prog = _link_program(vs_src, fs_src)

        quad = np.array(
            [
                -1.0, -1.0, 0.0, 0.0,
                 1.0, -1.0, 1.0, 0.0,
                 1.0,  1.0, 1.0, 1.0,
                -1.0,  1.0, 0.0, 1.0,
            ],
            dtype=np.float32,
        )
        indices = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        self.ebo = GL.glGenBuffers(1)

        GL.glBindVertexArray(self.vao)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, quad.nbytes, quad, GL.GL_STATIC_DRAW)

        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)

        stride = 4 * quad.itemsize
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, False, stride, ctypes.c_void_p(0))
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, False, stride, ctypes.c_void_p(2 * quad.itemsize))

        GL.glBindVertexArray(0)

        GL.glUseProgram(self.prog)
        loc = GL.glGetUniformLocation(self.prog, "tex")
        GL.glUniform1i(loc, 0)
        GL.glUseProgram(0)

        # Register PBO with CUDA
        flags = _cuda_gl_write_discard_flag()
        self.cuda_res = _cuda_register_gl_buffer(int(self.pbo), int(flags))

    # ---------------- Window lifecycle ----------------
    def should_close(self) -> bool:
        """
        Check whether the window has been requested to close.

        Usage:
            while not backend.should_close():
                ...

        Parameters:
            None

        Outputs:
            - Returns True if the GLFW window should close, else False.
        """
        return glfw.window_should_close(self.window)

    def begin_frame(self):
        """
        Perform start-of-frame event processing.

        Usage:
            backend.begin_frame()

        Parameters:
            None

        Outputs:
            - Polls GLFW events (keyboard/mouse/window).
        """
        glfw.poll_events()

    def end_frame(self):
        """
        Present the rendered frame.

        Usage:
            backend.end_frame()

        Parameters:
            None

        Outputs:
            - Swaps GLFW buffers to display the current rendered quad.
        """
        glfw.swap_buffers(self.window)

    def set_window_title(self, title: str):
        """
        Set the window title.

        Usage:
            backend.set_window_title(f"t={t:.3f}")

        Parameters:
            title: New window title string.

        Outputs:
            - Updates the OS window title via GLFW.
        """
        glfw.set_window_title(self.window, str(title))

    def close(self):
        """
        Release CUDA/GL resources and destroy the window.

        Usage:
            backend.close()

        Parameters:
            None

        Outputs:
            - Unregisters the CUDA graphics resource (best-effort).
            - Deletes GL program, buffers, and textures (best-effort).
            - Destroys the GLFW window and terminates GLFW.
        """
        if getattr(self, "cuda_res", None) is not None:
            try:
                _cuda_unregister_resource(self.cuda_res)
            except Exception:
                pass
            self.cuda_res = None

        try:
            if getattr(self, "prog", None):
                GL.glDeleteProgram(self.prog)
            if getattr(self, "vao", None):
                GL.glDeleteVertexArrays(1, [self.vao])
            if getattr(self, "vbo", None):
                GL.glDeleteBuffers(1, [self.vbo])
            if getattr(self, "ebo", None):
                GL.glDeleteBuffers(1, [self.ebo])
            if getattr(self, "tex", None):
                GL.glDeleteTextures(1, [self.tex])
            if getattr(self, "pbo", None):
                GL.glDeleteBuffers(1, [self.pbo])
        except Exception:
            pass

        try:
            if getattr(self, "window", None):
                glfw.destroy_window(self.window)
        finally:
            glfw.terminate()

    # ---------------- CUDA map/unmap ----------------
    def map_pbo(self) -> MappedPBO:
        """
        Map the CUDA-registered PBO and return its device pointer and size.

        Usage:
            mapped = backend.map_pbo()
            rgba = cuda_array_from_ptr(mapped.ptr, (H, W, 4), np.uint8)

        Parameters:
            None

        Outputs:
            - Maps the PBO for CUDA access.
            - Returns a MappedPBO with:
                ptr: integer device pointer to the PBO storage.
                nbytes: mapped size in bytes.
        """
        ptr, nbytes = _cuda_map_resource(self.cuda_res)
        return MappedPBO(ptr=int(ptr), nbytes=int(nbytes))

    def unmap_pbo(self):
        """
        Unmap the PBO after CUDA writes complete.

        Usage:
            backend.unmap_pbo()

        Parameters:
            None

        Outputs:
            - Unmaps the CUDA graphics resource so OpenGL can read the PBO again.
        """
        _cuda_unmap_resource(self.cuda_res)

    # ---------------- PBO -> texture upload + draw ----------------
    def _upload_pbo_to_texture(self):
        """
        Upload the current PBO contents into the OpenGL texture.

        Usage:
            backend._upload_pbo_to_texture()  # internal, called by upload_and_draw()

        Parameters:
            None

        Outputs:
            - Performs glTexSubImage2D from the PBO into the 2D texture.
            - Assumes the PBO is not currently mapped by CUDA.
        """
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        GL.glTexSubImage2D(
            GL.GL_TEXTURE_2D, 0, 0, 0, self.width, self.height,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, ctypes.c_void_p(0)
        )
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

    def _draw_fullscreen_quad(self):
        """
        Draw the texture to the window using a fullscreen quad.

        Usage:
            backend._draw_fullscreen_quad()  # internal, called by upload_and_draw()

        Parameters:
            None

        Outputs:
            - Clears the framebuffer and draws a textured quad using the shader program.
            - Calls _draw_hud() after drawing the quad.
        """
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glViewport(0, 0, self.width, self.height)
        GL.glClearColor(0.05, 0.05, 0.05, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        GL.glUseProgram(self.prog)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)

        GL.glBindVertexArray(self.vao)
        GL.glDrawElements(GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_INT, ctypes.c_void_p(0))
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

        self._draw_hud()

    def upload_and_draw(self):
        """
        Upload the PBO to the texture and draw it to the window.

        Usage:
            backend.unmap_pbo()
            backend.upload_and_draw()

        Parameters:
            None

        Outputs:
            - Uploads the PBO contents into the GL texture.
            - Draws the fullscreen quad displaying the updated texture.
        """
        self._upload_pbo_to_texture()
        self._draw_fullscreen_quad()

    # ---------------- HUD ----------------
    def set_hud_text(self, *, top: str = "", bottom: str = ""):
        """
        Set HUD overlay text for the top and/or bottom bars.

        Usage:
            backend.set_hud_text(top="E=...", bottom="err=...")

        Parameters:
            top: Text for the top HUD bar (empty to clear).
            bottom: Text for the bottom HUD bar (empty to clear).

        Outputs:
            - Stores HUD strings used by _draw_hud().
        """
        self.hud_top_text = str(top or "")
        self.hud_bottom_text = str(bottom or "")

    def _draw_text(self, x: int, y: int, s: str):
        """
        Draw a text string using a GLUT bitmap font at pixel coordinates.

        Usage:
            backend._draw_text(x, y, "text")  # internal

        Parameters:
            x, y: Pixel coordinates in HUD orthographic space.
            s: Text string to draw.

        Outputs:
            - Issues GLUT bitmap character calls to draw the string (best-effort).
        """
        GL.glRasterPos2i(int(x), int(y))
        for ch in (s or ""):
            try:
                GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_8_BY_13, ord(ch))
            except Exception:
                pass

    def _draw_hud_bar(self, *, y0: int, height: int, text_y: int, text: str):
        """
        Draw a translucent HUD bar with text using legacy fixed-function OpenGL.

        Usage:
            backend._draw_hud_bar(y0=0, height=18, text_y=13, text="...")  # internal

        Parameters:
            y0: Top edge of the bar in pixels.
            height: Bar height in pixels.
            text_y: Baseline y coordinate for text.
            text: Text string to draw.

        Outputs:
            - Draws a blended quad and renders text over it.
            - No-ops if height <= 0.
        """
        if height <= 0:
            return

        GL.glPushAttrib(GL.GL_ENABLE_BIT | GL.GL_COLOR_BUFFER_BIT | GL.GL_TRANSFORM_BIT)
        GL.glDisable(GL.GL_TEXTURE_2D)

        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPushMatrix()
        GL.glLoadIdentity()
        GL.glOrtho(0, self.width, self.height, 0, -1, 1)

        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPushMatrix()
        GL.glLoadIdentity()

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

        GL.glColor4f(0.0, 0.0, 0.0, 0.6)
        GL.glBegin(GL.GL_QUADS)
        GL.glVertex2i(0, y0)
        GL.glVertex2i(self.width, y0)
        GL.glVertex2i(self.width, y0 + height)
        GL.glVertex2i(0, y0 + height)
        GL.glEnd()

        GL.glColor3f(1.0, 1.0, 1.0)
        self._draw_text(8, text_y, text)

        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_PROJECTION)
        GL.glPopMatrix()
        GL.glMatrixMode(GL.GL_MODELVIEW)
        GL.glPopAttrib()

    def _draw_hud(self):
        """
        Draw top/bottom HUD overlays if enabled.

        Usage:
            backend._draw_hud()  # internal, called by _draw_fullscreen_quad()

        Parameters:
            None

        Outputs:
            - Draws a top bar if hud_top_text is non-empty.
            - Draws a bottom bar if hud_bottom_text is non-empty.
            - No-ops if HUD is disabled or both strings are empty.
        """
        if not self.hud_enabled:
            return

        top = self.hud_top_text
        bottom = self.hud_bottom_text
        if not top and not bottom:
            return

        bar_h = 18
        if top:
            self._draw_hud_bar(y0=0, height=bar_h, text_y=13, text=top)
        if bottom:
            self._draw_hud_bar(y0=self.height - bar_h, height=bar_h, text_y=self.height - 5, text=bottom)