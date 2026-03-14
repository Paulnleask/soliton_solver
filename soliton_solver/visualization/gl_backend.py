"""
CUDA OpenGL interoperability backend for soliton_solver.

Examples
--------
Use ``GLBackend(width, height, title="...")`` to create a window and rendering backend.
Use ``map_pbo`` and ``unmap_pbo`` to expose the PBO to CUDA kernels.
Use ``upload_and_draw`` to display the rendered RGBA buffer.
"""

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
    from cuda.bindings import runtime as cudart
except Exception:
    from cuda import cudart

def _cuda_err_to_int(err) -> int:
    """
    Convert a CUDA error value to an integer error code.

    Parameters
    ----------
    err
        CUDA error value returned by the runtime bindings.

    Returns
    -------
    int
        Integer error code.

    Examples
    --------
    Use ``code = _cuda_err_to_int(err)`` before comparing against the success code.
    """
    try:
        return int(err)
    except Exception:
        return 1

def _cuda_success_value() -> int:
    """
    Return the integer value corresponding to ``cudaSuccess``.

    Returns
    -------
    int
        Integer success code for the active CUDA runtime binding.

    Examples
    --------
    Use ``_cuda_err_to_int(err) == _cuda_success_value()`` to test whether a CUDA call succeeded.
    """
    if hasattr(cudart, "cudaError_t") and hasattr(cudart.cudaError_t, "cudaSuccess"):
        try:
            return int(cudart.cudaError_t.cudaSuccess)
        except Exception:
            return 0
    return 0

def _cuda_check(err, where: str = "CUDA call"):
    """
    Raise an error if a CUDA runtime call did not succeed.

    Parameters
    ----------
    err
        CUDA return value or tuple containing the return value.
    where : str, optional
        Description of the CUDA call.

    Returns
    -------
    None
        The function returns normally when the CUDA call succeeded.

    Raises
    ------
    RuntimeError
        Raised if the CUDA runtime call failed.

    Examples
    --------
    Use ``_cuda_check(out, "cudaGraphicsMapResources")`` after a CUDA runtime call.
    """
    if isinstance(err, tuple) and len(err) >= 1:
        err = err[0]
    if _cuda_err_to_int(err) != _cuda_success_value():
        raise RuntimeError(f"{where} failed with error={err!r}")

def _cuda_gl_write_discard_flag() -> int:
    """
    Return the CUDA OpenGL registration flag for write discard access.

    Returns
    -------
    int
        Integer flag value used with ``cudaGraphicsGLRegisterBuffer``.

    Examples
    --------
    Use ``flags = _cuda_gl_write_discard_flag()`` before registering a PBO with CUDA.
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

def _is_ctypes_instance(x) -> bool:
    """
    Check whether an object is a ctypes scalar or structure instance.

    Parameters
    ----------
    x
        Object to test.

    Returns
    -------
    bool
        ``True`` if the object is a ctypes scalar or structure instance and ``False`` otherwise.

    Examples
    --------
    Use ``_is_ctypes_instance(resource)`` to decide whether a ctypes fallback path is available.
    """
    return isinstance(x, (ctypes._SimpleCData, ctypes.Structure))

def _cuda_register_gl_buffer(pbo_id: int, flags: int):
    """
    Register an OpenGL buffer object with CUDA.

    Parameters
    ----------
    pbo_id : int
        OpenGL buffer object identifier.
    flags : int
        CUDA registration flags.

    Returns
    -------
    object
        CUDA graphics resource handle for the registered buffer.

    Raises
    ------
    RuntimeError
        Raised if the CUDA registration call failed.

    Examples
    --------
    Use ``resource = _cuda_register_gl_buffer(pbo_id, flags)`` to register a PBO with CUDA.
    """
    try:
        out = cudart.cudaGraphicsGLRegisterBuffer(int(pbo_id), int(flags))
        if isinstance(out, tuple) and len(out) >= 2:
            err, resource = out[0], out[1]
            _cuda_check(err, "cudaGraphicsGLRegisterBuffer")
            return resource
    except TypeError:
        pass

    resource = ctypes.c_void_p()
    out = cudart.cudaGraphicsGLRegisterBuffer(ctypes.byref(resource), int(pbo_id), int(flags))
    _cuda_check(out, "cudaGraphicsGLRegisterBuffer")
    return resource

def _cuda_unregister_resource(resource):
    """
    Unregister a CUDA graphics resource.

    Parameters
    ----------
    resource
        CUDA graphics resource handle.

    Returns
    -------
    None
        The resource is unregistered from CUDA.

    Raises
    ------
    RuntimeError
        Raised if the CUDA unregister call failed.

    Examples
    --------
    Use ``_cuda_unregister_resource(resource)`` during cleanup.
    """
    out = cudart.cudaGraphicsUnregisterResource(resource)
    _cuda_check(out, "cudaGraphicsUnregisterResource")

def _cuda_map_resource(resource):
    """
    Map a CUDA graphics resource and return its device pointer and size.

    Parameters
    ----------
    resource
        CUDA graphics resource handle.

    Returns
    -------
    tuple
        Pair ``(ptr, nbytes)`` containing the mapped device pointer and the mapped size in bytes.

    Raises
    ------
    RuntimeError
        Raised if a CUDA mapping call failed.
    TypeError
        Raised if all supported binding signatures fail.

    Examples
    --------
    Use ``ptr, nbytes = _cuda_map_resource(resource)`` before wrapping the mapped storage as a CUDA array.
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

def _cuda_unmap_resource(resource):
    """
    Unmap a previously mapped CUDA graphics resource.

    Parameters
    ----------
    resource
        CUDA graphics resource handle.

    Returns
    -------
    None
        The resource is unmapped and returned to OpenGL access.

    Raises
    ------
    RuntimeError
        Raised if a CUDA unmap call failed.
    TypeError
        Raised if all supported binding signatures fail.

    Examples
    --------
    Use ``_cuda_unmap_resource(resource)`` after CUDA writes to the mapped PBO have finished.
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

def _compile_shader(src: str, shader_type):
    """
    Compile a GLSL shader stage.

    Parameters
    ----------
    src : str
        GLSL source code.
    shader_type
        OpenGL shader type enum.

    Returns
    -------
    int
        OpenGL shader object identifier.

    Raises
    ------
    RuntimeError
        Raised if shader compilation failed.

    Examples
    --------
    Use ``shader = _compile_shader(src, GL.GL_VERTEX_SHADER)`` to compile a shader stage.
    """
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, src)
    GL.glCompileShader(shader)

    ok = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
    if not ok:
        log = GL.glGetShaderInfoLog(shader).decode("utf-8", errors="ignore")
        raise RuntimeError(f"Shader compile failed:\n{log}")
    return shader

def _link_program(vs_src: str, fs_src: str):
    """
    Compile vertex and fragment shaders and link them into an OpenGL program.

    Parameters
    ----------
    vs_src : str
        Vertex shader source code.
    fs_src : str
        Fragment shader source code.

    Returns
    -------
    int
        Linked OpenGL program identifier.

    Raises
    ------
    RuntimeError
        Raised if program linking failed.

    Examples
    --------
    Use ``prog = _link_program(vs_src, fs_src)`` to create a shader program.
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

def cuda_array_from_ptr(ptr_int: int, shape, dtype):
    """
    Wrap an existing device pointer as a Numba CUDA array view.

    Parameters
    ----------
    ptr_int : int
        Integer device pointer address.
    shape
        Shape used to interpret the raw buffer.
    dtype
        NumPy data type of the array view.

    Returns
    -------
    DeviceNDArray
        Numba CUDA array view over the existing device memory.

    Examples
    --------
    Use ``rgba = cuda_array_from_ptr(mapped.ptr, (H, W, 4), np.uint8)`` to interpret a mapped PBO as an RGBA array.
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
    Container describing a mapped CUDA registered PBO region.

    Parameters
    ----------
    ptr : int
        Integer device pointer to the mapped buffer.
    nbytes : int
        Size of the mapped region in bytes.

    Returns
    -------
    None
        The dataclass stores the mapped pointer and size.

    Examples
    --------
    Use ``mapped = backend.map_pbo()`` and access ``mapped.ptr`` and ``mapped.nbytes``.
    """
    ptr: int
    nbytes: int

class GLBackend:
    """
    GLFW window and OpenGL renderer backed by a CUDA mapped PBO.

    Parameters
    ----------
    width : int
        Window width in pixels.
    height : int
        Window height in pixels.
    title : str, optional
        Initial window title.

    Examples
    --------
    Use ``backend = GLBackend(width, height, title="CUDA-OpenGL")`` to create the rendering backend.
    Use ``backend.map_pbo()`` and ``backend.unmap_pbo()`` to expose the PBO to CUDA.
    Use ``backend.upload_and_draw()`` to display the current frame.
    """
    def __init__(self, width: int, height: int, title: str = "CUDA-OpenGL"):
        """
        Create the window, allocate OpenGL resources, and register the PBO with CUDA.

        Parameters
        ----------
        width : int
            Window width in pixels.
        height : int
            Window height in pixels.
        title : str, optional
            Initial window title.

        Returns
        -------
        None
            The backend is initialized with a window, texture, buffers, shaders, and CUDA interop state.

        Raises
        ------
        RuntimeError
            Raised if GLFW initialization or window creation failed.

        Examples
        --------
        Use ``backend = GLBackend(width, height, title="...")`` to create the rendering backend.
        """
        self.width = int(width)
        self.height = int(height)

        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)

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

        self.pbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, self.pbo)
        GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, self.width * self.height * 4, None, GL.GL_STREAM_DRAW)
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, 0)

        self.tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, self.width, self.height, 0,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

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

        flags = _cuda_gl_write_discard_flag()
        self.cuda_res = _cuda_register_gl_buffer(int(self.pbo), int(flags))

    def should_close(self) -> bool:
        """
        Return whether the window has been requested to close.

        Returns
        -------
        bool
            ``True`` if the window should close and ``False`` otherwise.

        Examples
        --------
        Use ``while not backend.should_close():`` to drive the render loop.
        """
        return glfw.window_should_close(self.window)

    def begin_frame(self):
        """
        Perform start of frame event processing.

        Returns
        -------
        None
            GLFW events are polled.

        Examples
        --------
        Use ``backend.begin_frame()`` at the start of each frame.
        """
        glfw.poll_events()

    def end_frame(self):
        """
        Present the rendered frame.

        Returns
        -------
        None
            The front and back buffers are swapped.

        Examples
        --------
        Use ``backend.end_frame()`` after drawing the current frame.
        """
        glfw.swap_buffers(self.window)

    def set_window_title(self, title: str):
        """
        Set the window title.

        Parameters
        ----------
        title : str
            New window title.

        Returns
        -------
        None
            The window title is updated.

        Examples
        --------
        Use ``backend.set_window_title(f"t={t:.3f}")`` to update the title during rendering.
        """
        glfw.set_window_title(self.window, str(title))

    def close(self):
        """
        Release CUDA and OpenGL resources and destroy the window.

        Returns
        -------
        None
            Registered resources, buffers, textures, and the window are released.

        Examples
        --------
        Use ``backend.close()`` when the render loop finishes.
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

    def map_pbo(self) -> MappedPBO:
        """
        Map the CUDA registered PBO and return its device pointer and size.

        Returns
        -------
        MappedPBO
            Object containing the mapped device pointer and mapped size in bytes.

        Examples
        --------
        Use ``mapped = backend.map_pbo()`` before writing RGBA data from CUDA.
        """
        ptr, nbytes = _cuda_map_resource(self.cuda_res)
        return MappedPBO(ptr=int(ptr), nbytes=int(nbytes))

    def unmap_pbo(self):
        """
        Unmap the PBO after CUDA writes complete.

        Returns
        -------
        None
            The PBO is returned to OpenGL access.

        Examples
        --------
        Use ``backend.unmap_pbo()`` after CUDA kernels finish writing to the mapped PBO.
        """
        _cuda_unmap_resource(self.cuda_res)

    def _upload_pbo_to_texture(self):
        """
        Upload the current PBO contents into the OpenGL texture.

        Returns
        -------
        None
            The texture contents are updated from the PBO.

        Examples
        --------
        Use ``backend._upload_pbo_to_texture()`` internally before drawing the textured quad.
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
        Draw the current texture to the window using a fullscreen quad.

        Returns
        -------
        None
            The framebuffer is cleared and the textured quad is drawn.

        Examples
        --------
        Use ``backend._draw_fullscreen_quad()`` internally after updating the texture.
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
        Upload the PBO to the texture and draw the frame.

        Returns
        -------
        None
            The current PBO contents are displayed in the window.

        Examples
        --------
        Use ``backend.upload_and_draw()`` after unmapping the PBO.
        """
        self._upload_pbo_to_texture()
        self._draw_fullscreen_quad()

    def set_hud_text(self, *, top: str = "", bottom: str = ""):
        """
        Set the top and bottom HUD text.

        Parameters
        ----------
        top : str, optional
            Text for the top HUD bar.
        bottom : str, optional
            Text for the bottom HUD bar.

        Returns
        -------
        None
            The HUD text strings are stored for later drawing.

        Examples
        --------
        Use ``backend.set_hud_text(top="E=...", bottom="err=...")`` to update the HUD text.
        """
        self.hud_top_text = str(top or "")
        self.hud_bottom_text = str(bottom or "")

    def _draw_text(self, x: int, y: int, s: str):
        """
        Draw a text string using a GLUT bitmap font.

        Parameters
        ----------
        x : int
            Pixel x coordinate.
        y : int
            Pixel y coordinate.
        s : str
            Text string to draw.

        Returns
        -------
        None
            The text is drawn using GLUT bitmap characters.

        Examples
        --------
        Use ``backend._draw_text(x, y, "text")`` internally when drawing HUD text.
        """
        GL.glRasterPos2i(int(x), int(y))
        for ch in (s or ""):
            try:
                GLUT.glutBitmapCharacter(GLUT.GLUT_BITMAP_8_BY_13, ord(ch))
            except Exception:
                pass

    def _draw_hud_bar(self, *, y0: int, height: int, text_y: int, text: str):
        """
        Draw a translucent HUD bar with text.

        Parameters
        ----------
        y0 : int
            Top edge of the bar in pixels.
        height : int
            Height of the bar in pixels.
        text_y : int
            Baseline y coordinate for the text.
        text : str
            Text to draw inside the bar.

        Returns
        -------
        None
            The HUD bar and its text are drawn when the height is positive.

        Examples
        --------
        Use ``backend._draw_hud_bar(y0=0, height=18, text_y=13, text="...")`` internally when drawing the HUD.
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
        Draw the top and bottom HUD overlays if enabled.

        Returns
        -------
        None
            The HUD overlays are drawn when enabled and text is present.

        Examples
        --------
        Use ``backend._draw_hud()`` internally after drawing the fullscreen quad.
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