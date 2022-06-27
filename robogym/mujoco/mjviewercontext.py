import os
import numpy as np
import glfw
import sys
from abc import ABCMeta, abstractmethod
from . import constants as const
import mujoco


class OpenGLContext(metaclass=ABCMeta):

    @abstractmethod
    def make_context_current(self):
        raise NotImplementedError()

    @abstractmethod
    def set_buffer_size(self, width, height):
        raise NotImplementedError()


class GlfwError(RuntimeError):
    pass


class GlfwContext(OpenGLContext):

    _INIT_WIDTH = 1000
    _INIT_HEIGHT = 1000
    _GLFW_IS_INITIALIZED = False

    def __init__(self, offscreen=False, quiet=False):
        GlfwContext._init_glfw()

        self._width = self._INIT_WIDTH
        self._height = self._INIT_HEIGHT
        self.window = self._create_window(offscreen, quiet=quiet)
        self._set_window_size(self._width, self._height)

    @staticmethod
    def _init_glfw():
        if GlfwContext._GLFW_IS_INITIALIZED:
            return

        if 'glfw' not in globals():
            raise GlfwError("GLFW not installed")

        glfw.set_error_callback(GlfwContext._glfw_error_callback)

        # HAX: sometimes first init() fails, while second works fine.
        glfw.init()
        if not glfw.init():
            raise GlfwError("Failed to initialize GLFW")

        GlfwContext._GLFW_IS_INITIALIZED = True

    def make_context_current(self):
        glfw.make_context_current(self.window)

    def set_buffer_size(self, width, height):
        self._set_window_size(width, height)
        self._width = width
        self._height = height

    def _create_window(self, offscreen, quiet=True):
        if offscreen:
            if not quiet:
                print("Creating offscreen glfw")
            glfw.window_hint(glfw.VISIBLE, 0)
            glfw.window_hint(glfw.DOUBLEBUFFER, 0)
            init_width, init_height = self._INIT_WIDTH, self._INIT_HEIGHT
        else:
            if not quiet:
                print("Creating window glfw")
            glfw.window_hint(glfw.SAMPLES, 4)
            glfw.window_hint(glfw.VISIBLE, 1)
            glfw.window_hint(glfw.DOUBLEBUFFER, 1)
            resolution, _, refresh_rate = glfw.get_video_mode(
                glfw.get_primary_monitor())
            init_width, init_height = resolution

        self._width = init_width
        self._height = init_height
        window = glfw.create_window(
            self._width, self._height, "robogym", None, None)

        if not window:
            raise GlfwError("Failed to create GLFW window")

        return window

    def get_buffer_size(self):
        return glfw.get_framebuffer_size(self.window)

    def _set_window_size(self, target_width, target_height):
        self.make_context_current()
        if target_width != self._width or target_height != self._height:
            self._width = target_width
            self._height = target_height
            glfw.set_window_size(self.window, target_width, target_height)

            # HAX: When running on a Mac with retina screen, the size
            # sometimes doubles
            width, height = glfw.get_framebuffer_size(self.window)
            if target_width != width and "darwin" in sys.platform.lower():
                glfw.set_window_size(self.window, target_width // 2, target_height // 2)

    @staticmethod
    def _glfw_error_callback(error_code, description):
        print("GLFW error (code %d): %s", error_code, description)


# TODO
# class OffscreenOpenGLContext():
#     def __init__(self, device_id):
#         self.device_id = device_id
#         res = initOpenGL(device_id)
#         if res != 1:
#             raise RuntimeError("Failed to initialize OpenGL")

#     def close(self):
#         # TODO: properly close OpenGL in our contexts
#         closeOpenGL()

#     def make_context_current(self):
#         makeOpenGLContextCurrent(self.device_id)

#     def set_buffer_size(self, int width, int height):
#         res = setOpenGLBufferSize(self.device_id, width, height)
#         if res != 1:
#             raise RuntimeError("Failed to set buffer size")


class MjRenderContext(object):
    """
    Class that encapsulates rendering functionality for a
    MuJoCo simulation.
    """

    def __init__(self, sim, offscreen=True, device_id=-1, opengl_backend=None, quiet=False):
        self.sim = sim
        self._setup_opengl_context(offscreen, device_id, opengl_backend, quiet=quiet)
        self.offscreen = offscreen

        # Ensure the model data has been updated so that there
        # is something to render
        sim.forward()
        sim.add_render_context(self)
        self._model_ptr = sim.model.model()
        self._data_ptr = sim.data.data()
        maxgeom = 1000
        self.scn = mujoco.MjvScene(self._model_ptr, maxgeom)
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)
        self.vopt = mujoco.MjvOption()
        self.con = None

        self.pert = mujoco.MjvPerturb()
        self.pert.active = 0
        self.pert.select = 0
        self.pert.skinselect = -1

        self._markers = []
        self._overlay = {}

        self._init_camera(sim)
        self._set_mujoco_buffers()

    def update_sim(self, new_sim):
        if new_sim == self.sim:
            return
        self._model_ptr = new_sim.model.model()
        self._data_ptr = new_sim.data.data()
        self._set_mujoco_buffers()
        for render_context in self.sim.render_contexts:
            new_sim.add_render_context(render_context)
        self.sim = new_sim

    def _set_mujoco_buffers(self):
        self.con = mujoco.MjrContext(self._model_ptr, mujoco.mjtFontScale.mjFONTSCALE_150)
        if self.offscreen:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
            if self.con.currentBuffer != mujoco.mjtFramebuffer.mjFB_OFFSCREEN:
                raise RuntimeError("Offscreen rendering not supported")
        else:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.con);
            if self.con.currentBuffer != mujoco.mjtFramebuffer.mjFB_WINDOW:
                raise RuntimeError("Window rendering not supported")

    def _setup_opengl_context(self, offscreen, device_id, opengl_backend, quiet=False):
        if opengl_backend is None and (not offscreen or sys.platform == 'darwin'):
            # default to glfw for onscreen viewing or mac (both offscreen/onscreen)
            opengl_backend = 'glfw'

        if opengl_backend == 'glfw':
            self.opengl_context = GlfwContext(offscreen=offscreen, quiet=quiet)
        else:
            raise NotImplementedError()
            if device_id < 0:
                if "GPUS" in os.environ:
                    device_id = os.environ["GPUS"]
                else:
                    device_id = os.getenv('CUDA_VISIBLE_DEVICES', '')
                if len(device_id) > 0:
                    device_id = int(device_id.split(',')[0])
                else:
                    # Sometimes env variable is an empty string.
                    device_id = 0
            self.opengl_context = OffscreenOpenGLContext(device_id)

    def _init_camera(self, sim):
        # Make the free camera look at the scene
        self.cam.type = const.CAMERA_FREE
        self.cam.fixedcamid = -1
        for i in range(3):
            self.cam.lookat[i] = np.median(sim.data.geom_xpos[:, i])
        self.cam.distance = sim.model.stat.extent

    def update_offscreen_size(self, width, height):
        if width != self.con.offWidth or height != self.con.offHeight:
            self._model_ptr.vis.global_.offwidth = width
            self._model_ptr.vis.global_.offheight = height
            # mujoco.mjr_freeContext(self._con)
            self._set_mujoco_buffers()

    def render(self, width, height, camera_id=None, segmentation=False):
        rect  = mujoco.MjrRect(0, 0, width, height)

        if self.sim.render_callback is not None:
            self.sim.render_callback(self.sim, self)

        # Sometimes buffers are too small.
        if width > self.con.offWidth or height > self.con.offHeight:
            new_width = max(width, self._model_ptr.vis.global_.offwidth)
            new_height = max(height, self._model_ptr.vis.global_.offheight)
            self.update_offscreen_size(new_width, new_height)

        if camera_id is not None:
            if camera_id == -1:
                self.cam.type = const.CAMERA_FREE
            else:
                self.cam.type = const.CAMERA_FIXED
            self.cam.fixedcamid = camera_id

        # This doesn't really do anything else rather than checking for the size of buffer
        # need to investigate further whi is that a no-op
        # self.opengl_context.set_buffer_size(width, height)

        mujoco.mjv_updateScene(
            self._model_ptr, self._data_ptr,
            self.vopt, self.pert,
            self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scn)

        if segmentation:
            self.scn.flags[const.RND_SEGMENT] = 1
            self.scn.flags[const.RND_IDCOLOR] = 1

        for marker_params in self._markers:
            self._add_marker_to_scene(marker_params)
        
        mujoco.mjr_render(rect, self.scn, self.con)
        for gridpos, (text1, text2) in self._overlay.items():
            mujoco.mjr_overlay(const.FONTSCALE_150, gridpos, rect, text1.encode(), text2.encode(), self.con)

        if segmentation:
            self.scn.flags[const.RND_SEGMENT] = 0
            self.scn.flags[const.RND_IDCOLOR] = 0

    def read_pixels(self, width, height, depth=True, segmentation=False):
        rect  = mujoco.MjrRect(0, 0, width, height)

        rgb_arr = np.zeros(3 * rect.width * rect.height, dtype=np.uint8)
        depth_arr = np.zeros(rect.width * rect.height, dtype=np.float32)
        
        mujoco.mjr_readPixels(rgb_arr, depth_arr, rect, self.con)
        rgb_img = rgb_arr.reshape(rect.height, rect.width, 3)
        ret_img = rgb_img
        if segmentation:
            seg_img = (rgb_img[:, :, 0] + rgb_img[:, :, 1] * (2**8) + rgb_img[:, :, 2] * (2 ** 16))
            seg_img[seg_img >= (self.scn.ngeom + 1)] = 0
            seg_ids = np.full((self.scn.ngeom + 1, 2), fill_value=-1, dtype=np.int32)

            for i in range(self.scn.ngeom):
                geom = self.scn.geoms[i]
                if geom.segid != -1:
                    seg_ids[geom.segid + 1, 0] = geom.objtype
                    seg_ids[geom.segid + 1, 1] = geom.objid
            ret_img = seg_ids[seg_img]

        if depth:
            depth_img = depth_arr.reshape(rect.height, rect.width)
            return (ret_img, depth_img)
        else:
            return ret_img

    def read_pixels_depth(self, buffer):
        ''' Read depth pixels into a preallocated buffer '''
        rect = mujoco.MjrRect(0, 0, buffer.shape[1], buffer.shape[0])
        mujoco.mjr_readPixels(0, buffer, rect, self.con)

    def upload_texture(self, tex_id):
        """ Uploads given texture to the GPU. """
        self.opengl_context.make_context_current()
        mujoco.mjr_uploadTexture(self._model_ptr, self.con, tex_id)

    def draw_pixels(self, image, left, bottom):
        """Draw an image into the OpenGL buffer."""
        assert isinstance(image, np.ndarray), f"expected image to be of type np.ndarray but got {type(image)}."
        assert image.dtype == np.uint8, f"expected image to be of dtype np.uint8 but got {image.dtype}."
        assert image.ndim == 3, f"expected image to have ndim=3 but got {image.ndim}."
        viewport = mujoco.MjrRect(0, 0, image.shape[1], image.shape[0])
        mujoco.mjr_drawPixels(image, 0, viewport, self.con)

    def move_camera(self, action, reldx, reldy):
        """ Moves the camera based on mouse movements. Action is one of mjMOUSE_*. """
        mujoco.mjv_moveCamera(self._model_ptr, action, reldx, reldy, self.scn, self.cam)

    def add_overlay(self, gridpos, text1, text2):
        """ Overlays text on the scene. """
        if gridpos not in self._overlay:
            self._overlay[gridpos] = ["", ""]
        self._overlay[gridpos][0] += text1 + "\n"
        self._overlay[gridpos][1] += text2 + "\n"

    def add_marker(self, **marker_params):
        self._markers.append(marker_params)

    def _add_marker_to_scene(self, marker_params):
        """ Adds marker to scene, and returns the corresponding object. """
        if self.scn.ngeom >= self.scn.maxgeom:
            raise RuntimeError('Ran out of geoms. maxgeom: %d' % self.scn.maxgeom)

        # cdef mjvGeom *g = self.scn.geoms + self.scn.ngeom
        g = mujoco.MjvGeom()

        # default values.
        g.dataid = -1
        g.objtype = const.OBJ_UNKNOWN
        g.objid = -1
        g.category = const.CAT_DECOR
        g.texid = -1
        g.texuniform = 0
        g.texrepeat[0] = 1
        g.texrepeat[1] = 1
        g.emission = 0
        g.specular = 0.5
        g.shininess = 0.5
        g.reflectance = 0
        g.type = const.GEOM_BOX
        g.size[:] = np.ones(3) * 0.1
        g.mat[:] = np.eye(3).flatten()
        g.rgba[:] = np.ones(4)

        for key, value in marker_params.items():
            if isinstance(value, (int, float)):
                setattr(g, key, value)
            elif isinstance(value, (tuple, list, np.ndarray)):
                attr = getattr(g, key)
                attr[:] = np.asarray(value).reshape(attr.shape)
            elif isinstance(value, str):
                assert key == "label", "Only label is a string in mjvGeom."
                if value == None:
                    g.label = 0
                else:
                    g.label = value.encode()
            elif hasattr(g, key):
                raise ValueError("mjvGeom has attr {} but type {} is invalid".format(key, type(value)))
            else:
                raise ValueError("mjvGeom doesn't have field %s" % key)

        self.scn.ngeom += 1
    # def __dealloc__(self):
    #     mjr_freeContext(&self._con)
    #     mjv_freeScene(&self._scn)


class MjRenderContextOffscreen(MjRenderContext):
    def __cinit__(self, sim, device_id):
        super().__init__(sim, offscreen=True, device_id=device_id)


class MjRenderContextWindow(MjRenderContext):
    def __init__(self, sim):
        super().__init__(sim, offscreen=False)
        self.render_swap_callback = None

        assert isinstance(self.opengl_context, GlfwContext), (
            "Only GlfwContext supported for windowed rendering")

    @property
    def window(self):
        return self.opengl_context.window

    def render(self):
        if self.window is None or glfw.window_should_close(self.window):
            return

        glfw.make_context_current(self.window)
        super().render(*glfw.get_framebuffer_size(self.window))
        if self.render_swap_callback is not None:
            self.render_swap_callback()
        glfw.swap_buffers(self.window)
