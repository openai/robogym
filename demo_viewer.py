# from robogym.envs.dactyl.locked import make_env
from robogym.envs.dactyl.reach import make_env
import glfw
import mujoco
import numpy as np

def init_window(max_width, max_height):
    glfw.init()
    window = glfw.create_window(width=max_width, height=max_height,
                                title='Demo', monitor=None,
                                share=None)
    glfw.make_context_current(window)
    return window

# Fixed camera -1 is the free (unfixed) camera, and each fixed camera has
# a positive index in range (0, self._model.ncam).
_FREE_CAMERA_INDEX = -1

# Index used to distinguish when a camera isn't tracking any particular body.
_NO_BODY_TRACKED_INDEX = -1

# Index used to distinguish a non-existing or an invalid body.
_INVALID_BODY_INDEX = -1

# Zoom factor used when zooming in on the entire scene.
_FULL_SCENE_ZOOM_FACTOR = 1.5

class SceneCamera:
  """A camera used to navigate around and render the scene."""

  def __init__(self,
               model,
               data,
               options,
               settings=None,
               zoom_factor=_FULL_SCENE_ZOOM_FACTOR):
    """Instance initializer.
    Args:
      model: MjModel instance.
      data: MjData instance.
      options: RenderSettings instance.
      settings: Optional, internal camera settings obtained from another
        SceneCamera instance using 'settings' property.
      zoom_factor: The initial zoom factor for zooming into the scene.
    """
    # Design notes:
    # We need to recreate the camera for each new model, because each model
    # defines different fixed cameras and objects to track, and therefore
    # severely the parameters of this class.
    self._scene = wrapper.MjvScene(model)
    self._data = data
    self._model = model
    self._options = options

    self._camera = wrapper.MjvCamera()
    self._camera.trackbodyid = _NO_BODY_TRACKED_INDEX
    self._camera.fixedcamid = _FREE_CAMERA_INDEX
    self._camera.type_ = mujoco.mjtCamera.mjCAMERA_FREE
    self._zoom_factor = zoom_factor

    if settings is not None:
      self._settings = settings
      self.settings = settings
    else:
      self._settings = self._camera

  def set_freelook_mode(self):
    """Enables 6 degrees of freedom of movement for the camera."""
    self._camera.trackbodyid = _NO_BODY_TRACKED_INDEX
    self._camera.fixedcamid = _FREE_CAMERA_INDEX
    self._camera.type_ = mujoco.mjtCamera.mjCAMERA_FREE

  def set_tracking_mode(self, body_id):
    """Latches the camera onto the specified body.
    Leaves the user only 3 degrees of freedom to rotate the camera.
    Args:
      body_id: A positive integer, ID of the body to track.
    """
    if body_id < 0:
      return
    self._camera.trackbodyid = body_id
    self._camera.fixedcamid = _FREE_CAMERA_INDEX
    self._camera.type_ = mujoco.mjtCamera.mjCAMERA_TRACKING

  def set_fixed_mode(self, fixed_camera_id):
    """Fixes the camera in a pre-defined position, taking away all DOF.
    Args:
      fixed_camera_id: A positive integer, Id of a fixed camera defined in the
        scene.
    """
    if fixed_camera_id < 0:
      return
    self._camera.trackbodyid = _NO_BODY_TRACKED_INDEX
    self._camera.fixedcamid = fixed_camera_id
    self._camera.type_ = mujoco.mjtCamera.mjCAMERA_FIXED

  def look_at(self, position, distance):
    """Positions the camera so that it's focused on the specified point."""
    self._camera.lookat[:] = position
    self._camera.distance = distance

  def move(self, action, viewport_offset):
    """Moves the camera around the scene."""
    # Not checking the validity of arguments on purpose. This method is designed
    # to be called very often, so in order to avoid the overhead, all arguments
    # are assumed to be valid.
    mujoco.mjv_moveCamera(self._model.ptr, action, viewport_offset[0],
                          viewport_offset[1], self._scene.ptr, self._camera.ptr)

  def raycast(self, viewport, screen_pos):
    """Shoots a ray from the specified viewport position into the scene."""
    if not self.is_initialized:
      return -1, None
    viewport_pos = viewport.screen_to_inverse_viewport(screen_pos)
    grab_world_pos = np.empty(3, dtype=np.double)
    selected_geom_id_arr = np.intc([-1])
    selected_skin_id_arr = np.intc([-1])
    selected_body_id = mujoco.mjv_select(
        self._model.ptr,
        self._data.ptr,
        self._options.visualization.ptr,
        viewport.aspect_ratio,
        viewport_pos[0],
        viewport_pos[1],
        self._scene.ptr,
        grab_world_pos,
        selected_geom_id_arr,
        selected_skin_id_arr,
    )
    del selected_geom_id_arr, selected_skin_id_arr  # Unused.
    if selected_body_id < 0:
      selected_body_id = _INVALID_BODY_INDEX
      grab_world_pos = None
    return selected_body_id, grab_world_pos

  def render(self, perturbation=None):
    """Renders the scene form this camera's perspective.
    Args:
      perturbation: (Optional), instance of Perturbation.
    Returns:
      Rendered scene, instance of MjvScene.
    """
    perturb_to_render = perturbation.ptr if perturbation else None
    mujoco.mjv_updateScene(self._model.ptr, self._data.ptr,
                           self._options.visualization.ptr, perturb_to_render,
                           self._camera.ptr, mujoco.mjtCatBit.mjCAT_ALL,
                           self._scene.ptr)
    return self._scene

  def zoom_to_scene(self):
    """Zooms in on the entire scene."""
    self.look_at(self._model.stat.center[:],
                 self._zoom_factor * self._model.stat.extent)

    self.settings = self._settings

  @property
  def transform(self):
    """Returns a tuple with camera transform.
    The transform comes in form: (3x3 rotation mtx, 3-component position).
    """
    pos = np.zeros(3)
    forward = np.zeros(3)
    up = np.zeros(3)
    for i in range(3):
      forward[i] = self._scene.camera[0].forward[i]
      up[i] = self._scene.camera[0].up[i]
      pos[i] = (self._scene.camera[0].pos[i] + self._scene.camera[1].pos[i]) / 2
    right = np.cross(forward, up)
    return np.array([right, up, forward]), pos

  @property
  def settings(self):
    """Returns internal camera settings."""
    return self._camera

  @settings.setter
  def settings(self, value):
    """Restores the camera settings."""
    self._camera.type_ = value.type_
    self._camera.fixedcamid = value.fixedcamid
    self._camera.trackbodyid = value.trackbodyid
    self._camera.lookat[:] = value.lookat[:]
    self._camera.distance = value.distance
    self._camera.azimuth = value.azimuth
    self._camera.elevation = value.elevation

  @property
  def name(self):
    """Name of the active camera."""
    if self._camera.type_ == mujoco.mjtCamera.mjCAMERA_TRACKING:
      body_name = self._model.id2name(self._camera.trackbodyid, 'body')
      if body_name:
        return 'Tracking body "%s"' % body_name
      else:
        return 'Tracking body id %d' % self._camera.trackbodyid
    elif self._camera.type_ == mujoco.mjtCamera.mjCAMERA_FIXED:
      camera_name = self._model.id2name(self._camera.fixedcamid, 'camera')
      if camera_name:
        return str(camera_name)
      else:
        return str(self._camera.fixedcamid)
    else:
      return 'Free'

  @property
  def mode(self):
    """Index of the mode the camera is currently in."""
    return self._camera.type_

  @property
  def is_initialized(self):
    """Returns True if camera is properly initialized."""
    if not self._scene:
      return False
    frustum_near = self._scene.camera[0].frustum_near
    frustum_far = self._scene.camera[0].frustum_far
    return frustum_near > 0 and frustum_near < frustum_far


class MouseState(object):
    def __init__(self):
        self.pressed = False


def standalone_viewer():
    window = init_window(600, 450)
    width, height = glfw.get_framebuffer_size(window)
    scalex, _ = glfw.get_window_content_scale(window)

    
    
    if scalex != 1.0:
        viewport = mujoco.MjrRect(-width, -height, width * 2, height * 2)
    else:
        viewport = mujoco.MjrRect(0, 0, width, height)
    env = make_env()
    obs = env.reset()
    context = mujoco.MjrContext(env.unwrapped.mujoco_simulation.sim.model.model(), mujoco.mjtFontScale.mjFONTSCALE_100)
    scene = mujoco.MjvScene(env.unwrapped.mujoco_simulation.sim.model.model(), 500)
    camera = mujoco.MjvCamera()
    mouse = MouseState()
    def handle_scroll(window, x_offset, y_offset):
        # scroll
        mujoco.mjv_moveCamera(env.unwrapped.mujoco_simulation.sim.model.model(), mujoco.mjtMouse.mjMOUSE_ZOOM, 0,
                              0.1 * y_offset, scene, camera)

    def button_change(window, button, state, mods):
        if button == glfw.MOUSE_BUTTON_LEFT and state == glfw.PRESS:
            mouse.pressed = True
            mouse.startpos = mouse.last_position
        elif button == glfw.MOUSE_BUTTON_LEFT and state != glfw.PRESS:
            mouse.pressed = False
        else:
            pass

    def cursor_pos_change(window, posx, posy):
        
        if mouse.pressed:
            dx = posx - mouse.last_position[0]
            dy = posy - mouse.last_position[1]
            mujoco.mjv_moveCamera(env.unwrapped.mujoco_simulation.sim.model.model(), mujoco.mjtMouse.mjMOUSE_MOVE_H, dx/width,
                                  dy/height, scene, camera)
        mouse.last_position = (posx, posy)

    glfw.set_scroll_callback(window, handle_scroll)
    glfw.set_mouse_button_callback(window, button_change)
    glfw.set_cursor_pos_callback(window, cursor_pos_change)
    # import ipdb; ipdb.set_trace()
    mujoco.mjv_updateScene(
        env.unwrapped.mujoco_simulation.sim.model.model(), env.unwrapped.mujoco_simulation.sim.data.data(),
        mujoco.MjvOption(), mujoco.MjvPerturb(),
        camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
    i = 0
    while True and not glfw.window_should_close(window):
        print(env.action_space.sample())
        obs, reward, done, _ = env.step(env.action_space.nvec // 2)
        mujoco.mjv_updateScene(
            env.unwrapped.mujoco_simulation.sim.model.model(), env.unwrapped.mujoco_simulation.sim.data.data(), mujoco.MjvOption(), None,
            camera, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()
        i += 1
        if i % 100 == 0:
            env.reset()


def mujoco_viewer():
    env = make_env()
    obs = env.reset()
    i = 0
    while True:
        obs, reward, done, _ = env.step(env.action_space.sample() * 0)
        env.render()
        i += 1
        if i % 100 == 0:
            env.reset()


def main():
  standalone = False
  if standalone:
    standalone_viewer()
  else:
    mujoco_viewer()


if __name__ == "__main__":
    main()
