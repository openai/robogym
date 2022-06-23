import os.path
import typing
import xml.etree.ElementTree as et

import mujoco
import numpy as np

ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../assets"))
XML_DIR = os.path.join(ASSETS_DIR, "xmls")


def _format_array(np_array, precision=6):
    """ Format numpy array into a nice string suitable for mujoco XML """
    if not isinstance(np_array, np.ndarray):
        np_array = np.array(np_array, dtype=float)

    # Make sure it's flattened
    if len(np_array.shape) > 1:
        np_array = np_array.reshape(-1)

    if np.min(np.abs(np_array)) > 0.001:
        format_str = "{:.%df}" % precision
    else:
        format_str = "{:.%de}" % precision

    # Finally format a string out of numpy array
    return " ".join(format_str.format(x) for x in np_array)


class StaleMjSimError(Exception):
    """
    Exception indicating the MjSim instance is stale and should no longer be used.
    """
    pass


class MjSim(object):
    def __init__(self, model, data, nsubsteps=1):
        self._stale: bool = False
        self._model = model
        self._data = data
        self._xml = None# model.get_xml()
        self.nsubsteps = nsubsteps
        self._udd_callback = None
        self._substep_callback = None
        self.render_contexts = []
        self._render_context_offscreen = None
        self._render_context_window = None

    def substep_callback(self):
        if self._substep_callback is not None:
            self._substep_callback(self._model.model(), self._data.data())

    def reset(self):
        mujoco.mj_resetData(self._model.model(), self._data.data())
        self.udd_state = None
        self.step_udd()

    def add_render_context(self, render_context):
        self.render_contexts.append(render_context)
        if render_context.offscreen and self._render_context_offscreen is None:
            self._render_context_offscreen = render_context
        elif not render_context.offscreen and self._render_context_window is None:
            self._render_context_window = render_context

    @property
    def udd_callback(self):
        return self._udd_callback

    @udd_callback.setter
    def udd_callback(self, value):
        self._udd_callback = value
        self.udd_state = None
        self.step_udd()
    
    def set_constants(self):
        """
        Set constant fields of mjModel, corresponding to qpos0 configuration.
        """
        mujoco.mj_setConst(self.model.model(), self.data.data())

    def step_udd(self):
        if self._udd_callback is None:
            self.udd_state = {}
        else:
            schema_example = self.udd_state
            self.udd_state = self._udd_callback(self)
            # Check to make sure the udd_state has consistent keys and dimension across steps
            if schema_example is not None:
                keys = set(schema_example.keys()) | set(self.udd_state.keys())
                for key in keys:
                    assert key in schema_example, "Keys cannot be added to udd_state between steps."
                    assert key in self.udd_state, "Keys cannot be dropped from udd_state between steps."
                    if isinstance(schema_example[key], (int, float)):
                        assert isinstance(self.udd_state[key], (int, float)), \
                            "Every value in udd_state must be either a number or a numpy array"
                    else:
                        assert isinstance(self.udd_state[key], np.ndarray), \
                            "Every value in udd_state must be either a number or a numpy array"
                        assert self.udd_state[key].shape == schema_example[key].shape, \
                            "Numpy array values in udd_state must keep the same dimension across steps."

    def get_xml(self):
        """
        Mujoco's internal get_xml() is unreliable as it seems to override the internal
        memory buffer when more than one sim is instantiated. We therefore cache the model
        xml on creation.
        :return:
        """
        return self._xml

    def set_stale(self):
        """
        Set this sim instance as stale so further access to properties of this
        instance will raise error.
        """
        self._stale = True

    def is_stale(self):
        return self._stale

    @property
    def data(self):
        self._ensure_not_stale()
        return self._data

    @property
    def model(self):
        self._ensure_not_stale()
        return self._model

    def _ensure_not_stale(self):
        if self._stale:
            raise StaleMjSimError(
                "You are accessing property of a stale sim instance which is no longer used"
                "by the environment."
            )

    def forward(self):
        return mujoco.mj_forward(self._model.model(), self._data.data())
    
    def step(self, with_udd=True):
        if with_udd:
            self.step_udd()
        for _ in range(self.nsubsteps):
            self.substep_callback()
            mujoco.mj_step(self.model.model(), self.data.data())



class DataProperty(object):
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj._data, self.name)

class ModelProperty(object):
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return getattr(obj._model, self.name)

    
class MjModel(object):
    def _extract_mj_names(self, name_adr, n, obj_type):
        # objects don't need to be named in the XML, so name might be None
        id2name = {i: None for i in range(n)}
        name2id = {}
        names = self._model.names
        for i in range(n):
            end = names.find(b"\x00", name_adr[i])
            name = names[name_adr[i]:end]
            decoded_name = name.decode()
            if decoded_name:
                obj_id = mujoco.mj_name2id(self._model, obj_type, name)
                assert 0 <= obj_id < n and id2name[obj_id] is None
                name2id[decoded_name] = obj_id
                id2name[obj_id] = decoded_name

        # sort names by increasing id to keep order deterministic
        return tuple(id2name[id] for id in sorted(name2id.values())), name2id.__getitem__, name2id, id2name
    
    def __init__(self, model):
        self._model = model
        self.init_names()

    def model(self):
        return self._model

    def init_names(self):
        self.body_names, self.body_name2id, self._body_name2id, self._body_id2name = self._extract_mj_names(self._model.name_bodyadr, self._model.nbody, mujoco.mjtObj.mjOBJ_BODY)
        self.joint_names, self.joint_name2id, self._joint_name2id, self._joint_id2name = self._extract_mj_names(self._model.name_jntadr, self._model.njnt, mujoco.mjtObj.mjOBJ_JOINT)
        self.geom_names, self.geom_name2id, self._geom_name2id, self._geom_id2name = self._extract_mj_names(self._model.name_geomadr, self._model.ngeom, mujoco.mjtObj.mjOBJ_GEOM)
        self.site_names, self.site_name2id, self._site_name2id, self._site_id2name = self._extract_mj_names(self._model.name_siteadr, self._model.nsite, mujoco.mjtObj.mjOBJ_SITE)
        self.light_names, self.light_name2id, self._light_name2id, self._light_id2name = self._extract_mj_names(self._model.name_lightadr, self._model.nlight, mujoco.mjtObj.mjOBJ_LIGHT)
        self.camera_names, self.camera_name2id, self._camera_name2id, self._camera_id2name = self._extract_mj_names(self._model.name_camadr, self._model.ncam, mujoco.mjtObj.mjOBJ_CAMERA)
        self.actuator_names, self.actuator_name2id, self._actuator_name2id, self._actuator_id2name = self._extract_mj_names(self._model.name_actuatoradr, self._model.nu, mujoco.mjtObj.mjOBJ_ACTUATOR)
        self.sensor_names, self.sensor_name2id, self._sensor_name2id, self._sensor_id2name = self._extract_mj_names(self._model.name_sensoradr, self._model.nsensor, mujoco.mjtObj.mjOBJ_SENSOR)
        self.tendon_names, self.tendon_name2id, self._tendon_name2id, self._tendon_id2name = self._extract_mj_names(self._model.name_tendonadr, self._model.ntendon, mujoco.mjtObj.mjOBJ_TENDON)
        self.mesh_names, self.mesh_name2id, self._mesh_name2id, self._mesh_id2name = self._extract_mj_names(self._model.name_meshadr, self._model.nmesh, mujoco.mjtObj.mjOBJ_MESH)
    
    def get_joint_qpos_addr(self, name):
        '''
        Returns the qpos address for given joint.
        Returns:
        - address (int, tuple): returns int address if 1-dim joint, otherwise
            returns the a (start, end) tuple for pos[start:end] access.
        '''
        joint_id = self._joint_name2id[name]
        joint_type = self._model.jnt_type[joint_id]
        joint_addr = self._model.jnt_qposadr[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            ndim = 7
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            ndim = 4
        else:
            assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            ndim = 1

        if ndim == 1:
            return joint_addr
        else:
            return (joint_addr, joint_addr + ndim)

    actuator_forcerange = ModelProperty()
    actuator_gainprm = ModelProperty()
    actuator_ctrlrange = ModelProperty()
    actuator_biasprm = ModelProperty()
    body_pos = ModelProperty()
    geom_size = ModelProperty()
    body_inertia = ModelProperty()
    geom_friction = ModelProperty()
    site_xpos = ModelProperty()
    site_pos = ModelProperty()
    opt = ModelProperty()
    tendon_range = ModelProperty()
    nv = ModelProperty()
    jnt_range = ModelProperty()
    dof_jntid = ModelProperty()
    dof_damping = ModelProperty()
    body_mass = ModelProperty()
    stat = ModelProperty()
    ncam = ModelProperty()
    geom_rgba = ModelProperty()
    geom_margin = ModelProperty()

    def get_joint_qvel_addr(self, name):
        '''
        Returns the qvel address for given joint.
        Returns:
        - address (int, tuple): returns int address if 1-dim joint, otherwise
            returns the a (start, end) tuple for vel[start:end] access.
        '''
        joint_id = self._joint_name2id[name]
        joint_type = self._model.jnt_type[joint_id]
        joint_addr = self._model.jnt_dofadr[joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            ndim = 6
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            ndim = 3
        else:
            assert joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
            ndim = 1

        if ndim == 1:
            return joint_addr
        else:
            return (joint_addr, joint_addr + ndim)


class MjData(object):
    def __init__(self, data, model):
        self._data = data
        self._model = model

    def data(self):
        return self._data

    def get_joint_qpos(self, name):
        addr = self._model.get_joint_qpos_addr(name)
        if isinstance(addr, (int, np.int32, np.int64)):
            return self.qpos[addr]
        else:
            start_i, end_i = addr
            return self.qpos[start_i:end_i]

    userdata = DataProperty()
    actuator_force = DataProperty()
    time = DataProperty()
    qpos = DataProperty()
    qvel = DataProperty()
    ctrl = DataProperty()
    site_xpos = DataProperty()
    xfrc_applied = DataProperty()
    ncon = DataProperty()
    contact = DataProperty()
    geom_xpos = DataProperty()
    solver_iter = DataProperty()

    def get_site_xpos(self, name):
        id = self._model.site_name2id(name)
        return self._data.site_xpos[id]
        

class MujocoXML:
    """
    Class that combines multiple MuJoCo XML files into a single one.
    """

    meshdir = os.path.join(ASSETS_DIR, "stls")
    texturedir = os.path.join(ASSETS_DIR, "textures")

    TEXTURE_ATTRIBUTES = [
        "file",
        "fileback" "filedown",
        "filefront",
        "fileleft",
        "fileright",
        "fileup",
    ]

    NAMED_FIELDS = {
        "actuator",
        "body1",
        "body2",
        "childclass",
        "class",
        "geom",
        "geom1",
        "geom2",
        "joint",
        "joint1",
        "joint2",
        "jointparent",
        "material",
        "mesh",
        "name",
        "sidesite",
        "site",
        "source",
        "target",
        "tendon",
        "texture",
    }

    ###############################################################################################
    # CONSTRUCTION
    @classmethod
    def parse(cls, xml_filename: str):
        """ Parse given xml filename into the MujocoXML model """

        xml_full_path = os.path.join(XML_DIR, xml_filename)
        if not os.path.exists(xml_full_path):
            raise Exception(xml_full_path)

        with open(xml_full_path) as f:
            xml_root = et.parse(f).getroot()

        xml = cls(xml_root)
        xml.load_includes(os.path.dirname(os.path.abspath(xml_full_path)))
        return xml

    @classmethod
    def from_string(cls, contents: str):
        """ Construct MujocoXML from string """
        xml_root = et.XML(contents)
        xml = cls(xml_root)
        xml.load_includes()
        return xml

    def __init__(self, root_element: typing.Optional[et.Element] = None):
        """ Create new MujocoXML class """
        # This is the root element of the XML document we'll be modifying
        if root_element is None:
            # Create empty root element
            self.root_element = et.Element("mujoco")
        else:
            # Initialize it from the existing thing
            self.root_element = root_element

    ###############################################################################################
    # COMBINING MUJOCO ELEMENTS
    def add_default_compiler_directive(self):
        """ Add a default compiler directive """
        self.root_element.append(
            et.Element(
                "compiler",
                {
                    "meshdir": self.meshdir,
                    "texturedir": self.texturedir,
                    "angle": "radian",
                    "coordinate": "local",
                },
            )
        )

        return self

    def append(self, other: "MujocoXML"):
        """ Append another XML object to this object """
        self.root_element.extend(other.root_element)
        return self

    def xml_string(self):
        """ Return combined XML as a string """
        return et.tostring(self.root_element, encoding="unicode", method="xml")

    def load_includes(self, include_root=""):
        """
        Some mujoco files contain includes that need to be process on our side of the system
        Find all elements that have an 'include' child
        """
        for element in self.root_element.findall(".//include/.."):
            # Remove in a second pass to avoid modifying list while iterating it
            elements_to_remove_insert = []

            for idx, subelement in enumerate(element):
                if subelement.tag == "include":
                    # Branch off initial filename
                    include_path = os.path.join(include_root, subelement.get("file"))

                    include_element = MujocoXML.parse(include_path)

                    elements_to_remove_insert.append(
                        (idx, subelement, include_element.root_element)
                    )

            # Iterate in reversed order to make sure indices are not screwed up
            for idx, to_remove, to_insert in reversed(elements_to_remove_insert):
                element.remove(to_remove)
                to_insert_list = list(to_insert)

                # Insert multiple elements
                for i in range(len(to_insert)):
                    element.insert(idx + i, to_insert_list[i])

        return self

    def _resolve_asset_paths(self, meshdir, texturedir):
        """Resolve relative asset path in xml to local file path."""
        for mesh in self.root_element.findall(".//mesh"):
            fname = mesh.get("file")

            if fname is not None:
                if fname[0] != "/":
                    fname = os.path.join(meshdir or self.meshdir, fname)

                mesh.set("file", fname)

        for texture in self.root_element.findall(".//texture"):
            for attribute in self.TEXTURE_ATTRIBUTES:
                fname = texture.get(attribute)

                if fname is not None:
                    if fname[0] != "/":
                        fname = os.path.join(texturedir or self.texturedir, fname)

                    texture.set(attribute, fname)

    def build(self, output_filename=None, meshdir=None, texturedir=None, **kwargs):
        """ Build and return a mujoco simulation """
        self._resolve_asset_paths(meshdir, texturedir)

        xml_string = self.xml_string()

        if output_filename is not None:
            with open(output_filename, "wt") as f:
                f.write(xml_string)

        mj_model = MjModel(mujoco.MjModel.from_xml_string(xml_string))
        mj_data = MjData(mujoco.MjData(mj_model.model()), mj_model)
        return MjSim(mj_model, mj_data, **kwargs)

    ###############################################################################################
    # MODIFICATIONS
    def set_objects_attr(self, tag: str = "*", **kwargs):
        """ Set given attribute to all instances of given tag within the tree """
        for element in self.root_element.findall(".//{}".format(tag)):
            for name, value in kwargs.items():
                if isinstance(value, (list, np.ndarray)):
                    value = _format_array(value)

                element.set(name, str(value))

        return self

    def set_objects_attrs(self, tag_args: dict):
        """
        Batch version of set_objects_attr where args for multiple tags can be specified as a dict.
        """
        for tag, args in tag_args.items():
            self.set_objects_attr(tag=tag, **args)

    def set_named_objects_attr(self, name: str, tag: str = "*", **kwargs):
        """ Sets xml attributes of all objects with given name """
        for element in self.root_element.findall(".//{}[@name='{}']".format(tag, name)):
            for name, value in kwargs.items():
                if isinstance(value, (list, np.ndarray)):
                    value = _format_array(value)

                element.set(name, str(value))

        return self

    def set_prefixed_objects_attr(self, prefix: str, tag: str = "*", **kwargs):
        """ Sets xml attributes of all objects with given name prefix """
        for element in self.root_element.findall(".//{}[@name]".format(tag)):
            if element.get("name").startswith(prefix):  # type: ignore
                for name, value in kwargs.items():
                    if isinstance(value, (list, np.ndarray)):
                        value = _format_array(value)

                    element.set(name, str(value))

        return self

    def add_name_prefix(self, name_prefix: str, exclude_attribs=[]):
        """
        Add a given name prefix to all elements with "name" attribute.

        Additionally, once we changed all "name" attributes we also have to change all
        attribute fields that refer to those names.
        """

        for element in self.root_element.iter():
            for attrib_name in element.keys():
                if (
                    attrib_name not in self.NAMED_FIELDS
                    or attrib_name in exclude_attribs
                ):
                    continue

                element.set(attrib_name, name_prefix + element.get(attrib_name))  # type: ignore

        return self

    def replace_name(self, old_name: str, new_name: str, exclude_attribs=[]):
        """
        Replace an old name string with an new name string in "name" attribute.
        """
        for element in self.root_element.iter():
            for attrib_name in element.keys():
                if (
                    attrib_name not in self.NAMED_FIELDS
                    or attrib_name in exclude_attribs
                ):
                    continue

                element.set(attrib_name, element.get(attrib_name).replace(old_name, new_name))  # type: ignore

        return self

    def remove_objects_by_tag(self, tag: str):
        """ Remove objects with given tag from XML """
        for element in self.root_element.findall(".//{}/..".format(tag)):
            for subelement in list(element):
                if subelement.tag != tag:
                    continue
                assert subelement.tag == tag
                element.remove(subelement)
        return self

    def remove_objects_by_prefix(self, prefix: str, tag: str = "*"):
        """ Remove objects with given name prefix from XML """
        for element in self.root_element.findall(".//{}[@name]/..".format(tag)):
            for subelement in list(element):
                if subelement.get("name").startswith(prefix):  # type: ignore
                    element.remove(subelement)

        return self

    def remove_objects_by_name(
        self, names: typing.Union[typing.List[str], str], tag: str = "*"
    ):
        """ Remove object with given name from XML """
        if isinstance(names, str):
            names = [names]

        for name in names:
            for element in self.root_element.findall(
                ".//{}[@name='{}']/..".format(tag, name)
            ):
                for subelement in list(element):
                    if subelement.get("name") == name:
                        element.remove(subelement)

        return self
