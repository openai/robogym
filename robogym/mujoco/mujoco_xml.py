import os.path
import typing
import xml.etree.ElementTree as et

import mujoco_py
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


class MjSim(mujoco_py.MjSim):
    """
    There are environments e.g. rearrange environment which recreates
    sim after reach env reset. This can cause potential bugs caused by
    other components still caching instance of old sim. These bugs are usually
    quite tricky to find. This class makes it easier to find these bugs by allowing
    invalidating the sim instance so any access to properties of stale sim instance
    will cause error.
    """

    __slots__ = ("_stale", "_xml")

    def __init__(self, model, **kwargs):
        # Note: we don't need to call super.__init__ because MjSim use __cinit__
        # for initialization which happens automatically before subclass __init__
        # is called.
        self._stale: bool = False
        self._xml = model.get_xml()

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
        return super().data

    @property
    def model(self):
        self._ensure_not_stale()
        return super().model

    def _ensure_not_stale(self):
        if self._stale:
            raise StaleMjSimError(
                "You are accessing property of a stale sim instance which is no longer used"
                "by the environment."
            )


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

        mj_model = mujoco_py.load_model_from_xml(xml_string)
        return MjSim(mj_model, **kwargs)

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
