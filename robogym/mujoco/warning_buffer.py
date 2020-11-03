import collections
import logging

import mujoco_py.cymj as cymj

logger = logging.getLogger(__name__)


class MujocoErrorException(Exception):
    """ Exception raised when mujoco error is called. """

    pass


def error_callback(message):
    """ Mujoco error callback """
    message = message.decode()
    full_message = f"MUJOCO ERROR: {message}"
    logger.error(full_message)
    raise MujocoErrorException(full_message)


# Set it once for all the processes
cymj.set_error_callback(error_callback)


class MjWarningBuffer:
    """
    Buffering MuJoCo warnings.

    That way they don't cause an exception being thrown which crashes the process,
    but at the same time we store them in memory and can process.

    One can potentially specify buffer capacity if one wants to use a circular buffer.
    """

    def __init__(self, maxlen=None):
        self.maxlen = maxlen

        self._buffer = collections.deque(maxlen=self.maxlen)
        self._prev_user_callback = None

    def _intercept_warning(self, warn_bytes):
        """ Intercept a warning """
        warn = warn_bytes.decode()  # Convert bytes to string

        logger.warning("MUJOCO WARNING: %s", str(warn))

        self._buffer.append(warn)

    @property
    def warnings(self):
        """ Return a list of warnings to the user """
        return list(self._buffer)

    def enter(self):
        """ Enable collecting warnings """
        if self._prev_user_callback is None:
            self._prev_user_callback = cymj.get_warning_callback()

        cymj.set_warning_callback(self._intercept_warning)

    def clear(self):
        """ Reset warning buffer """
        self._buffer.clear()

    def exit(self):
        """ Stop collecting warnings """
        if self._prev_user_callback is not None:
            cymj.set_warning_callback(self._prev_user_callback)
            self._prev_user_callback = None

    def __enter__(self):
        """ Enter - context manager magic method """
        self.enter()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ Exit - context manager magic method """
        self.exit()

    def __repr__(self):
        """ Text representation"""
        return "<{} warnings:{}>".format(self.__class__.__name__, len(self.warnings))
