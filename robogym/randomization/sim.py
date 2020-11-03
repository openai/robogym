import abc
import copy
from typing import List, Union

import numpy as np
from mujoco_py import MjSim
from numpy.random import RandomState

from robogym.mujoco.constants import OPT_FIELDS, PID_GAIN_PARAMS
from robogym.randomization.common import Randomizer
from robogym.randomization.parameters import (
    FloatRandomizerParameter,
    RandomizerParameter,
)
from robogym.utils.rotation import random_unity2


def has_prefixes(string, prefixes):
    if isinstance(prefixes, list):
        for p in prefixes:
            if has_prefixes(string, p):
                return True
        return False
    else:
        return string.startswith(prefixes)


class SimulationRandomizer(Randomizer[MjSim], abc.ABC):
    """
    Randomizer which randomize randomization.
    """

    def __init__(self, name):
        super().__init__(name, enabled=True)
        self.sim = None
        self._initial_value = None
        self._params = self._prepare_randomizer_params()

    def initialize(self, sim: MjSim):
        """
        Initialize state for the randomizer. This method will be called
        every time a different sim instance is passed in.
        """
        self.sim = sim
        self._initialize()

    def _initialize(self):
        """
        Add additional initialization logic.
        """
        pass

    def _randomize(self, target: MjSim, random_state: RandomState):
        if target != self.sim:
            self.initialize(target)

        self._randomize_sim(random_state)
        return self.sim

    @abc.abstractmethod
    def _randomize_sim(self, random_state: RandomState):
        """
        Implement this method to apply randomization to self.sim
        """
        pass

    def _register_sim_parameter(
        self,
        name="value",
        initial_value=0.0,
        value_min=-4.0,
        value_max=4.0,
        delta=None,
    ):
        if delta is None:
            delta = self._default_delta(value_min, value_max)

        return self.register_parameter(
            FloatRandomizerParameter(
                name,
                initial_value=initial_value,
                value_range=(value_min, value_max),
                delta=delta,
            )
        )

    @staticmethod
    def _default_delta(value_min, value_max):
        """
        If no delta is provided for given parameter, this method will be called to get
        default delta.
        """
        return None

    @abc.abstractmethod
    def _prepare_randomizer_params(
        self,
    ) -> Union[RandomizerParameter, List[RandomizerParameter]]:
        """
        Return all randomizer parameters associated with this randomizer.
        """
        pass

    @property
    def _randomizer_param_values(self) -> Union[float, np.ndarray]:
        """
        Get numerical values for all randomizer parameters.
        """
        if isinstance(self._params, RandomizerParameter):
            return self._params.get_value()
        else:
            return np.array([param.get_value() for param in self._params])


class GravityRandomizer(SimulationRandomizer):
    def __init__(self):
        super().__init__("gravity")

    def _initialize(self):
        self._initial_value = self.sim.model.opt.gravity.copy()

    def _prepare_randomizer_params(self):
        return self._register_sim_parameter(value_min=0.0)

    def _randomize_sim(self, random_state: RandomState):
        direction = random_unity2(random_state)
        mag = np.exp(self._randomizer_param_values) - 1.0
        noise = direction * 1.0 * mag
        self.sim.model.opt.gravity[:] = self._initial_value + noise

    @staticmethod
    def _default_delta(value_min, value_max):
        return (value_max - value_min) / 10


class PidRandomizer(SimulationRandomizer):
    def __init__(self, field_name):
        super().__init__(field_name)
        self._idx = PID_GAIN_PARAMS.index(field_name)

    def _initialize(self):
        self._initial_value = copy.deepcopy(
            self.sim.model.actuator_gainprm[:, self._idx]
        )

    def _prepare_randomizer_params(self):
        return [
            self._register_sim_parameter("mean"),
            self._register_sim_parameter("std", value_min=0.0),
        ]

    def _randomize_sim(self, random_state: RandomState):
        values = self._randomizer_param_values
        assert isinstance(values, np.ndarray)

        self.sim.model.actuator_gainprm[:, self._idx] = self._initial_value * np.exp(
            random_state.normal(
                values[0], scale=abs(values[1]), size=self._initial_value.shape
            )
        )


class JointMarginRandomizer(SimulationRandomizer):
    def __init__(self):
        super().__init__("jnt_margin")

    def _initialize(self):
        self._initial_value = copy.deepcopy(self.sim.model.jnt_margin)

    def _prepare_randomizer_params(self):
        return self._register_sim_parameter(value_min=0.0)

    def _randomize_sim(self, random_state: RandomState):
        new_values = self._initial_value + (
            random_state.uniform(size=self._initial_value.shape)
            * (np.exp(self._randomizer_param_values) - 1.0)
            * 0.15
        )
        self.sim.model.jnt_margin[:] = new_values


class GeomSolimpRandomizer(SimulationRandomizer):

    PARAMETER_NAMES = [
        "dmax_mean",
        "dmax_std",
        "delta_mean",
        "delta_std",
        "width_mean",
        "width_std",
    ]

    def __init__(self, drange=(0.5, 0.99)):
        assert len(drange) == 2
        super().__init__("geom_solimp")

        self._drange = drange
        self.parameters_shape = [6]
        self.parameter_names = self.PARAMETER_NAMES
        self.positive = [False, True, False, True, False, True]

    def _initialize(self):
        # Only take first three parameters
        self._initial_value = copy.deepcopy(self.sim.model.geom_solimp[:, :3])
        assert self._initial_value.shape[1] == 3

    def _prepare_randomizer_params(self):
        params = []
        for i in range(0, len(self.PARAMETER_NAMES), 2):
            params.extend(
                [
                    self._register_sim_parameter(name=self.PARAMETER_NAMES[i]),
                    self._register_sim_parameter(
                        name=self.PARAMETER_NAMES[i + 1], value_min=0.0
                    ),
                ]
            )

        return params

    def _randomize_sim(self, random_state: RandomState):
        values = self._randomizer_param_values
        assert isinstance(values, np.ndarray)

        dmax_mean, dmax_std, delta_mean, delta_std, width_mean, width_std = values
        assert dmax_std >= 0.0
        assert delta_std >= 0.0
        assert width_std >= 0.0

        # We randomize (1-dmax) since dmax typically very close to 1 and we'd like to cover the
        # range [0, 1] well. We then sample delta that is subtracted from dmax to produce dmin,
        # thus ensuring that dmin <= dmax holds.
        dmax = 1.0 - (1.0 - self._initial_value[:, 1]) * np.exp(
            random_state.normal(
                dmax_mean, scale=dmax_std, size=self._initial_value.shape[0]
            )
        )
        dmax = np.clip(dmax, *self._drange)
        delta = (self._initial_value[:, 1] - self._initial_value[:, 0]) * np.exp(
            random_state.normal(
                delta_mean, scale=delta_std, size=self._initial_value.shape[0]
            )
        )
        dmin = np.clip(dmax - delta, *self._drange)

        # Sample width.
        width = self._initial_value[:, 2] * np.exp(
            random_state.normal(
                width_mean, scale=width_std, size=self._initial_value.shape[0]
            )
        )

        # Validate constraints. Mujoco internally already ensures that dmin and dmax are clipped,
        # if necessary (http://mujoco.org/book/modeling.html#CSolver), but we enforce slightly
        # stronger constraints for additional stability.
        assert dmin.shape == dmax.shape == width.shape
        assert (dmin <= dmax).all()
        assert (self._drange[0] <= dmin).all()
        assert (dmin <= self._drange[1]).all()
        assert (self._drange[0] <= dmax).all()
        assert (dmax <= self._drange[1]).all()

        self.sim.model.geom_solimp[:, 0] = dmin
        self.sim.model.geom_solimp[:, 1] = dmax
        self.sim.model.geom_solimp[:, 2] = width


class GeomSolrefRandomizer(SimulationRandomizer):

    PARAMETER_NAMES = [
        "timeconst_mean",
        "timeconst_std",
        "dampratio_mean",
        "dampratio_std",
    ]

    def __init__(self):
        super().__init__("geom_solref")

    def _initialize(self):
        self._initial_value = copy.deepcopy(self.sim.model.geom_solref)

    def _prepare_randomizer_params(self):
        params = []
        for i in range(0, len(self.PARAMETER_NAMES), 2):
            params.extend(
                [
                    self._register_sim_parameter(name=self.PARAMETER_NAMES[i]),
                    self._register_sim_parameter(
                        name=self.PARAMETER_NAMES[i + 1], value_min=0.0
                    ),
                ]
            )

        return params

    def _randomize_sim(self, random_state: RandomState):
        values = self._randomizer_param_values
        assert isinstance(values, np.ndarray)

        timeconst_mean, timeconst_std, dampratio_mean, dampratio_std = values
        assert timeconst_std >= 0.0
        assert dampratio_std >= 0.0

        self.sim.model.geom_solref[:, 0] = self._initial_value[:, 0] * np.exp(
            random_state.normal(
                timeconst_mean, scale=timeconst_std, size=self._initial_value.shape[0]
            )
        )
        self.sim.model.geom_solref[:, 1] = self._initial_value[:, 1] * np.exp(
            random_state.normal(
                dampratio_mean, scale=dampratio_std, size=self._initial_value.shape[0]
            )
        )


class GenericSimRandomizer(SimulationRandomizer):
    def __init__(
        self,
        name,
        field_name,
        apply_mode="uncoupled_mean_variance",
        coef=1.0,
        geom_prefix=None,
        body_prefix=None,
        dof_jnt_prefix=None,
        jnt_prefix=None,
        positive_only=False,
        zero_threshold=0.0,
    ):
        """
        Generic randomizer for mujoco fields.

        :param field_name: name of the field to randomize (there must be a field in `sim.model`
            or `sim.model.opt` with the given name)
        :param apply_mode: specifies how to apply environment parameters to environment 'sample'
            samples environments based on the distribution defined by the environment parameters
            and 'set' applys environment parameters directly to the environment.
        :param coef: a scalar by which environment parameters are multiplied before being applied
        :param geom_prefix: If not None then this randomizer will only affect the geoms
            that has this prefix
        :param dof_jnt_prefix: If not None then this randomizer will only affect the
            DOFs that are associated with a joint with this prefix.
        :param jnt_prefix: If not None then this randomizer will only affect the
            joints that have this prefix.
        :param positive_only: If True, then the given mujoco field will only be
            set to positive values.
        :param zero_threshold: Maximum fraction of original values that are allowed to be zero,
            only applicable to multiplicative modes.
        """
        self._apply_mode = apply_mode
        super().__init__(name)

        self._field_name = field_name
        self._is_opt = field_name in OPT_FIELDS
        self._coef = coef
        self._positive_only = positive_only
        self._geom_prefix = geom_prefix
        self._body_prefix = body_prefix
        self._dof_jnt_prefix = dof_jnt_prefix
        self._jnt_prefix = jnt_prefix
        self._zero_threshold = zero_threshold
        self._ids = None

    def _initialize(self):
        self._ids = self.identify_fields(
            self._geom_prefix,
            self._body_prefix,
            self._dof_jnt_prefix,
            self._jnt_prefix,
        )

        self._initial_value = copy.deepcopy(self.get_params())

        self.multiplicative_mode_sanity_check(self._zero_threshold)

    def _prepare_randomizer_params(
        self,
    ) -> Union[RandomizerParameter, List[RandomizerParameter]]:
        if self._apply_mode in (
            "coupled",
            "uncoupled",
            "coupled_mean_variance",
            "max_additive",
        ):
            params = self._register_sim_parameter()
        elif self._apply_mode in (
            "coupled_additive",
            "coupled_symmetric_ranges",
            "variance",
            "variance_additive",
        ):
            params = self._register_sim_parameter(value_min=0.0)
        elif self._apply_mode in ("ranges", "coupled_ranges", "semicorrelated"):
            params = [
                self._register_sim_parameter(name="low"),
                self._register_sim_parameter(name="high"),
            ]
        elif self._apply_mode == "variance_mean_additive":
            params = [
                self._register_sim_parameter(name="mean", value_min=0.0),
                self._register_sim_parameter(name="std", value_min=0.0),
            ]
        elif self._apply_mode == "uncoupled_mean_variance":
            params = [
                self._register_sim_parameter(name="mean"),
                self._register_sim_parameter(name="std", value_min=0.0),
            ]
        else:
            raise ValueError("Invalid mode: {}".format(self._apply_mode))

        return params

    @staticmethod
    def _default_delta(value_min, value_max):
        return (value_max - value_min) / 10

    def multiplicative_mode_sanity_check(self, zero_threshold):
        """
        Ensure that multiplicative apply modes are not applied to parameters whose initial
        values are mostly zeros.
        """
        multiplicative_apply_modes = {
            "coupled",
            "uncoupled",
            "ranges",
            "coupled_ranges",
            "semicorrelated",
            "coupled_symmetric_ranges",
            "variance",
            "coupled_mean_variance",
            "uncoupled_mean_variance",
        }
        if self._apply_mode in multiplicative_apply_modes:
            params = self._initial_value
            zeros = np.isclose(params, 0.0).mean()

            assert zeros <= zero_threshold, (
                f"Mode is multiplicative on field {self._field_name}, but too many "
                f"values are zero, maximum fraction allowed is {zero_threshold:.3f} but got "
                f"{zeros:.3f}: {self._initial_value}. If you think that is expected, please "
                f"adjust the zero_threshold value or add an exception above."
            )

    def identify_fields(self, geom_prefix, body_prefix, dof_jnt_prefix, jnt_prefix):
        if geom_prefix is not None:
            assert self._field_name.startswith("geom_")
            geom_names = [
                name
                for name in self.sim.model.geom_names
                if has_prefixes(name, geom_prefix)
            ]
            ids = np.array(
                sorted([self.sim.model.geom_name2id(name) for name in geom_names])
            )
        elif body_prefix is not None:
            assert self._field_name.startswith("body_")
            body_names = [
                name
                for name in self.sim.model.body_names
                if has_prefixes(name, body_prefix)
            ]
            ids = np.array(
                sorted([self.sim.model.body_name2id(name) for name in body_names])
            )
        elif dof_jnt_prefix is not None:

            def has_prefix(jnt_id):
                return has_prefixes(
                    self.sim.model.joint_id2name(jnt_id), dof_jnt_prefix
                )

            assert self._field_name.startswith("dof_")
            ids = [
                idx
                for idx, jnt_id in enumerate(self.sim.model.dof_jntid)
                if has_prefix(jnt_id)
            ]
        elif jnt_prefix is not None:

            def has_prefix(jnt_id):
                return has_prefixes(self.sim.model.joint_id2name(jnt_id), jnt_prefix)

            assert self._field_name.startswith("jnt_")
            ids = [
                idx for idx in range(len(self.sim.model.jnt_type)) if has_prefix(idx)
            ]
        else:
            ids = None

        if ids is not None:
            ids = np.array(sorted(ids))
            assert len(ids) > 0, "no IDs matched for {}".format(self._field_name)
        else:
            ids = None

        return ids

    def __repr__(self):
        return "<{} : {}>".format(self.__class__.__name__, self._field_name)

    def _get_params(self):
        if self._is_opt:
            return getattr(self.sim.model.opt, self._field_name)
        return getattr(self.sim.model, self._field_name)

    def get_params(self):
        out = self._get_params()
        if self._ids is not None:
            return out[self._ids]
        return out

    def set_params(self, new_values):
        v = self._get_params()
        if self._ids is not None:
            v[self._ids] = new_values
        else:
            v[:] = new_values

    def _randomize_sim(self, random_state: RandomState):
        param_value = self._randomizer_param_values * self._coef
        if self._apply_mode == "coupled":
            new_value = self._initial_value * np.exp(param_value)
        elif self._apply_mode == "coupled_additive":
            new_value = self._initial_value + (np.exp(param_value) - 1.0)
        elif self._apply_mode == "uncoupled":
            new_value = self._initial_value * np.exp(
                random_state.normal(param_value, size=self._initial_value.shape)
                * np.absolute(param_value)
            )
        elif self._apply_mode == "ranges":
            low = min(0, -param_value[0])
            high = max(0, param_value[1])
            new_value = self._initial_value * np.exp(
                random_state.uniform(low, high, size=self._initial_value.shape)
            )
        elif self._apply_mode == "coupled_ranges":
            low = min(0, -param_value[0])
            high = max(0, param_value[1])
            new_value = self._initial_value * np.exp(random_state.uniform(low, high))
        elif self._apply_mode == "coupled_symmetric_ranges":
            low = -abs(param_value)
            high = abs(param_value)  # This is intentially domain_param_value
            new_value = self._initial_value * np.exp(
                random_state.uniform(low, high, size=self._initial_value.shape)
            )
        elif self._apply_mode == "variance":
            variance = abs(param_value)
            new_value = self._initial_value * np.exp(
                random_state.normal(0, size=self._initial_value.shape) * variance
            )
        elif self._apply_mode == "variance_additive":
            scale = np.exp(abs(param_value)) - 1.0
            noise = random_state.normal(0, scale=scale, size=self._initial_value.shape)
            new_value = self._initial_value + noise
        elif self._apply_mode == "variance_mean_additive":
            pos = np.exp(param_value[0]) - 1.0
            scale = np.exp(abs(param_value[1])) - 1.0
            noise = np.abs(
                random_state.normal(pos, scale=scale, size=self._initial_value.shape)
            )
            new_value = self._initial_value + noise
        elif self._apply_mode == "coupled_mean_variance":
            new_value = self._initial_value * np.exp(
                random_state.normal(
                    param_value, scale=abs(param_value), size=self._initial_value.shape
                )
            )
        elif self._apply_mode == "uncoupled_mean_variance":
            new_value = self._initial_value * np.exp(
                random_state.normal(
                    param_value[0],
                    scale=abs(param_value[1]),
                    size=self._initial_value.shape,
                )
            )
        elif self._apply_mode == "max_additive":
            high = np.exp(abs(param_value)) - 1.0
            noise = random_state.uniform(
                low=0, high=high, size=self._initial_value.shape
            )
            new_value = self._initial_value + noise
        else:
            raise RuntimeError()

        if self._positive_only:
            new_value = np.clip(new_value, 0, np.inf)

        self.set_params(new_value)
