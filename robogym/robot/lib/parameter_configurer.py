from typing import Dict


class ParameterConfigurer:
    def parameter_bounds(self, actuator: str):
        """Get valid parameter bounds for the given actuator."""
        pass

    def current_parameters(self, actuator: str) -> Dict[str, float]:
        """Get current parameters for the given actuator."""
        pass

    def set_parameters(self, actuator: str, assignments: Dict[str, float]):
        """Set parameters for the given actuator."""
        pass

    def export_parameters(self):
        """Export current parameters.

        For example, save parameters to XML or JSON configuration file."""
        pass
