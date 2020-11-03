import attr

from robogym.robot_env import build_nested_attr


def test_build_nested_attr():
    @attr.s(auto_attribs=True)
    class NestedParameters:
        a: int = 0
        b: int = 1

    @attr.s(auto_attribs=True)
    class Parameters:
        nested: NestedParameters = build_nested_attr(NestedParameters)

    @attr.s(auto_attribs=True)
    class ParametersOverload(Parameters):
        nested: NestedParameters = build_nested_attr(
            NestedParameters, default=dict(a=2)
        )

    parameters = Parameters()
    assert isinstance(parameters.nested, NestedParameters)
    assert parameters.nested.a == 0
    assert parameters.nested.b == 1

    parameters = Parameters(nested={"a": 2})
    assert isinstance(parameters.nested, NestedParameters)
    assert parameters.nested.a == 2
    assert parameters.nested.b == 1

    parameters = ParametersOverload()
    assert parameters.nested.a == 2
    assert parameters.nested.b == 1

    parameters = ParametersOverload(nested={"a": 3})
    assert parameters.nested.a == 3
    assert parameters.nested.b == 1

    parameters = ParametersOverload(nested={"b": 3})
    assert parameters.nested.a == 2
    assert parameters.nested.b == 3
