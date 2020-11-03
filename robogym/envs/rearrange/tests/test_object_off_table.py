import pytest

from robogym.envs.rearrange.composer import make_env


class ObjectOffTableTest:
    def __init__(self):
        env = make_env()
        env.reset()
        self.sim = env.unwrapped.mujoco_simulation

    @pytest.mark.parametrize(
        "object_positions,expected_off_table",
        [
            ([[1.3, 0.75, 0.4]], [False]),
            ([[1.05, 0.4, 0.4]], [False]),
            ([[1.05, 1.1, 0.4]], [False]),
            ([[1.55, 0.4, 0.4]], [False]),
            ([[1.55, 1.1, 0.4]], [False]),
        ],
    )
    def test_single_obj_on_table(self, object_positions, expected_off_table):
        assert expected_off_table == self.sim.check_objects_off_table(object_positions)

    @pytest.mark.parametrize(
        "object_positions,expected_off_table",
        [
            ([[1.3, 0.75, 0.299]], [True]),
            ([[1.05, 0.4, 0.299]], [True]),
            ([[1.05, 1.1, 0.299]], [True]),
            ([[1.55, 0.4, 0.299]], [True]),
            ([[1.55, 1.1, 0.299]], [True]),
        ],
    )
    def test_single_obj_under_table(self, object_positions, expected_off_table):
        assert expected_off_table == self.sim.check_objects_off_table(object_positions)

    @pytest.mark.parametrize(
        "object_positions,expected_off_table",
        [
            ([[-0.1 + 1.05, 0.4, 0.4]], [True]),
            ([[1.05, -0.1 + 0.4, 0.4]], [True]),
            ([[-0.1 + 1.05, 1.1, 0.4]], [True]),
            ([[1.05, +0.1 + 1.1, 0.4]], [True]),
            ([[+0.1 + 1.55, 0.4, 0.4]], [True]),
            ([[1.55, -0.1 + 0.4, 0.4]], [True]),
            ([[+0.1 + 1.55, 1.1, 0.4]], [True]),
            ([[1.55, +0.1 + 1.1, 0.4]], [True]),
        ],
    )
    def test_single_obj_outside_table_hor(self, object_positions, expected_off_table):
        assert expected_off_table == self.sim.check_objects_off_table(object_positions)

    @pytest.mark.parametrize(
        "object_positions,expected_off_table",
        [
            ([[-0.1 + 1.05, 0.4, 0.299]], [True]),
            ([[1.05, -0.1 + 0.4, 0.299]], [True]),
            ([[-0.1 + 1.05, 1.1, 0.299]], [True]),
            ([[1.05, +0.1 + 1.1, 0.299]], [True]),
            ([[+0.1 + 1.55, 0.4, 0.299]], [True]),
            ([[1.55, -0.1 + 0.4, 0.299]], [True]),
            ([[+0.1 + 1.55, 1.1, 0.299]], [True]),
            ([[1.55, +0.1 + 1.1, 0.299]], [True]),
        ],
    )
    def test_single_obj_outside_and_under(self, object_positions, expected_off_table):
        assert expected_off_table == self.sim.check_objects_off_table(object_positions)

    # Multiple objects

    @pytest.mark.parametrize(
        "object_positions,expected_off_table",
        [
            ([[1.3, 0.75, 0.4]], [False]),
            ([[1.05, 0.4, 0.4]], [False]),
            ([[1.05, 1.1, 0.4]], [False]),
            ([[1.55, 0.4, 0.4]], [False]),
            ([[1.55, 1.1, 0.4]], [False]),
        ],
    )
    def test_mul_obj_on_table(self, object_positions, expected_off_table):
        object_positions.append([1.3, 0.75, 0.4])
        expected_off_table.append(False)

        assert (
            expected_off_table == self.sim.check_objects_off_table(object_positions)
        ).all()

    @pytest.mark.parametrize(
        "object_positions,expected_off_table",
        [
            ([[1.3, 0.75, 0.299]], [True]),
            ([[1.05, 0.4, 0.299]], [True]),
            ([[1.05, 1.1, 0.299]], [True]),
            ([[1.55, 0.4, 0.299]], [True]),
            ([[1.55, 1.1, 0.299]], [True]),
        ],
    )
    def test_mul_obj_under_table(self, object_positions, expected_off_table):
        object_positions.append([1.3, 0.75, 0.4])
        expected_off_table.append(False)

        assert (
            expected_off_table == self.sim.check_objects_off_table(object_positions)
        ).all()

    @pytest.mark.parametrize(
        "object_positions,expected_off_table",
        [
            ([[-0.1 + 1.05, 0.4, 0.4]], [True]),
            ([[1.05, -0.1 + 0.4, 0.4]], [True]),
            ([[-0.1 + 1.05, 1.1, 0.4]], [True]),
            ([[1.05, +0.1 + 1.1, 0.4]], [True]),
            ([[+0.1 + 1.55, 0.4, 0.4]], [True]),
            ([[1.55, -0.1 + 0.4, 0.4]], [True]),
            ([[+0.1 + 1.55, 1.1, 0.4]], [True]),
            ([[1.55, +0.1 + 1.1, 0.4]], [True]),
        ],
    )
    def test_mul_obj_outside_table_hor(self, object_positions, expected_off_table):
        object_positions.append([1.3, 0.75, 0.4])
        expected_off_table.append(False)

        assert (
            expected_off_table == self.sim.check_objects_off_table(object_positions)
        ).all()

    @pytest.mark.parametrize(
        "object_positions,expected_off_table",
        [
            ([[-0.1 + 1.05, 0.4, 0.299]], [True]),
            ([[1.05, -0.1 + 0.4, 0.299]], [True]),
            ([[-0.1 + 1.05, 1.1, 0.299]], [True]),
            ([[1.05, +0.1 + 1.1, 0.299]], [True]),
            ([[+0.1 + 1.55, 0.4, 0.299]], [True]),
            ([[1.55, -0.1 + 0.4, 0.299]], [True]),
            ([[+0.1 + 1.55, 1.1, 0.299]], [True]),
            ([[1.55, +0.1 + 1.1, 0.299]], [True]),
        ],
    )
    def test_mul_obj_outside_and_under(self, object_positions, expected_off_table):
        object_positions.append([1.3, 0.75, 0.4])
        expected_off_table.append(False)

        assert (
            expected_off_table == self.sim.check_objects_off_table(object_positions)
        ).all()
