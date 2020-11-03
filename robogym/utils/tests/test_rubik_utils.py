import unittest

import pycuber

from robogym.utils.rubik_utils import solve_fast


class RubikTest(unittest.TestCase):
    def test_solver(self):
        cube = pycuber.Cube()
        initial_cube = str(cube)
        alg = pycuber.Formula()
        random_alg = alg.random()
        cube(random_alg)
        assert initial_cube != str(cube), "Randomization haven't worked."
        solution = solve_fast(cube)
        print(solution)
        for step in solution.split(" "):
            cube.perform_step(step)
        assert initial_cube == str(cube), "Fast solution doesn't work"
