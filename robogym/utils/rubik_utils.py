import kociemba
import pycuber


def solve_fast(cube, max_depth=24):
    assert isinstance(cube, pycuber.Cube)
    coloring = str(cube).replace("[", "").replace("]", "").replace("   ", " ")
    coloring = coloring.split("\n")
    seq = coloring[0].strip() + coloring[1].strip() + coloring[2].strip()
    seq += coloring[3][6:9]
    seq += coloring[4][6:9]
    seq += coloring[5][6:9]
    seq += coloring[3][3:6]
    seq += coloring[4][3:6]
    seq += coloring[5][3:6]
    seq += coloring[6][3:6]
    seq += coloring[7][3:6]
    seq += coloring[8][3:6]
    seq += coloring[3][:3]
    seq += coloring[4][:3]
    seq += coloring[5][:3]
    seq += coloring[3][9:12]
    seq += coloring[4][9:12]
    seq += coloring[5][9:12]
    seq = seq.replace("y", "U")
    seq = seq.replace("g", "F")
    seq = seq.replace("w", "D")
    seq = seq.replace("r", "L")
    seq = seq.replace("o", "R")
    seq = seq.replace("b", "B")
    return kociemba.solve(seq, max_depth=max_depth)
