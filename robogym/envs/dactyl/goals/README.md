# Goal generation 

In this module we have various modes of goal generation implemented.

Cube alignment means:
- face angles are within threshold from round angles
- one face is within threshold of being oriented upwards


### FaceCurriculumGoal

If the cube is not aligned, we reorient the cube.
If the cube is aligned, we reorient the cube with 25% probability.
    
   
If we reorient the cube:
- Face angle goal is just rounded angles of current face angles
- New orientation is sampled from a distribution constrained one face is pointing up


If we don't reorient the cube:
- Face angle goal is current rounded angles of faces, with top face rotated clockwise or
 counterclockwise with 50% probability
- Orientation goal is initially selected cube orientation rounded to closes round euler angles


### FaceFreeRotationGoal

In this case, we are not constraining cube rotation along the z axis during face rotation,
but we still constrain selected face to be directed upwards.


If the cube is not aligned, we reorient the cube.
If the cube is aligned, we reorient the cube with 25% probability.

    
If we reorient the cube:
- Face angle goal is just rounded angles of current face angles
- New orientation is sampled uniformly from a distribution constrained that one face is pointing up


If we don't reorient the cube:
- Face angle goal is current rounded angles of faces, with top face rotated clockwise or
 counterclockwise with 50% probability
- Target orientation is current (as in different in every frame) cube orientation
  aligned so that initially selected rotate pointing upwards

### FullUnconstrainedGoal

Goal generation is not constrained in this solver besides face angle alignment criteria.
- Face angle goal is current rounded angles of faces, with any face rotated clockwise or
 counterclockwise with uniform probability
- No orientation goal is set

### FaceCubeSolver

Utilizes a Rubik's cube solver to generate FaceFree style goals with the structure that the goal
sequence is particularly selected to solve a Rubik's cube.

### UnconstrainedCubeSolver

Utilizes a Rubik's cube solver to generate FullUnconstrainedGoal style goals with the structure 
that the goal sequence is particularly selected to solve a Rubik's cube.
 