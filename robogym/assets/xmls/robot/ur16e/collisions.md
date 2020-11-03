# Collisions

Mujoco uses the following bitwise equality for determining collisions between object 1 and object 2:

`(contype1 & conaffinity2) || (contype2 & conaffinity1)`

Below is a table of contype and conaffinity for all classes and geoms used in the env:



| Class/Name | Contype, Conaffinity | Description
| --- | :---: | --- |
| ur16e_viz                     | 0x3, 0x8 | Collide with everything except itself.
| robot0:J4                     | 0x1, 0x1 | 
| robot0:J5                     | 0x1, 0x1 | 
| gripper_viz                   | 0x3, 0x2 | Collides with everything except `gripper_base_geom_v`.
| table_viz                     | 0x1, 0x1 |
| backdrop_viz                  | 0x1, 0x1 |
| table_collision_plane_viz     | 0x0, 0x2 | Collides with `ur16e_viz`, `gripper_viz`, and `wrist_cam_collision_viz`.
| wrist_cam_collision_viz       | 0x2, 0x1 | Collides with everything including `table_collision_plane_viz`.
| gripper_base_geom_v           | 0x1, 0x1 | Collides with everything except `gripper_viz` and itself.



