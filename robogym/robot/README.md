# Robot Control Interface

We provide a general robot control framework that is targeted for position control of a robot arm with a gripper. The library implements control classes for the UR16e robot arm that has 6 actuated joints along with a 1-DOF RobotIQ 2f-85 Gripper.  Robogym rearrange environments can be customized via the `RobotControlParameter` class as described in this section, while dactyl environments use a fixed PID controller to achieve position control of the 20 joints of the ShadowHand robot.

## `RobotControlParameters` specification

`max_position_change`: A parameter that governs the control delta for robot control. Currently, it applies to all the robots within a `CompositeRobot`. RobotControlParameters class implements a helper `default_max_pos_change_for_solver` to initialize each control_mode/solver pair with a reasonable value.

### `ControlMode`
Three modes of robot control are currently supported for the robot arm are as follows:

|Control Mode|Description|Action Dimensions|
|:---|---|---|
|Joint| Joint position control mode, where joints are actuated via PID or Cascaded PI controllers based on the `arm_joint_calibration_path` specification. See [mujoco-py](https://github.com/openai/mujoco-py/blob/master/mujoco_py/mjpid.pyx) for more details on the low-level controller implementation.|6|
|tcp+roll+yaw (default) |Tool center point (TCP) relative position control in local coordinates and 2-DOF rotation control via wrist rotation and wrist tilt. | 5|
|tcp+wrist|TCP relative position control in local coordinates and 1-DOF rotation control via wrist rotation.| 4|


### `TCPSolverMode`

For TCP based control, we provide two solvers to govern robot dynamics.

|TCP Solver Mode|Description|
|:---|---|
|mocap|Control is achieved via the MuJoCo [mocap](http://mujoco.org/book/modeling.html) mechanism, which is used as a simulation based Inverse Kinematics (IK) solver. In this mode, robot joint dynamics cannot be enforced and motion is dictated by the solver parameters of the MuJoCo sim, which may result in high contact forces and simulation instabilities. |
|mocap_ik (default) | This mode is provided for applications that use joint actuated robots that are controllable in the TCP domain. One example of such application would be when a policy is trained to output relative position and rotation actions in the tooltip space, that are physically realized by servoing in the joint space by the robot. In this mode, we use a solver simulation that uses the `mocap` mechanism as described above as an IK solver. The abstract solver interface can also be used to develop an analytical IK solver, however, stability of such solver has been poor in our experience. The positions achieved by this solver simulation are then used as targets to a joint-controlled robot simulation, whose dynamics will be determined by the specific controller.|

## Environment Interaction

Each environment contains a `robot` object accessible via `env.robot` that implements the [`RobotInterface`](robot_interface.py). For environments that require multiple robots such as a robot arm and a gripper, `env.robot` is a [`CompositeRobot`](composite/composite_robot.py) that handles splitting the action space appropriately across different robot implementations.

Below is a list of compatible robot configurations for each environment group.

|Environment Name|Robot Control Parameters|Robot Class|Action Dimension|
|:---|---|---|---|
|`dactyl/*`| N/A | `MuJoCoShadowHand` | 20|
|`rearrange/*`| `{'control_mode': 'joint'}` | `MujocoURJointGripperCompositeRobot` | 7|
|`rearrange/*`| `{'control_mode': 'tcp+roll+yaw', 'tcp_solver_mode':'mocap_ik'}` | `MujocoURTcpJointGripperCompositeRobot` | 6|
|`rearrange/*`| `{'control_mode': 'tcp+wrist', 'tcp_solver_mode':'mocap_ik'}` | `MujocoURTcpJointGripperCompositeRobot` | 5|
|`rearrange/*`| `{'control_mode': 'tcp+roll+yaw', 'tcp_solver_mode':'mocap'}` | `MujocoIdealURGripperCompositeRobot` | 6|
|`rearrange/*`| `{'control_mode': 'tcp+wrist', 'tcp_solver_mode':'mocap'}` | `MujocoIdealURGripperCompositeRobot` | 5|
