from typing import List, Optional

import numpy as np


class RegraspHelper:
    """
    A helper that attempts to facilitate persistent object grasping by a gripper. When the helper
    detects backdrive due to neutral commands preceded by a close command on the gripper, it will
    re-issue the previous command that is deemed as more desirable than the backdrive result to
    prevent objects slipping due to incomplete grasping control.
    """

    def __init__(self, initial_position: np.ndarray):
        self.regrasp_command = None  # current command that we issue so that we re-grasp
        self.prev_obs_position = (
            initial_position  # previous joint observation (=current val)
        )
        self.last_nonzero_cmd_direction = None  # last user desired trajectory
        self.last_nonzero_obs_direction = None  # last actual trajectory
        self.prev_action = None  # last command
        self.second_prev_action = None  # second to last command

        self.debug_regrasp = (
            False  # set this to True to print debug information for re-grasping
        )
        if self.debug_regrasp:
            self.debug_desired_action_history: List[float] = []
            self.debug_desired_action_dir_history: List[str] = []
            self.debug_observed_pos_history: List[float] = []
            self.debug_observed_pos_dir_history: List[str] = []
            self.debug_returned_ctrl_history: List[np.ndarray] = []

    @staticmethod
    def debug_dir_to_string(direction: Optional[float]):
        """For debugging only, given a float representing a direction, return a human friendly string.

        :param direction: Positive for Closing, Negative for Opening and zero for Keeping. None is also accepted.
        :return: String representation of the float interpreted as a direction.
        """
        if direction is None:
            return "None"
        elif direction == 0:
            return "Keep"
        elif direction > 0:
            return "Close"
        elif direction < 0:
            return "Open"

    @staticmethod
    def debug_add_to_history(
        history: List, item: object, max_history: float = 20
    ) -> None:
        """For debugging only. Adds the given item to the given list, and removes the first object if the list
        length becomes bigger than the max history limit. This method will pop at most one item, so it expects
        max history limit to be constant.

        :param history: List to add the item.
        :param item: Item to add.
        :param max_history: Limit for the list. This method will pop one item if the list exceeds this limit after
        adding the item.
        """
        history.append(item)
        if len(history) > max_history:
            history.pop(0)

    def debug_print_regrasp_history(self) -> None:
        """Pretty print the history of debug variables that help debug regrasp."""
        print("- - - - -")
        print(
            f"DesCmdHist   : {['{0:0.5f}'.format(i) for i in self.debug_desired_action_history]}"
        )
        print(f"DesCmdDirHist: {self.debug_desired_action_dir_history}")
        print(
            f"ObsPosHist   : {['{0:0.5f}'.format(i) for i in self.debug_observed_pos_history]}"
        )
        print(f"ObsPosDirHist: {self.debug_observed_pos_dir_history}")
        print(
            f"PrevReturns  : {['{0:0.5f}'.format(i) for i in self.debug_returned_ctrl_history]}"
        )

    def compute_regrasp_control(
        self,
        position_control: np.ndarray,
        default_control: np.ndarray,
        current_position: np.ndarray,
    ) -> np.ndarray:
        """
        Computes control override if applicable given the current state of gripper and controls
        :param position_control: Applied absolute position control
        :param default_control: Computed default denormalized control that would apply without re-grasp correction
        :param current_position: Current gripper joint position reading
        :return: re-grasp corrected control
        """
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 1) Compute variables that will help us make the re-grasp decision
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # note some of these variables are not used, but its code is reflected for completeness purposes

        # current position delta
        assert self.prev_obs_position is not None
        obs_move = current_position - self.prev_obs_position
        obs_direction = (
            0.0 if np.allclose(obs_move, 0, atol=1e-5) else np.sign(obs_move)
        )

        # what does the user want to do now?
        user_wants_to_open = position_control < 0.0
        user_wants_to_close = position_control > 0.0
        user_wants_to_keep = position_control == 0.0

        # what did the user want to do last
        # user_last_trajectory_was_opening = self.last_nonzero_cmd_direction and self.last_nonzero_cmd_direction < 0.0
        user_last_trajectory_was_closing = (
            self.last_nonzero_cmd_direction and self.last_nonzero_cmd_direction > 0.0
        )

        # what is the gripper doing now? (note this is influenced by the previous command, not the desired command)
        gripper_is_opening = obs_direction < 0.0
        # gripper_is_closing = obs_direction > 0.0
        # gripper_is_still = obs_direction == 0.0

        # what was the gripper last trajectory
        # gripper_was_opening_or_still = self.last_nonzero_obs_direction and self.last_nonzero_obs_direction < 0.0
        gripper_was_closing_or_still = (
            self.last_nonzero_obs_direction and self.last_nonzero_obs_direction > 0.0
        )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 2) If we are currently regrasping, we have special handling to do first
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        is_regrasping = self.regrasp_command is not None
        if is_regrasping:

            if user_wants_to_open:
                # stop re-grasping
                self.regrasp_command = None

            elif user_wants_to_close:
                # if the user wants to close, let the code continue down. The default behavior for the algorithm
                # will compute the user desired control, and compare it to the regrasp, actually enacting the
                # user one if re-grasping would have been a worse command (in terms of closing the gripper)
                pass

            else:
                # directly continue re-issuing the same command we decided when we started re-grasping
                return self.regrasp_command

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 3) This is where we decide if we should re-grasp
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # we want to regrasp if all of these are true:
        #  1) the user wants to close or keep the position
        #  2) the last thing the user wanted to do was closing the gripper (not opening). Being still is ignored here *
        #  3) the gripper was indeed closing or staying still after closing
        #  4) the gripper is now opening
        # Since it was closing, and now opening, and the user wants to close or keep closed, and the gripper did already
        # try to close, the fact that everything says that it should be closing, but it is actually opening, hints
        # at an external force trying to open it, and we asumme here that is backdrive.
        #
        # * note on (2). The reason why this matters is because if the user wanted to open, we would expect the gripper
        # to open some time after that user command (typically 1-n ticks later depending on current momementum.) Just
        # because the user wants to close now we can't expect the gripper to close. In order to expect the gripper
        # to close or keep closed, we must require that the last intention was to close from a more open position.
        user_wants_to_close_or_keep = user_wants_to_close or user_wants_to_keep
        user_expects_close_or_keep = (
            user_wants_to_close_or_keep and user_last_trajectory_was_closing
        )
        if (
            user_expects_close_or_keep
            and gripper_was_closing_or_still
            and gripper_is_opening
        ):

            # This is the command that we will issue as part of the regrasp. Note that instead we could calculate
            # force applied on the object. In this case, what we do is re-issue the last command that led to
            # a positive move / closing the gripper. Since the last command led to opening now, we must use at
            # least the second to last, which given our algorithm that requires past trajectories, we can guarantee
            # that it exists by now.
            assert self.second_prev_action is not None
            self.regrasp_command = self.second_prev_action

            # now print debug information so that we know that we are regrasping (if debug is required)
            if self.debug_regrasp:
                print(
                    f"user wants to : {self.debug_dir_to_string(position_control[0])}"
                )
                print(f"gripper is    : {self.debug_dir_to_string(obs_direction)}")
                self.debug_print_regrasp_history()

                print("This is an undesired opening!! Enabling re-grasp:")
                print(
                    f"We would like to keep {self.prev_obs_position}, and will reissue {self.regrasp_command} for it"
                )

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 4) Compare re-grasping command to what the user wants to do
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

        # if we have a re-grasp command, we will keep it if it's better than the user one
        if self.regrasp_command is None:
            returned_control = default_control
        else:

            # check if the user command is better, and if so update regrasp to the new command
            user_is_better = default_control[0] > self.regrasp_command[0]
            if user_is_better:
                if self.debug_regrasp:
                    print(
                        f"The user command {default_control} is better than {self.regrasp_command}, will update it."
                    )
                self.regrasp_command = default_control

            returned_control = self.regrasp_command

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # 5) Update cached values to help next frame make the re-grasp decision
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        command_direction = (
            None
            if np.allclose(position_control, 0, atol=1e-5)
            else np.sign(position_control)
        )
        # user trajectory
        if command_direction != 0.0:
            self.last_nonzero_cmd_direction = command_direction
        # observations and observation trajectory
        self.prev_obs_position = current_position
        if obs_direction != 0.0:
            self.last_nonzero_obs_direction = obs_direction
        # actual actions that are returned
        self.second_prev_action = self.prev_action
        self.prev_action = returned_control

        # update history only if we are debugging
        if self.debug_regrasp:
            self.debug_add_to_history(
                self.debug_desired_action_history, position_control[0]
            )
            self.debug_add_to_history(
                self.debug_desired_action_dir_history,
                self.debug_dir_to_string(command_direction),
            )
            self.debug_add_to_history(
                self.debug_observed_pos_history, current_position[0]
            )
            self.debug_add_to_history(
                self.debug_observed_pos_dir_history,
                self.debug_dir_to_string(obs_direction),
            )
            self.debug_add_to_history(
                self.debug_returned_ctrl_history, returned_control[0]
            )

        return returned_control
