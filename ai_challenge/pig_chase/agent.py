# Copyright (c) 2017 Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================================================================

from __future__ import division

import sys
import time
import random
from collections import namedtuple
from tkinter import ttk, Canvas, W

import numpy as np
from common import visualize_training, Entity, ENV_TARGET_NAMES, ENV_ENTITIES, ENV_AGENT_NAMES, \
    ENV_ACTIONS, ENV_CAUGHT_REWARD, ENV_BOARD_SHAPE
from six.moves import range

from malmopy.agent import AStarAgent
from malmopy.agent import QLearnerAgent, BaseAgent, RandomAgent
from malmopy.agent.gui import GuiAgent

P_FOCUSED = .5
P_RANDOM  = 0.25
P_DEFECT = 0.25
CELL_WIDTH = 33

State = namedtuple('State', ['Xs', 'Xe', 'Xp', 'Ae'])

class PigChaseQLearnerAgent(QLearnerAgent):
    """A thin wrapper around QLearnerAgent that normalizes rewards to [-1,1]"""

    def act(self, state, reward, done, is_training=False):

        reward /= ENV_CAUGHT_REWARD
        return super(PigChaseQLearnerAgent, self).act(state, reward, done,
                                                      is_training)


class PigChaseChallengeAgent(BaseAgent):
    """Pig Chase challenge agent - behaves focused or random."""


    def __init__(self, name, visualizer=None):

        nb_actions = len(ENV_ACTIONS)
        super(PigChaseChallengeAgent, self).__init__(name, nb_actions,
                                                     visualizer = visualizer)

        self._agents = []
        self._agents.append(FocusedAgent(name, ENV_TARGET_NAMES[0],
                                         visualizer = visualizer))
        self._agents.append(RandomAgent(name, nb_actions,
                                        visualizer = visualizer))
        self._agents.append(DefectAgent(name, visualizer = visualizer))
        self.current_agent = self._select_agent(P_FOCUSED)

    def _select_agent(self, p_focused):
        return self._agents[np.random.choice(range(len(self._agents)), p = [P_FOCUSED, P_RANDOM, P_DEFECT])]

    def act(self, new_state, reward, done, is_training=False):
        if done:
            self.current_agent = self._select_agent(P_FOCUSED)
        return self.current_agent.act(new_state, reward, done, is_training)

    def save(self, out_dir):
        self.current_agent.save(out_dir)

    def load(self, out_dir):
        self.current_agent(out_dir)

    def inject_summaries(self, idx):
        self.current_agent.inject_summaries(idx)


class FocusedAgent(AStarAgent):
    ACTIONS = ENV_ACTIONS
    Neighbour = namedtuple('Neighbour', ['cost', 'x', 'z', 'direction', 'action'])

    def __init__(self, name, target, visualizer = None):
        super(FocusedAgent, self).__init__(name, len(FocusedAgent.ACTIONS),
                                           visualizer = visualizer)
        self._target = str(target)
        self._previous_target_pos = None
        self._action_list = []

    def act(self, state, reward, done, is_training=False):
        if done:
            self._action_list = []
            self._previous_target_pos = None

        if state is None:
            return np.random.randint(0, self.nb_actions)
        entities = state[1]
        state = state[0]

        me = [(j, i) for i, v in enumerate(state) for j, k in enumerate(v) if self.name in k]
        me_details = [e for e in entities if e['name'] == self.name][0]
        yaw = int(me_details['yaw'])
        direction = ((((yaw - 45) % 360) // 90) - 1) % 4  # convert Minecraft yaw to 0=north, 1=east etc.
        target = [(j, i) for i, v in enumerate(state) for j, k in enumerate(v) if self._target in k]

        # Get agent and target nodes
        me = FocusedAgent.Neighbour(1, me[0][0], me[0][1], direction, "")
        target = FocusedAgent.Neighbour(1, target[0][0], target[0][1], 0, "")

        # If distance to the pig is one, just turn and wait
        if self.heuristic(me, target) == 1:
            return FocusedAgent.ACTIONS.index("turn 1")  # substitutes for a no-op command

        if not self._previous_target_pos == target:
            # Target has moved, or this is the first action of a new mission - calculate a new action list
            self._previous_target_pos = target

            path, costs = self._find_shortest_path(me, target, state=state)
            self._action_list = []
            for point in path:
                self._action_list.append(point.action)

        if self._action_list is not None and len(self._action_list) > 0:
            action = self._action_list.pop(0)
            return FocusedAgent.ACTIONS.index(action)

        # reached end of action list - turn on the spot
        return FocusedAgent.ACTIONS.index("turn 1")  # substitutes for a no-op command

    def neighbors(self, pos, state=None):
        state_width = state.shape[1]
        state_height = state.shape[0]
        dir_north, dir_east, dir_south, dir_west = range(4)
        neighbors = []
        inc_x = lambda x, dir, delta: x + delta if dir == dir_east else x - delta if dir == dir_west else x
        inc_z = lambda z, dir, delta: z + delta if dir == dir_south else z - delta if dir == dir_north else z
        # add a neighbour for each potential action; prune out the disallowed states afterwards
        for action in FocusedAgent.ACTIONS:
            if action.startswith("turn"):
                neighbors.append(
                    FocusedAgent.Neighbour(1, pos.x, pos.z, (pos.direction + int(action.split(' ')[1])) % 4, action))
            if action.startswith("move "):  # note the space to distinguish from movemnorth etc
                sign = int(action.split(' ')[1])
                weight = 1 if sign == 1 else 1.5
                neighbors.append(
                    FocusedAgent.Neighbour(weight, inc_x(pos.x, pos.direction, sign), inc_z(pos.z, pos.direction, sign),
                                           pos.direction, action))
            if action == "movenorth":
                neighbors.append(FocusedAgent.Neighbour(1, pos.x, pos.z - 1, pos.direction, action))
            elif action == "moveeast":
                neighbors.append(FocusedAgent.Neighbour(1, pos.x + 1, pos.z, pos.direction, action))
            elif action == "movesouth":
                neighbors.append(FocusedAgent.Neighbour(1, pos.x, pos.z + 1, pos.direction, action))
            elif action == "movewest":
                neighbors.append(FocusedAgent.Neighbour(1, pos.x - 1, pos.z, pos.direction, action))

        # now prune:
        valid_neighbours = [n for n in neighbors if
                            n.x >= 0 and n.x < state_width and n.z >= 0 and n.z < state_height and state[
                                n.z, n.x] != 'sand']
        return valid_neighbours

    def heuristic(self, a, b, state=None):
        (x1, y1) = (a.x, a.z)
        (x2, y2) = (b.x, b.z)
        return abs(x1 - x2) + abs(y1 - y2)

    def matches(self, a, b):
        return a.x == b.x and a.z == b.z  # don't worry about dir and action


class DefectAgent(AStarAgent):
    ACTIONS = ENV_ACTIONS
    Neighbour = namedtuple('Neighbour', ['cost', 'x', 'z', 'direction', 'action'])

    def __init__(self, name, visualizer = None):
        super(DefectAgent, self).__init__(name, len(DefectAgent.ACTIONS),visualizer = visualizer)
        self.me = { 'name': str(name) }
        self._action_list = []

        self.right_hole = DefectAgent.Neighbour(1, 7, 4, 0, "")
        self.left_hole = DefectAgent.Neighbour(1, 1, 4, 0, "")

    def matches(self, a, b):
        return a.x == b.x and a.z == b.z  # don't worry about dir and action

    # convert Minecraft yaw to: 0 = north; 1 = east; 2 = south; 3 = west
    def yaw_to_direction(self, yaw):
        return ((((yaw - 45) % 360) // 90) - 1) % 4;

    def get_action_list(self, path):
        self._action_list = []
        for point in path:
            self._action_list.append(point.action)

    def act(self, obs, reward, done, is_training=False):
        if done:
           self._action_list = []

        if obs is None:
        	return np.random.randint(0, self.nb_actions)

        world = obs[0]
        entities = obs[1]

        # Get my direction and enemy's direction
        for entitie in entities:
            if entitie[u'name'] == self.me['name']:
                dir_me = self.yaw_to_direction( int(entitie[u'yaw']) )

        me = [(j, i) for i, v in enumerate(obs) for j, k in enumerate(v) if self.name in k]

        # Get my position
        self.me['position'] = [(j, i, dir_me) for i, v in enumerate(world) for j, k in enumerate(v) if self.me['name'] in k][0]
        self.me_def = DefectAgent.Neighbour(1, self.me['position'][0], self.me['position'][1], self.me['position'][2], "")

        path_to_right, cost_to_right = self._find_shortest_path(self.me_def, self.right_hole, state=world)
        path_to_left, cost_to_left = self._find_shortest_path(self.me_def, self.left_hole, state=world)

        if len(path_to_right) > len(path_to_left):
            self.get_action_list(path_to_left)
        else:
            self.get_action_list(path_to_right)


        if self._action_list is not None and len(self._action_list) > 0:
            action = self._action_list.pop(0)
            return DefectAgent.ACTIONS.index(action)

        return DefectAgent.ACTIONS.index('turn 1')

    def heuristic(self, a, b, state=None):
        (x1, y1) = (a.x, a.z)
        (x2, y2) = (b.x, b.z)
        return abs(x1 - x2) + abs(y1 - y2)


    def neighbors(self, pos, state=None):
        state_width = state.shape[1]
        state_height = state.shape[0]
        dir_north, dir_east, dir_south, dir_west = range(4)
        neighbors = []
        inc_x = lambda x, dir, delta: x + delta if dir == dir_east else x - delta if dir == dir_west else x
        inc_z = lambda z, dir, delta: z + delta if dir == dir_south else z - delta if dir == dir_north else z
        # add a neighbour for each potential action; prune out the disallowed states afterwards
        for action in FocusedAgent.ACTIONS:
            if action.startswith("turn"):
                neighbors.append(FocusedAgent.Neighbour(1, pos.x, pos.z, (pos.direction + int(action.split(' ')[1])) % 4, action))
            if action.startswith("move "):  # note the space to distinguish from movemnorth etc
                sign = int(action.split(' ')[1])
                weight = 1 if sign == 1 else 1.5
                neighbors.append(FocusedAgent.Neighbour(weight, inc_x(pos.x, pos.direction, sign), inc_z(pos.z, pos.direction, sign),pos.direction, action))

        # now prune:
        valid_neighbours = [n for n in neighbors if
                            n.x >= 0 and n.x < state_width and n.z >= 0 and n.z < state_height and state[
                                n.z, n.x] != 'sand']
        return valid_neighbours


class PigChaseHumanAgent(GuiAgent):
    def __init__(self, name, environment, keymap, max_episodes, max_actions,
                 visualizer, quit):
        self._max_episodes = max_episodes
        self._max_actions = max_actions
        self._action_taken = 0
        self._episode = 1
        self._scores = []
        self._rewards = []
        self._episode_has_ended = False
        self._episode_has_started = False
        self._quit_event = quit
        super(PigChaseHumanAgent, self).__init__(name, environment, keymap,
                                                 visualizer=visualizer)

    def _build_layout(self, root):
        # Left part of the GUI, first person view
        self._first_person_header = ttk.Label(root, text='First Person View', font=(None, 14, 'bold')) \
            .grid(row=0, column=0)
        self._first_person_view = ttk.Label(root)
        self._first_person_view.grid(row=1, column=0, rowspan=10)

        # Right part, top
        self._first_person_header = ttk.Label(root, text='Symbolic View', font=(None, 14, 'bold')) \
            .grid(row=0, column=1)
        self._symbolic_view = Canvas(root)
        self._symbolic_view.configure(width=ENV_BOARD_SHAPE[0]*CELL_WIDTH,
                                      height=ENV_BOARD_SHAPE[1]*CELL_WIDTH)
        self._symbolic_view.grid(row=1, column=1)

        # Bottom information
        self._information_panel = ttk.Label(root, text='Game stats', font=(None, 14, 'bold'))
        self._current_episode_lbl = ttk.Label(root, text='Episode: 0', font=(None, 12))
        self._cum_reward_lbl = ttk.Label(root, text='Score: 0', font=(None, 12, 'bold'))
        self._last_action_lbl = ttk.Label(root, text='Previous action: None', font=(None, 12))
        self._action_done_lbl = ttk.Label(root, text='Actions taken: 0', font=(None, 12))
        self._action_remaining_lbl = ttk.Label(root, text='Actions remaining: 0', font=(None, 12))

        self._information_panel.grid(row=2, column=1)
        self._current_episode_lbl.grid(row=3, column=1, sticky=W, padx=20)
        self._cum_reward_lbl.grid(row=4, column=1, sticky=W, padx=20)
        self._last_action_lbl.grid(row=5, column=1, sticky=W, padx=20)
        self._action_done_lbl.grid(row=6, column=1, sticky=W, padx=20)
        self._action_remaining_lbl.grid(row=7, column=1, sticky=W, padx=20)
        self._overlay = None

        # Main rendering callback
        self._pressed_binding = root.bind('<Key>', self._on_key_pressed)
        self._user_pressed_enter = False

        # UI Update callback
        root.after(self._tick, self._poll_frame)
        root.after(1000, self._on_episode_start)

        root.focus()

    def _draw_arrow(self, yaw, x, y, cell_width, colour):
        if yaw == 0.:
            x1, y1 = (x + .15) * cell_width, (y + .15) * cell_width
            x2, y2 = (x + .5) * cell_width, (y + .4) * cell_width
            x3, y3 = (x + .85) * cell_width, (y + .85) * cell_width

            self._symbolic_view.create_polygon(x1, y1, x2, y3, x3, y1, x2, y2, fill=colour)
        elif yaw == 90.:
            x1, y1 = (x + .15) * cell_width, (y + .15) * cell_width
            x2, y2 = (x + .6) * cell_width, (y + .5) * cell_width
            x3, y3 = (x + .85) * cell_width, (y + .85) * cell_width

            self._symbolic_view.create_polygon(x1, y2, x3, y1, x2, y2, x3, y3, fill=colour)
        elif yaw == 180.:
            x1, y1 = (x + .15) * cell_width, (y + .15) * cell_width
            x2, y2 = (x + .5) * cell_width, (y + .6) * cell_width
            x3, y3 = (x + .85) * cell_width, (y + .85) * cell_width

            self._symbolic_view.create_polygon(x1, y3, x2, y1, x3, y3, x2, y2, fill=colour)
        else:
            x1, y1 = (x + .15) * cell_width, (y + .15) * cell_width
            x2, y2 = (x + .4) * cell_width, (y + .5) * cell_width
            x3, y3 = (x + .85) * cell_width, (y + .85) * cell_width

            self._symbolic_view.create_polygon(x1, y3, x2, y2, x1, y1, x3, y2, fill=colour)

    def _poll_frame(self):
        """
        Main callback for UI rendering.
        Called at regular intervals.
        The method will ask the environment to provide a frame if available (not None).
        :return:
        """
        cell_width = CELL_WIDTH
        circle_radius = 10

        # are we done?
        if self._env.done and not self._episode_has_ended:
            self._on_episode_end()

        # build symbolic view
        board = None
        if self._env is not None:
            board, _ = self._env._internal_symbolic_builder.build(self._env)
        if board is not None:
            board = board.T
            self._symbolic_view.delete('all')  # Remove all previous items from Tkinter tracking
            width, height = board.shape
            for x in range(width):
                for y in range(height):
                    cell_contents = str.split(str(board[x][y]), '/')
                    for block in cell_contents:
                        if block == 'sand':
                            self._symbolic_view.create_rectangle(x * cell_width, y * cell_width,
                                                                 (x + 1) * cell_width, (y + 1) * cell_width,
                                                                 outline="black", fill="orange", tags="square")
                        elif block == 'grass':
                            self._symbolic_view.create_rectangle(x * cell_width, y * cell_width,
                                                                 (x + 1) * cell_width, (y + 1) * cell_width,
                                                                 outline="black", fill="lawn green", tags="square")
                        elif block == 'lapis_block':
                            self._symbolic_view.create_rectangle(x * cell_width, y * cell_width,
                                                                 (x + 1) * cell_width, (y + 1) * cell_width,
                                                                 outline="black", fill="black", tags="square")
                        elif block == ENV_TARGET_NAMES[0]:
                            self._symbolic_view.create_oval((x + .5) * cell_width - circle_radius,
                                                            (y + .5) * cell_width - circle_radius,
                                                            (x + .5) * cell_width + circle_radius,
                                                            (y + .5) * cell_width + circle_radius,
                                                            fill='pink')
                        elif block == self.name:
                            yaw = self._env._world_obs['Yaw'] % 360
                            self._draw_arrow(yaw, x, y, cell_width, 'red')
                        elif block == ENV_AGENT_NAMES[0]:
                            # Get yaw of other agent:
                            entities = self._env._world_obs[ENV_ENTITIES]
                            other_agent = list(
                                map(Entity.create, filter(lambda e: e['name'] == ENV_AGENT_NAMES[0], entities)))
                            if len(other_agent) == 1:
                                other_agent = other_agent.pop()
                                yaw = other_agent.yaw % 360
                                self._draw_arrow(yaw, x, y, cell_width, 'blue')

        # display the most recent frame
        frame = self._env.frame
        if frame is not None:
            from PIL import ImageTk
            self._first_person_view.image = ImageTk.PhotoImage(image=frame)
            self._first_person_view.configure(image=self._first_person_view.image)
            self._first_person_view.update()

        self._first_person_view.update()

        # process game state (e.g., has the episode started?)
        if self._episode_has_started and time.time() - self._episode_start_time < 3:
            if not hasattr(self, "_init_overlay") or not self._init_overlay:
                self._create_overlay()
            self._init_overlay.delete("all")
            self._init_overlay.create_rectangle(
                10, 10, 590, 290, fill="white", outline="red", width="5")
            self._init_overlay.create_text(
                300, 80, text="Get ready to catch the pig!",
                font=('Helvetica', '18'))
            self._init_overlay.create_text(
                300, 140, text=str(3 - int(time.time() - self._episode_start_time)),
                font=('Helvetica', '18'), fill="red")
            self._init_overlay.create_text(
                300, 220, width=460,
                text="How to play: \nUse the left/right arrow keys to turn, "
                     "forward/back to move. The pig is caught if it is "
                     "cornered without a free block to escape to.",
                font=('Helvetica', '14'), fill="black")
            self._root.update()

        elif self._episode_has_ended:

            if not hasattr(self, "_init_overlay") or not self._init_overlay:
                self._create_overlay()
            self._init_overlay.delete("all")
            self._init_overlay.create_rectangle(
                10, 10, 590, 290, fill="white", outline="red", width="5")
            self._init_overlay.create_text(
                300, 80, text='Finished episode %d of %d' % (self._episode, self._max_episodes),
                font=('Helvetica', '18'))
            self._init_overlay.create_text(
                300, 120, text='Score: %d' % sum(self._rewards),
                font=('Helvetica', '18'))
            if self._episode > 1:
                self._init_overlay.create_text(
                    300, 160, text='Average over %d episodes: %.2f' % (self._episode, np.mean(self._scores)),
                    font=('Helvetica', '18'))
            self._init_overlay.create_text(
                300, 220, width=360,
                text="Press RETURN to start the next episode, ESC to exit.",
                font=('Helvetica', '14'), fill="black")
            self._root.update()

        elif hasattr(self, "_init_overlay") and self._init_overlay:
            self._destroy_overlay()

        # trigger the next update
        self._root.after(self._tick, self._poll_frame)

    def _create_overlay(self):
        self._init_overlay = Canvas(self._root, borderwidth=0, highlightthickness=0, width=600, height=300, bg="gray")
        self._init_overlay.place(relx=0.5, rely=0.5, anchor='center')

    def _destroy_overlay(self):
        self._init_overlay.destroy()
        self._init_overlay = None

    def _on_key_pressed(self, e):
        """
        Main callback for keyboard events
        :param e:
        :return:
        """
        if e.keysym == 'Escape':
            self._quit()

        if e.keysym == 'Return' and self._episode_has_ended:

            if self._episode >= self._max_episodes:
                self._quit()

            # start the next episode
            self._action_taken = 0
            self._rewards = []
            self._episode += 1
            self._env.reset()

            self._on_episode_start()
            print('Starting episode %d' % self._episode)

        if self._episode_has_started and time.time() - self._episode_start_time >= 3:
            if e.keysym in self._keymap:
                mapped_action = self._keymap.index(e.keysym)

                _, reward, done = self._env.do(mapped_action)
                self._action_taken += 1
                self._rewards.append(reward)
                self._on_experiment_updated(mapped_action, reward, done)

    def _on_episode_start(self):
        self._episode_has_ended = False
        self._episode_has_started = True
        self._episode_start_time = time.time()
        self._on_experiment_updated(None, 0, self._env.done)

    def _on_episode_end(self):
        # do a turn to ensure we get the final reward and observation
        no_op_action = 0
        _, reward, done = self._env.do(no_op_action)
        self._action_taken += 1
        self._rewards.append(reward)
        self._on_experiment_updated(no_op_action, reward, done)

        # report scores
        self._scores.append(sum(self._rewards))
        self.visualize(self._episode, 'Reward', sum(self._rewards))

        # set flags to start a new episode
        self._episode_has_started = False
        self._episode_has_ended = True

    def _on_experiment_updated(self, action, reward, is_done):
        self._current_episode_lbl.config(text='Episode: %d' % self._episode)
        self._cum_reward_lbl.config(text='Score: %d' % sum(self._rewards))
        self._last_action_lbl.config(text='Previous action: %s' % action)
        self._action_done_lbl.config(text='Actions taken: {0}'.format(self._action_taken))
        self._action_remaining_lbl.config(text='Actions remaining: %d' % (self._max_actions - self._action_taken))
        self._first_person_view.update()

    def _quit(self):
        self._quit_event.set()
        self._root.quit()
        sys.exit()

State = namedtuple('State', ['Xs', 'Xe', 'Xp', 'Ae'])

class QLearnerAgent(AStarAgent):
    ACTIONS = ENV_ACTIONS
    Neighbour = namedtuple('Neighbour', ['cost', 'x', 'z', 'direction', 'action'])

    def __init__(self, name, enemy, target, alpha, gamma, eps, visualizer = None):
        super(QLearnerAgent, self).__init__(name, len(QLearnerAgent.ACTIONS),visualizer = visualizer)
        self.me = { 'name': str(name) }
        self.enemy = { 'name': str(enemy) }
        self.pig = { 'name': str(target) }
        self._action_list = []

        self._previous_enemy_pos = None
        self._previous_pig_pos = None
        self._previous_enemy_def = None
        self._previous_pig_def = None

        self.right_hole = QLearnerAgent.Neighbour(1, 7, 4, 0, "")
        self.left_hole = QLearnerAgent.Neighbour(1, 1, 4, 0, "")

        self.Q = np.load('Q.npy').item()

        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eps = float(eps)

    def matches(self, a, b):
        return a.x == b.x and a.z == b.z  # don't worry about dir and action

    def get_key(self, obs):
        # Get current world state
        world = obs[0]
        entities = obs[1]

        # Get my direction and enemy's direction
        for entitie in entities:
            if entitie[u'name'] == self.me['name']:
                dir_me = self.yaw_to_direction( int(entitie[u'yaw']) )
            elif entitie[u'name'] == self.enemy['name']:
                dir_enemy = self.yaw_to_direction( int(entitie[u'yaw']) )

        # Get my position
        self.me['position'] = [(j, i, dir_me) for i, v in enumerate(world) for j, k in enumerate(v) if self.me['name'] in k][0]

        # Get enemy position
        aux_l = len([(j, i, dir_enemy) for i, v in enumerate(world) for j, k in enumerate(v) if self.enemy['name'] in k])

        if aux_l == 0:
            self.enemy['position'] = (-1, -1, 1)
        else:
            self.enemy['position'] = [(j, i, dir_enemy) for i, v in enumerate(world) for j, k in enumerate(v) if self.enemy['name'] in k][0]

        # Get pig position
        aux_l = len([(j, i, 0) for i, v in enumerate(world) for j, k in enumerate(v) if self.pig['name'] in k])
        if aux_l == 0:
            self.pig['position'] = (-2, -2, 1)
        else:
            self.pig['position'] = [(j, i, 0) for i, v in enumerate(world) for j, k in enumerate(v) if self.pig['name'] in k][0]

        self.me_def = QLearnerAgent.Neighbour(1, self.me['position'][0], self.me['position'][1], self.me['position'][2], "")
        self.pig_def = QLearnerAgent.Neighbour(1, self.pig['position'][0], self.pig['position'][1], self.pig['position'][2], "")
        self.enemy_def = QLearnerAgent.Neighbour(1, self.enemy['position'][0], self.enemy['position'][1], self.enemy['position'][2], "")

        if self._previous_enemy_pos == None:
            enemy_action = True
        else:
            enemy_action = self.enemy_chasing_pig(world)

        return State( (self.me['position'][0], self.me['position'][1]),(self.enemy['position'][0], self.enemy['position'][1]),(self.pig['position'][0], self.pig['position'][1]),enemy_action )

    def updateQ(self, obs, action, new_obs, reward, intention, len_path):

        if obs == None or new_obs == None:
            return

        aux = self.get_key(obs)
        q_prev = ( aux, action )

        aux = self.get_key(new_obs)

        # Defect or Cooperate
        A = [0, 1]

        maximum = -25
        m = 0
        for i, a in enumerate(A):
            S = (aux, a)
            if S in self.Q.keys() and self.Q[S] > maximum:
                maximum = self.Q[S]
                m = i

        if maximum == -25:
            m = random.choice(A)

        q_next = (aux, m)

        if intention == 0:
            rew = ( len_path * (-1) ) + 5
        else:
            rew = ( len_path * (-1) ) + 25

        self.Q[q_prev] = self.Q.setdefault(q_prev, 0) + self.alpha * (rew + self.eps * (self.Q.setdefault(q_next, 0) - self.Q.setdefault(q_prev, 0)))

    def print_map(self, world):
        print 'World shape: {}'.format(world.shape)
        for l in world:
            for c in l:
                if 'Agent_1' in c:
                    print 'E',
                elif 'Agent_2' in c:
                    print 'S',
                elif 'Pig' in c:
                    print 'P',
                elif 'sand' in c:
                    print 'X',
                elif 'lapis_block' in c:
                    print 'L',
                else:
                    print 'O',
            print ''

    # convert Minecraft yaw to: 0 = north; 1 = east; 2 = south; 3 = west
    def yaw_to_direction(self, yaw):
        return ((((yaw - 45) % 360) // 90) - 1) % 4;

    def get_action_list(self, path):
        self._action_list = []
        for point in path:
            self._action_list.append(point.action)

    def pig_moved(self):
        return not ( (self._previous_pig_pos[0] == self.pig['position'][0]) and (self._previous_pig_pos[1] == self.pig['position'][1]) )

    def enemy_chasing_pig(self, world):
        if self.pig_moved():
            past_path, past_cost = self._find_shortest_path(self._previous_enemy_def, self._previous_pig_def, state=world)
            path, cost = self._find_shortest_path(self.enemy_def, self._previous_pig_def, state=world)
        else:
            past_path, past_cost = self._find_shortest_path(self._previous_enemy_def, self.pig_def, state=world)
            path, cost = self._find_shortest_path(self.enemy_def, self.pig_def, state=world)

        # print 'Enemy Distance from pig: {} {}'.format(len(past_path), len(path))

        if len(path) > len(past_path):
            return False
        else:
            return True

    def get_action_list(self, path):
        self._action_list = []
        for point in path:
            self._action_list.append(point.action)

    def act(self, obs, reward, done, is_training=False):
        if done:
           self._action_list = []
           self._chasing_pig = True
           self._previous_pig_pos = None
           self._previous_enemy_pos = None
           self._previous_pig_def = None
           self._previous_enemy_def = None

        m = 0
        len_path = 100
        if obs == None:
            return QLearnerAgent.ACTIONS.index('turn 1'), m, len_path

        # Get current world state
        world = obs[0]
        entities = obs[1]

        # self.print_map(world)

        # 'Xs', 'Xe', 'Xp', 'Ae'
        s = self.get_key(obs)

        # Defect or Cooperate
        A = [0, 1]

        maximum = -25
        m = 0

        for i, a in enumerate(A):
            S = (s, a)
            if S in self.Q.keys() and self.Q[S] > maximum:
                maximum = self.Q[S]
                m = i

        if maximum == -25 or random.random() < self.eps:
            m = random.choice(A)

        if m == 0: # Defect
            path_to_right, cost_to_right = self._find_shortest_path(self.me_def, self.right_hole, state=world)
            path_to_left, cost_to_left = self._find_shortest_path(self.me_def, self.left_hole, state=world)
            # print 'Distance Left: Hole: {} Distance Right Hole: {}'.format(len(path_to_left), len(path_to_right))

            if len(path_to_right) > len(path_to_left):
                self.get_action_list(path_to_left)
            else:
                self.get_action_list(path_to_right)
        else: #Cooperate
            path, cost = self._find_shortest_path(self.me_def, self.pig_def, state=world)
            self.get_action_list(path)

        self._previous_pig_pos = self.pig['position']
        self._previous_enemy_pos = self.enemy['position']
        self._previous_pig_def = self.pig_def
        self._previous_enemy_def = self.enemy_def

        len_path = len(self._action_list)

        # print ''

        if self._action_list is not None and len(self._action_list) > 0:
            action = self._action_list.pop(0)
            return QLearnerAgent.ACTIONS.index(action), m, len_path

        return QLearnerAgent.ACTIONS.index('turn 1'), m, len_path

    def neighbors(self, pos, state=None):
        state_width = state.shape[1]
        state_height = state.shape[0]
        dir_north, dir_east, dir_south, dir_west = range(4)
        neighbors = []
        inc_x = lambda x, dir, delta: x + delta if dir == dir_east else x - delta if dir == dir_west else x
        inc_z = lambda z, dir, delta: z + delta if dir == dir_south else z - delta if dir == dir_north else z
        # add a neighbour for each potential action; prune out the disallowed states afterwards
        for action in FocusedAgent.ACTIONS:
            if action.startswith("turn"):
                neighbors.append(FocusedAgent.Neighbour(1, pos.x, pos.z, (pos.direction + int(action.split(' ')[1])) % 4, action))
            if action.startswith("move "):  # note the space to distinguish from movemnorth etc
                sign = int(action.split(' ')[1])
                weight = 1 if sign == 1 else 1.5
                neighbors.append(FocusedAgent.Neighbour(weight, inc_x(pos.x, pos.direction, sign), inc_z(pos.z, pos.direction, sign),pos.direction, action))

        # now prune:
        valid_neighbours = [n for n in neighbors if
                            n.x >= 0 and n.x < state_width and n.z >= 0 and n.z < state_height and state[
                                n.z, n.x] != 'sand']
        return valid_neighbours

    def heuristic(self, a, b, state=None):
        (x1, y1) = (a.x, a.z)
        (x2, y2) = (b.x, b.z)
        return abs(x1 - x2) + abs(y1 - y2)
