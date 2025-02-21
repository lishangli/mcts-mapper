import numpy as np
import copy
import math
from utils import softmax
from line_profiler import profile


class MCTNode:
    def __init__(self, prob, distance, parent=None):
        # self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0
        self.p = prob
        self.u = 0
        self.q = -500
        self.h = 1 / (distance + 1)  # heuristic value
        self.policy = None
        self.reward = -1
        self.discount = 0.9

    def is_fully_expanded(self, state):

        return len(self.children) >= len(state.get_actions()) or self.children == {}

    def is_leaf(self):
        return self.children == {}

    def best_child(self, t, c_param=1000):
        # print("children nums: {}".format(len(self.children)))
        return max(self.children.items(), key=lambda x: x[1].get_value(c_param, t))

    def get_value(self, c_param, t):
        alpha = math.sqrt(self.parent.visits) / (1 + self.visits)
        # alpha = 0
        self.u = (alpha * (c_param * self.p) + self.q + 500) * (1 - t) + (
            (alpha * self.h) * t * c_param
        )
        # print("alpha: {}, h: {}".format(alpha, self.h))
        # self.u = (alpha * ((c_param * self.p) + self.q + 500)) + self.h * 10000
        # print(
        #     "q: {}, p: {}, u: {}, h: {}, parent visits: {}, visits: {}, step: {}".format(
        #         (self.q + 500),
        #         self.p,
        #         c_param * self.p * alpha * (1 - t),
        #         self.h * t * 10000,
        #         self.parent.visits,
        #         self.visits,
        #         t,
        #     )
        # )
        # self.reward = 0
        # self.q = 0
        return self.u

    def expand(self, action_probs, state, mode):
        # tried_actions = [child.state.last_action for action, child in self.children.items()]
        if mode == "placement":
            legal_actions = state.get_actions()
        else:
            legal_actions = state.get_route_actions()
        for action, prob in action_probs:
            if action not in self.children and action in legal_actions:
                # new_state = state.take_action(action)
                distance = 10000000000
                if mode == "placement":
                    distance = state.get_distance(action)
                else:
                    distance = state.get_route_distance(action)
                child = MCTNode(prob, distance, self)
                self.children[action] = child

    def simulate(self, action_probs, state):
        current_state = copy.deepcopy(state)
        node = self
        while not current_state.is_terminal():
            node.expand(action_probs, current_state, mode="route")
            act, node = self.best_child()
            current_state.take_action(act)

        ## print("sim finish")
        if current_state.is_route_success():
            print("route success")
        return current_state.get_reward()

    def backpropagate(self, result):
        self.visits += 1
        if self.parent:
            self.parent.backpropagate(result)
        self.q += 1.0 * (result - self.q) / self.visits


""" MCT definition"""


class MCTS:
    def __init__(self, agent_net, t=0, mode="placement"):
        self.root = MCTNode(1.0, 1)
        self.agent_net = agent_net
        self.exploration_weight = 1.4
        self.mode = mode
        self.t = math.exp(-t)

    def step(self, t):
        self.t = math.exp(-(t))

    @profile
    def search(self, state, alpha=0.9):
        reward = 0
        if self.mode == "placement":
            act, node = self.select_reward(state)
            action_probs, value = self.agent_net.policy_value_fn(state)
        else:
            act, node = self.select(state)
            alpha = 1
            action_probs, value = self.agent_net.route_policy_value_fn(state)
            ## PRINT INFO FOR ACTION_PROBS

        node.policy = action_probs

        if self.mode == "placement":
            if not state.is_terminal():
                node.expand(action_probs, state, self.mode)
        else:
            if state.is_terminal():
                value = state.get_reward()
            else:
                node.expand(action_probs, state, self.mode)
                value = value * alpha + node.reward
        node.backpropagate(-value)

    @profile
    def select(self, state):
        node = self.root
        act = None
        while not node.is_leaf() and not state.is_terminal():
            act, node = node.best_child(self.t)

            # ? should we take placement action and route action in the same MCT?
            if self.mode == "placement":
                state.take_action(act)
            else:
                state.take_route_action(act)
        return act, node

    @profile
    def select_reward(self, state):
        node = self.root
        act = None
        reward = 0
        if not node.is_leaf() and not state.is_terminal():
            act, node = node.best_child(self.t)
            if node.reward == -1:
                node.reward = self.agent_net.reward_model(state, act)[0]

            if self.mode == "placement":
                state.take_action(act)
            else:
                state.take_route_action(act)

        return act, node

    def get_move_probs(self, state, iters, temp=1e-3):
        for i in range(iters):
            state_cp = copy.deepcopy(state)
            self.search(state_cp)

        act_visits = [(act, node.visits) for act, node in self.root.children.items()]
        if act_visits == []:
            return None, 0
        acts, visits = zip(*act_visits)
        visits = np.array(visits)
        act_probs = softmax(1.0 / temp * np.log(visits + 1))
        return acts, act_probs

    def update_with_move(self, last_move):
        """update mcts root with last move"""
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = MCTNode(1.0, 1)


class place_MCTS:
    def __init__(self, agent_net, t=0):
        self.root = MCTNode(1.0, 1)
        self.agent_net = agent_net
        self.exploration_weight = 1.4
        self.t = math.exp(-t)

    def step(self, t):
        self.t = math.exp(-(t + 1))

    def search(self, state, alpha=0.9):

        act, node, reward = self.select(state)
        action_probs, value = self.agent_net.policy_value_fn(state)
        ## PRINT INFO FOR ACTION_PROBS

        node.policy = action_probs
        if not state.is_terminal():
            node.expand(action_probs, state, reward, self.mode)

        value = value * alpha + reward
        node.backpropagate(-value)

    def select(self, state):
        node = self.root
        act = None
        act, node = node.best_child(self.t)
        reward = self.reward_model(state, act)
        state.take_action(act)
        return act, node, reward


class route_MCTS:
    def __init__(self, agent_net, t=0):
        self.root = MCTNode(1.0, 1)
        self.agent_net = agent_net
        self.exploration_weight = 1.4
        self.t = math.exp(-t)

    def step(self, t):
        self.t = math.exp(-(t))

    def search(self, state, alpha=0.9):

        act, node = self.select(state)
        action_probs, value = self.agent_net.route_policy_value_fn(state)
        ## PRINT INFO FOR ACTION_PROBS

        node.policy = action_probs
        if not state.is_terminal():
            node.expand(action_probs, state, self.mode)
        else:
            value = state.get_reward()
        node.backpropagate(-value)

    def select(self, state):
        node = self.root
        act = None
        while not node.is_leaf() and not state.is_terminal():
            act, node = node.best_child(self.t)
            state.take_route_action(act)
        return act, node

    def get_move_probs(self, state, iters, temp=1e-3):
        for i in range(iters):
            state_cp = copy.deepcopy(state)
            self.search(state_cp)

        act_visits = [(act, node.visits) for act, node in self.root.children.items()]
        if act_visits == []:
            return None, 0
        acts, visits = zip(*act_visits)
        visits = np.array(visits)
        act_probs = softmax(1.0 / temp * np.log(visits + 1e-10))
        return acts, act_probs

    def update_with_move(self, last_move):
        """update mcts root with last move"""
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = MCTNode(1.0, 1)


class MCTSPlayer(object):
    def __init__(self, agent_net, t=0):
        self.mcts = MCTS(agent_net, t)
        self.route_mcts = MCTS(agent_net, t, "route")
        self.t = t

    def step(self):
        self.t += 0.0001
        self.mcts.step(self.t)
        self.route_mcts.step(self.t)

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def reset_route_player(self):
        self.route_mcts.update_with_move(-1)

    def get_action(self, state, temp=1e-3, return_prob=0, method="random"):
        actions = state.get_actions()
        action_probs = np.zeros(
            state.env.adg.getNodeNums() * len(state.env.adg_features_vec)
        )
        if len(actions) > 0:
            acts, probs = self.mcts.get_move_probs(state, 10)  ## 10000 iters
            if acts == None:
                return -1, None
            action_probs[list(acts)] = probs
            # print("legal actions is {}, and  action_probs is {}".format(acts, probs))
            if method == "greedy":
                action = acts[np.argmax(probs)]
            else:
                action = np.random.choice(
                    acts,
                    p=1.0 * probs
                    + 0.0 * np.random.dirichlet(0.3 * np.ones(len(probs))),
                )
                # print("action is {}".format(action))
            self.mcts.update_with_move(action)

            if return_prob:
                return action, action_probs
            else:
                return action
        else:
            print("WARNING: the state has no legal actions")
            return -1, None

    def get_route_action(self, state, temp=1e-3, return_prob=0):
        actions = state.get_route_actions()
        # print("route actions len {}".format(len(actions)))
        action_probs = np.zeros(len(state.env.adg.getEdges()))
        if len(actions) > 0:
            acts, probs = self.route_mcts.get_move_probs(state, 4)
            if acts == None:
                return -2, None
            action_probs[list(acts)] = probs
            action = np.random.choice(
                acts,
                p=1.0 * probs + 0.0 * np.random.dirichlet(0.3 * np.ones(len(probs))),
            )
            self.route_mcts.update_with_move(action)

            if return_prob:
                return action, action_probs
            else:
                return action
        else:
            if state.is_terminal():
                print("WARNING: the state has no route legal actions")
                return -1, None
            print("WARNING: the state don't need to route actions")
            return -2, None

    def reset(self):
        self.mcts.update_with_move(-1)
