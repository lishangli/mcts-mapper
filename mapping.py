from adgParser import *
from dfgParser import *
from state import Environment, MappingState
import numpy as np
from ppo import PPOPlayer
from mcts import MCTSPlayer
from visual import GraphVisual
from line_profiler import profile


class DFGNodeAttr:
    def __init__(self):
        self.minLat = 0
        self.maxLat = 0
        self.lat = 0
        self.adgNode = None


class ADGNodeAttr:
    def __init__(self):
        self.dfgNode = None
        self.inPortUsed = []
        self.outPortUsed = []


class Mapping:
    def __init__(self, dfg, adg, ops, is_heuristic=False):
        self.env = Environment(dfg, adg, ops, is_heuristic)
        self.state = MappingState(self.env)

    @profile
    def start_route(self, agent):
        (
            states,
            mcts_probs,
        ) = (
            [],
            [],
        )
        self.state.mode = "route"
        route_latencies = []
        route_mask = copy.deepcopy(self.state.env.route_mask)
        adg_adj = copy.deepcopy(self.state.env.adg_adj)
        adg_features_vec = copy.deepcopy(self.state.env.adg_features_vec)
        route_step = copy.deepcopy(self.env.route_step)

        while True:
            move, move_probs = agent.get_route_action(
                self.state, temp=1e-3, return_prob=1
            )

            states.append(self.state)
            mcts_probs.append(move_probs)
            if move < 0:
                return None

            srcN = self.state.env.adg.getEdges()[move].getSrcId()
            dstN = self.state.env.adg.getEdges()[move].getDstId()
            # print("cur route action is {} to {}".format(srcN, dstN))

            self.state.take_route_action(move)

            sum_latency = self.state.get_reward()

            route_latencies.append(sum_latency)
            if self.state.is_terminal():
                # self.state.mode = "placement"
                # if len(self.state.env.route_cur_node) > 0:
                #     self.state.env.route_mask = route_mask
                #     self.state.adg_adj = adg_adj
                #     self.state.adg_features_vec = adg_features_vec
                #     self.route_step = route_step
                agent.reset_route_player()
                return zip(states, mcts_probs, route_latencies)

    @profile
    def start_mapping(self, agent, print_action=False):
        self.env.reset()
        self.state = MappingState(self.env)
        self.state.mode = "placement"
        policy_value_net = agent.mcts.agent_net.policy_value_fn
        states, mcts_probs, current_players = [], [], []
        route_states, route_mcts_probs, route_latencies = [], [], []

        latencies, rewards, actions = [], [], []

        step = 0
        while True:

            step = step + 1
            move, move_probs = agent.get_action(self.state, temp=1e-2, return_prob=1)
            print(move_probs)
            if move == -1:
                print("cur move is -1")
                break
            # print("cur move is {}".format(move))
            states.append(self.state)
            mcts_probs.append(move_probs)
            pre_reward = self.state.get_reward()

            actions.append(move)
            print("pre_reward is {}".format(pre_reward))
            if print_action:
                print(
                    "step {}, select dfg node {} on adg node {}".format(
                        step, move[0], move[1]
                    )
                )

            self.state.take_action(move)  # take placement action
            latencies.append(pre_reward)
            self.state.mode = "route"
            route_buffer = self.start_route(agent)  # take routing actions
            self.state.mode = "placement"
            if route_buffer == None:
                continue
            else:
                for route_state, route_probs, route_latency in route_buffer:
                    route_states.append(route_state)
                    route_mcts_probs.append(route_probs)
                    route_latencies.append(route_latency)
            # rewards.append(self.state.get_reward())
            reward = self.state.get_reward() - pre_reward
            rewards.append(reward)
            # act, next_value = policy_value_net(self.state)
            # latencies.append(reward + next_value)
            print("cur reward is {}".format(pre_reward - self.state.get_reward()))
            if self.state.is_terminal():
                agent.reset_player()
                break

        final_val = self.state.get_reward()
        latencies = [final_val - latency for latency in latencies]
        route_latencies = [
            final_val - route_latency for route_latency in route_latencies
        ]
        # for route_latency in route_latencies:
        #     route_latency = final_val - route_latency

        # for latency in latencies:
        #     print("latency is {}".format(latency))
        # for route_latency in route_latencies:
        #     print("route_latency is {}".format(route_latency))
        if self.state.mapping_state:
            print("the mapping is success")
        return (
            self.state.get_reward(),
            zip(states, mcts_probs, latencies, rewards, actions),
            zip(route_states, route_mcts_probs, route_latencies),
        )

    @profile
    def test_mapping(self, agent, is_shown=False, temp=1e-2):
        self.env.reset()
        self.state = MappingState(self.env)
        sum_latency = 0
        latencies = []
        g_visual = GraphVisual(self.env.dfg, self.env.adg)
        self.state.start_anime(g_visual)
        while True:
            move, move_probs = agent.get_action(self.state, temp=temp, return_prob=1)
            if move == -1:
                print("cur move is -1")
                if is_shown:
                    self.state.end_anime(g_visual)
                break

            # print(
            #     "cur action is {}, {}".format(
            #         move // self.env.adg_actions, move % self.env.adg_actions
            #     )
            # )

            self.state.take_action(action=move, print_action=True)
            self.state.mode = "route"
            route_buffer = self.start_route(agent)  # take routing actions
            self.state.mode = "placement"
            self.state.draw(g_visual)
            if route_buffer == None:
                continue
            sum_latency = self.state.get_reward()
            latencies.append(sum_latency)
            if self.state.is_terminal():
                if not self.state.mapping_state:
                    print("the mapping is failed")
                # print("the mapping state latency is {}".format(sum_latency))
                # print(np.where(self.state.env.route_mask == 0)[0].shape)
                # print(np.where(self.state.route_mask == 0)[0].shape)
                if is_shown:
                    self.state.end_anime(g_visual)
                return sum_latency

    def ppo_mapping(self, agent, is_shown=False):
        self.env.reset()
        self.state = MappingState(self.env)
        sum_latency = 0
        rewards = []
        values = []
        actions = []
        states, ppo_probs, current_players = [], [], []
        route_states, route_mcts_probs, route_latencies = [], [], []
        while True:
            act, probs = agent.get_ppo_action(self.state, return_probs=True)
            states.append(self.state)
            ppo_probs.append(probs)
            actions.append(act)
            pre_reward = self.state.get_reward()
            self.state.take_action(action=act, print_action=True)
            self.state.mode = "route"
            route_buffer = self.start_route(agent)
            self.state.mode = "placement"
            if route_buffer == None:
                continue
            else:
                for route_state, route_probs, route_latency in route_buffer:
                    route_states.append(route_state)
                    route_mcts_probs.append(route_probs)
                    route_latencies.append(route_latency)
            sum_latency = self.state.get_reward()
            rewards.append(pre_reward - sum_latency)
            values.append(sum_latency)
            if self.state.is_terminal():
                break
        if self.state.mapping_state:
            print("the mapping is success")
        return (
            self.state.get_reward(),
            zip(states, ppo_probs, rewards, values),
            zip(route_states, route_mcts_probs, route_latencies),
        )

    def ppo_mapping(self, agent, is_shown=False):
        self.env.reset()
        self.state = MappingState(self.env)
        sum_latency = 0
        latencies = []
        while True:
            act = agent.get_ppo_action(self.state)

            self.state.take_action(action=act, print_action=True)
            self.state.mode = "route"
            route_buffer = self.start_route(agent)
            self.state.mode = "placement"
            if route_buffer == None:
                continue
            sum_latency = self.state.get_reward()
            latencies.append(sum_latency)
            if self.state.is_terminal():
                if not self.state.mapping_state:
                    print("the mapping is failed")
                if is_shown:
                    self.state.draw()
                return sum_latency

    def use_true_reward(self):
        self.env.heristic_reward = False
