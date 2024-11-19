from adgParser import *
from dfgParser import * 
from model import Environment, MappingState
from mcts import MCTSPlayer
import matplotlib.pyplot as plt


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
    def __init__(self, dfg, adg, ops):
        self.env = Environment(dfg, adg, ops)
        self.state = MappingState(self.env)
        # self.mcts = MCTSPlayer(agent_net)

    def start_mapping(self, agent, print_action = False):
        self.env.reset()
        self.state = MappingState(self.env)
        states, mcts_probs, current_players = [], [], []
        rewards =[]
        sum_latency = 0
        step = 0
        while True:
            # print("#")
            step = step + 1
            move, move_probs = agent.get_action(self.state, temp=1e-3, return_prob=1)
            # if move == -1:
            #     break
            states.append(self.state)
            mcts_probs.append(move_probs)
            # current_players.append(self.env.current_player)
            if print_action:
                print("step {}, select dfg node {} on adg node {}".format(step, move[0], move[1]))
            self.state.take_action(move)
            # rewards.append(self.state.get_reward())
            sum_latency -= self.state.reward[move]
            rewards.append(self.state.reward[move])
            # print(rewards)
            if self.state.is_terminal():
                 latency = [sum_latency - reward for reward in rewards ]
                 
                 return sum_latency, zip(states, mcts_probs, latency)

    def test_mapping(self, agent, is_shown=False, temp=1e-2):
        self.env.reset()
        self.state = MappingState(self.env)
        sum_latency = 0
        while True:
            move, move_probs = agent.get_action(self.state, temp=temp, return_prob=1)
            print(len(self.state.get_actions()))
            if move == -1:
                break
            print("cur action is {}, {}".format(move//self.env.adg_actions, move%self.env.adg_actions))
            # if is_shown:
            #     self.state.draw()
            self.state.take_action(action = move, print_action=True)
            print(len(self.state.get_actions()))
            print("cur reward is {}".format(self.state.reward[move]))
            sum_latency += self.state.reward[move]
            # print("cur action is {}".format(move))
            if self.state.is_terminal():
                if not self.state.mapping_state:
                    print("the mapping is failed")
                    # sum_latency
                print("the mapping state latency is {}".format(sum_latency))
                if is_shown:
                    self.state.draw()
                return sum_latency
