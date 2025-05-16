import numpy as np
import copy
import math
from line_profiler import profile
from typing import List

MAX_DISTANCE = 10000000000

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class MCTNode:
    def __init__(self, prob, distance, parent=None):
        # self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 1
        self.value = 0
        self.p = prob
        self.u = 0
        self.q = distance
        self.v = distance
        self.h = 1 / (distance + 1)  # heuristic value
        self.policy = None
        self.reward = 1
        self.discount = 0.99
    
    # def __hash__(self):
    #     return id(self)  # 使用对象地址作为哈希

    # def __eq__(self, other):
    #     return self is other  # 比较地址是否相同

    def is_fully_expanded(self, state):

        return len(self.children) >= len(state.get_actions()) or self.children == {}

    def is_leaf(self):
        return self.children == {}

    def best_child(self, t, c_param=0.05):
        # print("children nums: {}".format(len(self.children)))
        qa_std = np.std([child.q + self.discount * child.reward for a, child in self.children.items()])
        # print(f"std is {qa_std}")
        return max(self.children.items(), key=lambda x: x[1].get_value(c_param, qa_std))

    def get_value(self, c_param, t=1):
        alpha = math.sqrt(self.parent.visits) / (1 + self.visits)

        mcts_p = (self.visits+1e-8) / self.parent.visits
        kl = mcts_p * np.log(mcts_p/self.p)

        self.u = self.p * alpha * c_param + (self.q + self.discount * self.reward) #W- kl
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

    def expandv1(self, action_probs, state, mode, top_n=5):
        if mode == "placement":
            legal_actions = state.get_actions()
        else:
            legal_actions = state.get_route_actions()

        # 根据概率对 action_probs 进行排序 (降序)
        sorted_action_probs = sorted(action_probs, key=lambda item: item[1], reverse=True)

        # 选择概率最高的 top_n 个动作进行拓展
        actions_to_expand = sorted_action_probs[:top_n]

        for action, prob in actions_to_expand:
            if action not in self.children and action in legal_actions:
                distance = 10000000000
                if mode == "placement":
                    distance = state.get_distance(action)
                else:
                    distance = state.get_route_distance(action)
                child = MCTNode(prob, distance, self)
                self.children[action] = child

    def expandv2(self, action_probs, state, mode, top_n=10):
        if mode == "placement":
            legal_actions = state.get_actions()
        else:
            legal_actions = state.get_route_actions()
        # 创建一个列表来存储 (action, distance) 对
        action_distances = []
        for action, prob in action_probs:
            if action in legal_actions and action not in self.children:
                distance = 10000000000
                if mode == "placement":
                    distance = state.get_distance(action)
                else:
                    distance = state.get_route_distance(action)
                action_distances.append((action, distance , prob)) # 同时保存 action 和 distance

        # 根据 distance 对 action_distances 列表进行排序 (升序，因为我们想要距离小的优先)
        sorted_action_distances = sorted(action_distances, key=lambda item: item[1])

        # 选择距离最小的 top_n 个动作进行拓展
        actions_to_expand = sorted_action_distances[:top_n]

        total_prob = sum([prob for _, _, prob in actions_to_expand])
        # print([prob for _, _, prob in actions_to_expand])
        prob_model = []
        for action, distance, prob in actions_to_expand:
            prob_model.append(prob/total_prob)
            child = MCTNode(prob/total_prob, self.value, self)
            self.children[action] = child
        # print(f"prob net out is {prob_model}")

    def simulate(self, action_probs, state):
        current_state = copy.deepcopy(state)
        node = self
        while not current_state.is_terminal():
            node.expand(action_probs, current_state, mode="route")
            act, node = self.best_child(t=0)
            current_state.take_action(act)

        ## print("sim finish")
        if current_state.is_route_success():
            print("route success")
        return current_state.get_reward()

    def backpropagate(self, result, qf=0):
        """mcts td back up version"""
        self.visits += 1
        coff = 1.0 * (result-self.q + qf) / self.visits
        self.q += coff
        if self.parent:
            coff *= (self.visits-1)
            ret = self.discount * self.q +  self.reward
            self.parent.backpropagate(ret, coff)

    def backpropagate_v2(self, result):
        """mcts mc back up version"""
        self.visits += 1
        self.q += 1.0*(result - self.q) / self.visits 
        if self.parent:
            ret = self.discount * result + self.reward
            self.parent.backpropagate_v2(ret)
    

    def backpropagate_gae(self, advanatges):
        """update q using advantages"""

        if self.parent:
            delta = self.reward + self.discount * self.q - self.parent.q
            new_advantages = delta + self.discount * advanatges
            self.parent.backprogagate_gae(new_advantages)

        # update value
        self.q += (advanatges)/self.visits

""" MCT definition"""


class MCTS:
    def __init__(self, agent_net, t=0, mode="placement"):
        self.root = MCTNode(1.0, 1)
        self.agent_net = agent_net
        self.exploration_weight = 1.4
        self.mode = mode
        self.t = math.exp(-t)
        self.states = {}

    def step(self, t):
        self.t = 1/(1+math.exp(-t+10))

    @profile
    def search(self, state, alpha=1):
        reward = 0
        if self.mode == "placement":
            act, node = self.select(state)
            action_probs, value = self.agent_net.policy_value_fn(state)
        else:
            act, node = self.select(state)
            alpha = 1
            action_probs, value = self.agent_net.route_policy_value_fn(state)
            ## PRINT INFO FOR ACTION_PROBS

        # node.policy = action_probs
        node.value = value


        if self.mode == "placement":
            if not state.is_terminal():
                node.expandv2(action_probs, state, self.mode)
            else:
                
                # print(f"terminsl state in {state.mode} and state cost is {state.sum_reward}")
                value = 0
        else:
            if state.is_terminal():
                value = 0
            else:
                node.expandv2(action_probs, state, self.mode)
        node.backpropagate_v2(value)

    @profile
    def select(self, state):
        node = self.root
        act = None
        
        while not node.is_leaf() and not state.is_terminal():
            # print("route select ...")
            cur_latency = state.get_reward()
            act, node = node.best_child(self.t)
            ### node {node.reward, node.terminal, }

            if self.mode == "placement":
                state.take_action(act)
                if not self.route_process(state):
                    state.sum_reward -= 10
                    break
            else:
                state.take_route_action(act)
                

            node.reward = state.get_reward() - cur_latency
        return act, node
    
    def route_process(self,state):
        state.mode = "route"
        while not state.is_terminal():
            act = state.get_route_action()
            if act == -1:
                return True 
            elif act < 0:
                return False 

            state.take_route_action(act)
        state.mode = "placement"
        return True

    @profile
    def select_reward(self, state):
        node = self.root
        act = None
        while not node.is_leaf() and not state.is_terminal():
            cur_latency = state.get_reward()
            act, node = node.best_child(self.t)
            # if node.reward == -1:
            #     node.reward = self.agent_net.reward_model(state, act)[0]

            if self.mode == "placement":
                state.take_action(act)
            else:
                state.take_route_action(act)
            node.reward = state.get_reward() - cur_latency
        return act, node

    def get_move_probs(self, state, iters, temp=1):
        for i in range(iters):
            state_cp = copy.deepcopy(state)
            self.search(state_cp)

        act_visits = [(act, node.visits) for act, node in self.root.children.items()]
        if act_visits == []:
            return None, 0
        acts, visits = zip(*act_visits)
        visits = np.array(visits)
        print(f"root children visits is {visits}")
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))
        print(f"act probs is {act_probs}")
        return acts, act_probs
    
    def get_advantages(self):

        advantages = [node.q + node.reward - self.root.q for act, node in self.root.children.items()]
        return advantages

    def get_advantagesv2(self):
        values = [node.q + node.reward for act, node in self.root.children.items()]
        average_value = sum(values) / len(values) if values else 0
        advantages = [node.q + node.reward - average_value for act, node in self.root.children.items()]
        return advantages
    
    # def get_root_advantages(self) -> List[float]:
    #     print([child.q - self.root.q for key, child in self.root.children.items()])
    #     return [child.q - self.root.q for key, child in self.root.children.items()]

    def update_with_move(self, last_move):
        """update mcts root with last move"""
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = MCTNode(1.0, 1)


class place_MCTNode(MCTNode):

    def get_value(self, c_param, t):
        alpha = math.sqrt(self.parent.visits) / (1 + self.visits)
        self.u = (self.p * (1 - t) + self.h) * c_param * alpha + (self.q + 500) * (
            1 - t
        )

        return self.u

    def expand(self, action_probs, state):
        legal_actions = state.get_actions()
        for action, prob in action_probs:
            if action not in self.children and action in legal_actions:
                distance = MAX_DISTANCE
                distance = state.get_distance(action)
                child = place_MCTNode(prob, distance, self)
                self.children[action] = child


class route_MCTNode(MCTNode):
    def get_value(self, c_param, t):
        alpha = math.sqrt(self.parent.visits) / (1 + self.visits)
        self.u = self.h * alpha

        return self.u

    def expand(self, action_probs, state):
        legal_actions = state.get_route_actions()

        for action, prob in action_probs:
            if action not in self.children and action in legal_actions:
                distance = MAX_DISTANCE
                distance = state.get_route_distance(action)
                child = route_MCTNode(prob, distance, self)
                self.children[action] = child

    def simulate(self, action_probs, state):
        current_state = copy.deepcopy(state)
        node = self
        while current_state.is_terminal():
            node.expand(action_probs, current_state)
            act, node = self.best_child(t=0)
            current_state.take_action(act)

        if current_state.is_route_success():
            print("route success!")

        return current_state.get_reward()


class MCTSPlayer(object):
    def __init__(self, agent_net, t=0):
        self.mcts = MCTS(agent_net, t)
        self.route_mcts = MCTS(agent_net, t, "route")
        self.agent_net = agent_net
        self.t = t

    def step(self, step=1):
        self.t += step
        self.mcts.step(self.t)
        self.route_mcts.step(self.t)

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def reset_route_player(self):
        self.route_mcts.update_with_move(-1)
    
    
    @profile
    def get_action(self, state, temp=1e-3, return_prob=0, method="random"):
        actions = state.get_actions()
        action_probs = np.zeros(
            state.env.adg.getNodeNums() * len(state.env.adg_features_vec)
        )
        if len(actions) > 0:
            acts, probs = self.mcts.get_move_probs(state, 50)  ## 10000 iters
            if acts == None:
                return -1, None
            action_probs[list(acts)] = probs
            v = self.mcts.root.q
            # print("legal actions is {}, and  action_probs is {}".format(acts, probs))
            if method == "greedy":
                action = acts[np.argmax(probs)]
            else:
                action = np.random.choice(
                    acts,
                    p=0.95 * probs
                    + 0.05 * np.random.dirichlet(0.3 * np.ones(len(probs))),
                )
                # print("action is {}".format(action))
            print(f"children q values is {[child.q + child.reward for child in self.mcts.root.children.values()]}")
           
            self.mcts.update_with_move(action)
            q = 0.99*self.mcts.root.q +  self.mcts.root.reward
            print(f"self.mcts.root.reward is {self.mcts.root.reward}")
            print(f"v is {v} and qn is {self.mcts.root.q}, q is {q}, root u is {self.mcts.root.u - v}")

            if return_prob:
                return action, action_probs, q, v
            else:
                return action
        else:
            # print("WARNING: the state has no legal actions")
            return -1, None, None
        
    def get_actionv2(self, state, temp=1e-3, return_prob=0, method="random"):
        actions = state.get_actions()
        action_probs = np.zeros(
            state.env.adg.getNodeNums() * len(state.env.adg_features_vec)
        )
        if len(actions) > 0:
            acts, probs = self.mcts.get_move_probs(state, 20)  ## 10000 iters
            if acts == None:
                return -1, None
            action_probs[list(acts)] = probs
            v = self.mcts.root.q
            if method == "greedy":
                action = acts[np.argmax(probs)]
            else:
                action = np.random.choice(
                    acts,
                    p=0.95 * probs
                    + 0.05 * np.random.dirichlet(0.3 * np.ones(len(probs))),
                )
            print(f"children q values is {[child.q + child.reward for child in self.mcts.root.children.values()]}")
           
            self.mcts.update_with_move(action)
            # q = 0.99*self.mcts.root.q +  self.mcts.root.reward
            # print(f"self.mcts.root.reward is {self.mcts.root.reward}")
            # print(f"v is {v} and qn is {self.mcts.root.q}, q is {q}, root u is {self.mcts.root.u - v}")

            if return_prob:
                return action, acts, action_probs, self.mcts.get_advantagesv2()
            else:
                return action
        else:
            # print("WARNING: the state has no legal actions")
            return -1, None, None    

    @profile
    def get_route_action(self, state, temp=1e-3, return_prob=0):
        actions = state.get_route_actions()
        # print("route actions len {}".format(len(actions)))
        action_probs = np.zeros(len(state.env.adg.getEdges()))
        if len(actions) > 0:
            acts, probs = self.route_mcts.get_move_probs(state, 4)
            if acts == None:
                return -2, None
            action_probs[list(acts)] = probs
            action = np.argmax(action_probs) 
            # action = np.random.choice(
            #     acts,
            #     p=1.0 * probs + 0.0 * np.random.dirichlet(0.3 * np.ones(len(probs))),
            # )
            self.route_mcts.update_with_move(action)

            if return_prob:
                return action, action_probs
            else:
                return action
        else:
            if state.is_terminal():
                # print("WARNING: the state has no route legal actions")
                return -1, None
            # print("WARNING: the state don't need to route actions")
            return -2, None

    def get_ppo_action(self, state, return_prob=True, method="random"):
        actions = state.get_actions()
        actions_probs = np.zeros(
            state.env.adg.getNodeNums() * len(state.env.adg_features_vec)
        )
        if len(actions) > 0:
            probs, value = self.agent_net.policy_value_fn_pure(state)
            distances = [(action,state.get_distance(action)) for action in actions]
            distances = sorted(distances, key=lambda item: item[1])
            top_distances = distances[:10]
            top_a = [item[0] for item in top_distances]
            # print(f"new probs is {probs}")
            # actions_probs[actions] = probs[actions
            # probs = np.array(probs)
            legal_probs = probs[np.array(top_a)]
            legal_probs = np.asarray(legal_probs).astype("float64")
            normalized_probs = legal_probs / legal_probs.sum()
            actions_probs[top_a] = normalized_probs
            # n_probs = normalized_probs / np.sum(normalized_probs)
            if method == "greedy":
                action = actions[np.argmax(probs)]
            else:
                action = np.random.choice(
                    top_a,
                    p=1.0 * normalized_probs
                    + 0.0 * np.random.dirichlet(0.3 * np.ones(len(normalized_probs))),
                )
            if return_prob:
                return action, probs
            else:
                return action
        else:
            # print("WARNING: the state has no legal actions")
            return -1, None

    def get_route_ppo_action(self, state, return_prob=True, method="random"):
        actions = state.get_route_actions()
        # print("route actions len {}".format(len(actions)))
        action_probs = np.zeros(len(state.env.adg.getEdges()))
        if len(actions) > 0:
            probs, value = self.agent_net.route_policy_value_fn_pure(state)
            legal_probs = probs[np.array(actions)]
            legal_probs = np.asarray(legal_probs).astype("float64")
            normalized_probs = legal_probs / legal_probs.sum()
            action_probs[actions] = normalized_probs
            if method == "greedy":
                action = actions[np.argmax(probs)]
            else:
                action = np.random.choice(
                    actions,
                    p=1.0 * normalized_probs
                    + 0.0 * np.random.dirichlet(0.3 * np.ones(len(normalized_probs))),
                )

            if return_prob:
                return action, action_probs
            else:
                return action
        else:
            if state.is_terminal():
                # print("WARNING: the state has no route legal actions")
                return -1, None
            # print("WARNING: the state don't need to route actions")
            return -2, None
        
    def get_h_action(self, state, return_prob=True, method="random"):
        actions = state.get_actions()
        actions_probs = np.zeros(
            state.env.adg.getNodeNums() * len(state.env.adg_features_vec)
        )
        if len(actions) > 0:
            distances = [(action,state.get_distance(action)) for action in actions]
            distances = sorted(distances, key=lambda item: item[1])
            top_distances = distances[:20]
            top_a = [item[0] for item in top_distances]
            top_d = [item[1] for item in top_distances]
            weights = np.exp(-0.4 * np.array(top_d))
            # weights = 1 / (np.array(distances)**4 + 1e-5)
        
            # weights =  np.array(distances)
            # weights = 1 / (np.array(distances)**4 + 1e-5)
            probs = weights / np.sum(weights)
            # print(f"probs is {probs}")
            # print(f"probs is {probs}")
            actions_probs[top_a] = probs
            if method == "greedy":
                action = actions[np.argmax(probs)]
            else:
                action = np.random.choice(
                    top_a,
                    p=1.0 * probs
                    + 0.0 * np.random.dirichlet(0.3 * np.ones(len(probs))),
                )
            if return_prob:
                    return action, actions_probs
            else:
                return action
        else:
            if state.is_terminal():
                # print("WARNING: the state has no route legal actions")
                return -1, None
            # print("WARNING: the state don't need to route actions")
            return -2, None
        
    def get_h_action_r(self, state, return_prob=True, method="random"):
        actions = state.get_route_actions()
        action_probs = np.zeros(len(state.env.adg.getEdges()))
        if len(actions) > 0:
            distances = [state.get_route_distance(action) for action in actions]
            weights = np.exp(-4 * np.array(distances))
            # weights = 1 / (np.array(distances)**4 + 1e-5)
            
            probs =  weights / np.sum(weights)
            action_probs[actions] = probs
            if method == "greedy":
                action = actions[np.argmax(probs)]
            else:
                action = np.random.choice(
                    actions,
                    p=1.0 * probs
                    + 0.0 * np.random.dirichlet(0.3 * np.ones(len(probs))),
                )
            if return_prob:
                    return action, action_probs
            else:
                return action
        else:
            if state.is_terminal():
                # print("WARNING: the state has no route legal actions")
                return -1, None
            # print("WARNING: the state don't need to route actions")
            return -2, None


    def reset(self):
        self.mcts.update_with_move(-1)
