from model import MCTS
import numpy as np
from model import Environment

# class MCTNode:
#     def __init__(self, state, parent=None):
#         self.state = state
#         self.parent = parent
#         self.children = []
#         self.visits = 0
#         self.value = 0
#         self.policy = None
    
#     def is_fully_expanded(self):
#         return len(self.children) == len(self.state.getActions()) or self.children == {}
    
#     def best_child(self, c_param=1.4):
#         choices_weights = [
#             (child.value / child.visits) + c_param * math.sqrt((2*math.log(self.visits) / child.visits)) for child in self.children
#         ]

#         return self.children[choices_weights.index(max(choices_weights))]
    
#     def expand(self, action_probs):
#         tried_actions = [child.state.last_action for action, child in self.children.items()]
#         legal_actions = self.state.get_policy_action(action_probs)
#         for action in legal_actions:
#             if action not in tried_actions:
#                 new_state = self.state.take_action(action)
#                 child = MCTNode(new_state, self)
#                 self.children[action] = child
    
#     def simulate(self):
#         current_state = self.state.clone()
#         while not current_state.isTerminal():
#             current_state = current_state.takeRandomAction()
#         return current_state.get_reward()
    
#     def backpropagate(self, result, ct):
#         self.visits += 1
#         if self.parent:
#             self.parent.backpropagate(self.value + result, ct+1)
#         self.value = result / ct 


# class MCTS:
#     def __init(self, state):
#         self.root = MCTNode(state)
#         self.exploration_weight = 1.4

#     def search(self, iterations, agent_net):
#         for _ in range(iterations):
#             node = self.select()
#             action_probs, value = agent_net(node.state)
#             node.policy = action_probs
#             if not node.state.isTerminal():
#                 node = node.expand(action_probs)

#             result = node.simulate()
#             node.backpropagate(result, 1)
    
#     def select(self):
#         node = self.root
#         while node.is_fully_expanded():
#             node = node.best_child()
#         return node


class MCTSPlayer(object):
    def __init__(self, policy_value_fn):
        self.mcts = MCTS(policy_value_fn)
    
    def get_action(self, state, temp=1e-3, return_prob=0):
        actions = state.get_actions()
        action_probs = np.zeros(len(state.env.dfg_features_vec) * len(state.env.adg_features_vec)) ## TODO: fix adg action nums
        if len(actions) > 0:
            acts, probs = self.mcts.get_move_probs(state, 10) ## 10000 iters
            action_probs[list(acts)] = probs
            action = np.random.choice(acts, p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs))))
            self.mcts.update_with_move(action)

            if return_prob:
                return action, action_probs
            else:
                return action 
        else:
            print("WARNING: the state has no legal actions")
            return -1, None

    def reset(self):
        self.mcts.update_with_move(-1)

    
