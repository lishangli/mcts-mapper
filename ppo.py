import numpy as np
from mcts import MCTS


class PPOPlayer(object):
    def __init__(self, agent_net, t=0):
        self.policy_fn = agent_net.ppo_policy_value_fn
        self.route_mcts = MCTS(agent_net, t, "route")

    def get_ppo_action(self, state, return_prob=0):
        acts, probs, value = self.policy_fn(state)
        print(np.sum(probs))
        if len(acts) > 0:
            pp = 0.9 * probs + 0.1 * np.random.dirichlet(0.3 * np.ones(len(probs)))
            action = np.random.choice(
                acts,
                p=(pp) / np.sum(pp),
            )

            # self.ppo.update_with_move(action)
            if return_prob:
                return action, probs
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
                p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs))),
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
