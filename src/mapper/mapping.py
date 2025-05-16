from parser import *
from env import Environment, MappingState
from utils import GraphVisual
import numpy as np
from line_profiler import profile
import copy


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
        route_actions = []
        rewards = []
        step = 0
        max_step = 2000

        # Get policy value function from agent
        policy_value_net = agent.mcts.agent_net.route_policy_value_fn

        while step < max_step:
            step = step + 1
            move, move_probs = agent.get_h_action_r(
                self.state, return_prob=1
            )

            if move == -1:
                return None
            elif move == -2:
                self.state.sum_reward -= 500
                return  None

            cur_state = copy.deepcopy(self.state)
            states.append(cur_state)
            mcts_probs.append(move_probs)
            route_actions.append(move)

            srcN = self.state.env.adg.getEdges()[move].getSrcId()
            dstN = self.state.env.adg.getEdges()[move].getDstId()
            # print("cur route action is {} to {}".format(srcN, dstN))

            self.state.take_route_action(move)

            sum_latency = self.state.get_reward()
            # probs,next_value = policy_value_net(self.state)

            route_latencies.append(cur_state.get_reward())
            rewards.append(sum_latency - cur_state.get_reward())
            if self.state.is_terminal():
                # self.state.mode = "placement"
                # if len(self.state.env.route_cur_node) > 0:
                #     self.state.env.route_mask = route_mask
                #     self.state.adg_adj = adg_adj
                #     self.state.adg_features_vec = adg_features_vec
                #     self.route_step = route_step
                agent.reset_route_player()
                return zip(states, mcts_probs, route_latencies, rewards, route_actions)

    # @profile
    # def start_mapping(self, agent, print_action=False):
    #     self.env.reset()
    #     self.state = MappingState(self.env)
    #     self.state.mode = "placement"
    #     policy_value_net = agent.mcts.agent_net.policy_value_fn
    #     states, mcts_probs, current_players = [], [], []
    #     route_states, route_mcts_probs, route_latencies = [], [], []

    #     latencies, rewards, actions = [], [], []

    #     step = 0
    #     while True:

    #         step = step + 1
    #         move, move_probs = agent.get_action(self.state, temp=1e-2, return_prob=1)
    #         # print(move_probs)
    #         if move == -1:
    #             print("cur move is -1")
    #             break
    #         # print("cur move is {}".format(move))
    #         states.append(self.state)
    #         mcts_probs.append(move_probs)
    #         pre_reward = self.state.get_reward()

    #         actions.append(move)
    #         # print("pre_reward is {}".format(pre_reward))
    #         if print_action:
    #             print(
    #                 "step {}, select dfg node {} on adg node {}".format(
    #                     step, move[0], move[1]
    #                 )
    #             )

    #         self.state.take_action(move)  # take placement action
    #         latencies.append(pre_reward)
    #         self.state.mode = "route"
    #         route_buffer = self.start_route(agent)  # take routing actions
    #         self.state.mode = "placement"
    #         if route_buffer == None:
    #             continue
    #         else:
    #             for route_state, route_probs, route_latency in route_buffer:
    #                 route_states.append(route_state)
    #                 route_mcts_probs.append(route_probs)
    #                 route_latencies.append(route_latency)
    #         # rewards.append(self.state.get_reward())
    #         reward = self.state.get_reward() - pre_reward
    #         rewards.append(reward)
    #         act, next_value = policy_value_net(self.state)
    #         print("cur latency is {}, pred latency is {}".format(self.state.get_reward(), next_value))
    #         # latencies.append(reward + next_value)
    #         # print("cur reward is {}".format(pre_reward - self.state.get_reward()))
    #         if self.state.is_terminal():
    #             agent.reset_player()
    #             break

    #     final_val = self.state.get_reward()
    #     reversed_latencies = [final_val - latency for latency in latencies]
    #     reversed_route_latencies = [
    #         final_val - route_latency for route_latency in route_latencies
    #     ]
    #     print(latencies)
    #     print(route_latencies)
    #     # for route_latency in route_latencies:
    #     #     route_latency = final_val - route_latency

    #     # for latency in latencies:
    #     #     print("latency is {}".format(latency))
    #     # for route_latency in route_latencies:
    #     #     print("route_latency is {}".format(route_latency))
    #     if self.state.mapping_state:
    #         print("the mapping is success")
    #     return (
    #         self.state.get_reward(),
    #         zip(states, mcts_probs, reversed_latencies, rewards, actions),
    #         zip(route_states, route_mcts_probs, reversed_route_latencies),
    #     )

    def compute_gae(self, values, rewards, gamma=0.99, lam=0.99):
        advantages = np.zeros_like(rewards, dtype = np.float32)
        last_advantage = 0
        next_values = np.append(values[1:], [0])
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_values[t] - values[t]
            # print(f"delta is {delta} and reward is {rewards[t]}, and next value is {next_values[t]} and cur value is {values[t]}")
            advantages[t] = delta + gamma * lam * last_advantage
            last_advantage = advantages[t]
        
        value_target = [advantages[i] + values[i] for i in range(len(advantages))]

        return advantages.tolist(), value_target
    
    def compute_grpo_gae(self, values, qvalues,rewards, gamma=0.99, lam=0.99):
        advantages = np.zeros_like(qvalues, dtype = np.float32)
        last_advantage = 0
        next_values = np.append(values[1:], [0])
        mcv = np.zeros_like(qvalues, dtype = np.float32)
        v=0
        for t in reversed(range(len(qvalues))):
            delta = qvalues[t] - values[t]
            advantages[t] = delta + gamma * lam * last_advantage
            last_advantage = advantages[t]
            
            v *= gamma
            v += rewards[t]
            mcv[t] = v
            # print(f"delta is {delta} and gae is {advantages[t]}, mcv is {mcv[t]}, values is {values[t]} ")
        
        value_target = [mcv[i] for i in range(len(advantages))]

        return advantages.tolist(), value_target

    def start_mapping(self, agent, print_action=False, max_steps=1000, is_ppo=False):
        """
        Start the mapping process with improved error handling and resource management.

        Args:
            agent: The agent that provides actions
            print_action: Boolean flag to enable/disable action printing
            max_steps: Maximum number of steps to prevent infinite loops

        Returns:
            tuple: (final_reward, placement_data, routing_data)
        """
        try:
            # Initialize environment
            self.env.reset()
            self.state = MappingState(self.env)
            self.state.mode = "placement"
            policy_value_net = agent.mcts.agent_net.policy_value_fn_pure

            # Data collection containers
            states, mcts_probs, latencies, rewards, actions = [], [], [], [], []
            mcts_q = []
            (
                route_states,
                route_mcts_probs,
                route_latencies,
                route_rewards,
                route_actions,
            ) = ([], [], [], [], [])
            get_action = agent.get_action
            if is_ppo:
                get_action = agent.get_ppo_action
                print("is ppo")
            step = 0
            while step < max_steps:
                step += 1

                # Get action from agent
                try:
                    move, move_probs, q= get_action(self.state, return_prob=1)
                except Exception as e:
                    print(f"Error getting action: {e}")
                    break

                # Check if agent returned invalid move
                if move == -1:
                    if print_action:
                        print("Stopping: Agent returned -1 move")
                    break

                # Record pre-action state
                cur_state = copy.deepcopy(self.state)
                states.append(cur_state)
                mcts_probs.append(move_probs)
                # mcts_q.append(q)
                pre_reward = self.state.get_reward()
                actions.append(move)
                latencies.append(pre_reward)

                # Print action if requested
                if print_action:
                    print(
                        f"Step {step}, select dfg node {move[0]} on adg node {move[1]}"
                    )

                # Take placement action
                try:
                    self.state.take_action(move)
                    # Get value prediction
                    # act, next_value = policy_value_net(self.state)
                    # Calculate and record rewards
                    reward = self.state.get_reward() - pre_reward
                    rewards.append(reward)
                except Exception as e:
                    print(f"Error taking action: {e}")
                    break

                # Handle routing
                self.state.mode = "route"
                try:
                    route_buffer = self.start_route(agent)
                except Exception as e:
                    print(f"Error during routing: {e}")
                    self.state.mode = "placement"  # Restore mode before continuing
                    continue

                self.state.mode = "placement"

                # Process routing results
                if route_buffer is not None:
                    for (
                        route_state,
                        route_probs,
                        route_latency,
                        route_reward,
                        route_action,
                    ) in route_buffer:
                        route_states.append(route_state)
                        route_mcts_probs.append(route_probs)
                        route_latencies.append(route_latency)
                        route_rewards.append(route_reward)
                        route_actions.append(route_action)

                # if print_action:
                print(
                    f"Current latency: {self.state.get_reward()}, predicted latency: {q}"
                )

                # Check if terminal state reached
                if self.state.is_terminal():
                    agent.reset_player()
                    break

            # Handle max steps reached
            if step >= max_steps:
                print(
                    f"Warning: Maximum steps ({max_steps}) reached without termination"
                )

            # Calculate final values
            final_val = self.state.get_reward()
            reversed_latencies = [(final_val - latency) for latency in latencies]
            reversed_route_latencies = [
                final_val - route_latency for route_latency in route_latencies
            ]
            print(reversed_latencies)
            print(reversed_route_latencies)
            ##! TODO: here need change!
            # advantages = self.compute_gae(latencies, rewards)
            # route_advantages = self.compute_gae(route_latencies, route_rewards)

            # Log results
            if print_action:
                print(f"Placement latencies: {latencies}")
                print(f"Routing latencies: {route_latencies}")
                if self.state.mapping_state:
                    print("The mapping is successful")

            # Convert iterators to lists for multiple use
            placement_data = list(
                zip(states, mcts_probs, reversed_latencies, rewards, actions)
            )
            routing_data = list(
                zip(
                    route_states,
                    route_mcts_probs,
                    reversed_route_latencies,
                    route_rewards,
                    route_actions,
                )
            )

            return final_val, placement_data, routing_data

        except Exception as e:
            print(f"Unexpected error in start_mapping: {e}")
            # Clean up resources if needed
            return None, [], []

    @profile
    def test_mapping(self, agent, is_shown=False, temp=1e-2):
        self.env.reset()
        self.state = MappingState(self.env)
        sum_latency = 0
        latencies = []
        g_visual = GraphVisual(self.env.dfg, self.env.adg)
        self.state.start_anime(g_visual)
        policy_value_net = agent.mcts.agent_net.policy_value_fn_pure
        print("2")
        while True:
            print("4")
            move, move_probs, q, v = agent.get_action(self.state, return_prob=True)
            print("5")
            if move == -1:
                print("cur move is -1")
                if is_shown:
                    self.state.end_anime(g_visual)
                break
            print("3")
            # print(
            #     "cur action is {}, {}".format(
            #         move // self.env.adg_actions, move % self.env.adg_actions
            #     )
            # )
            cur_state = copy.deepcopy(self.state)
            self.state.take_action(action=move, print_action=True)
            self.state.mode = "route"
            route_buffer = self.start_route(agent)  # take routing actions
            self.state.mode = "placement"
            self.state.draw(g_visual)
            sum_latency = cur_state.get_reward()
            act, next_value = policy_value_net(cur_state)
            reward = self.state.get_reward() - cur_state.get_reward()
            print(
                "cur latency is {}, pred latency is {}, reward is {}".format(sum_latency, next_value, reward)
            )
            if route_buffer == None:
                continue
            latencies.append(sum_latency)
            if self.state.is_terminal():
                if not self.state.mapping_state:
                    print("the mapping is failed")
                # print("the mapping state latency is {}".format(sum_latency))
                # print(np.where(self.state.env.route_mask == 0)[0].shape)
                # print(np.where(self.state.route_mask == 0)[0].shape)
                self.state.dump_config()
                if is_shown:
                    self.state.end_anime(g_visual)
                return sum_latency

    def ppo_route(self, agent):
        self.state.mode = "route"
        values = []
        states = []
        rewards = []
        ppo_probs = []
        actions = []
        step = 0
        max_step = 2000

        policy_value_net = agent.mcts.agent_net.policy_value_fn_pure

        while step < max_step:
            step += 1
            move, probs = agent.get_route_ppo_action(self.state, return_prob=True)
            if move < 0:
                return None
            actions.append(move)
            cur_state = copy.deepcopy(self.state)
            states.append(cur_state)
            ppo_probs.append(probs)

            self.state.take_route_action(move)

            probs, value = policy_value_net(cur_state)

            rewards.append(self.state.get_reward() - cur_state.get_reward())
            values.append(value)

            if self.state.is_terminal():
                return zip(states, ppo_probs, values, rewards, actions)

    @profile
    def ppo_mapping(self, agent, print_action=False, is_shown=False, max_steps=1000):
        self.env.reset()
        self.state = MappingState(self.env)
        self.state.mode = "placement"
        sum_latency = 0
        rewards = []
        values = []
        actions = []
        latencies = []
        states, ppo_probs, current_players = [], [], []
        route_states, route_ppo_probs, route_values, route_rewards, route_actions = (
            [],
            [],
            [],
            [],
            [],
        )
        step = 0
        # Get policy value function from agent
        policy_value_net = agent.mcts.agent_net.policy_value_fn_pure
        while step < max_steps:
            step += 1

            try:
                act, probs = agent.get_ppo_action(self.state, return_prob=True)
            except Exception as e:
                print(f"Error getting action: {e}")
                break

            # Check if agent returned invalid move
            if act == -1:
                if print_action:
                    print("Stopping: Agent returned -1 move")
                break

            # Record pre-action state
            cur_state = copy.deepcopy(self.state)
            states.append(cur_state)
            ppo_probs.append(probs)
            # print(f"ppo probs is {probs}")
            pre_reward = self.state.get_reward()
            actions.append(act)
            latencies.append(pre_reward)

            self.state.take_action(action=act)
           
            self.state.mode = "route"
            route_buffer = self.start_route(agent)
            self.state.mode = "placement"
            sum_latency = self.state.get_reward()
            reward = sum_latency - pre_reward
            rewards.append(reward)
            probs, value = policy_value_net(cur_state)
            values.append(value)
            # print(f"pred val is {value}, reward is {reward}")
            if route_buffer == None:                    
                continue
            else:
                for (
                    route_state,
                    route_probs,
                    route_value,
                    route_reward,
                    route_action,
                ) in route_buffer:
                    route_states.append(route_state)
                    route_ppo_probs.append(route_probs)
                    route_values.append(route_value)
                    route_rewards.append(route_reward)
                    route_actions.append(route_action)
            if self.state.is_terminal():
                break
        # Log results
        if print_action:
            if self.state.mapping_state:
                print("The mapping is successful")
        # print(f"a1")
        advantages, values_target = self.compute_gae(values, rewards)
        route_advantages, route_values_target = self.compute_gae(route_values, route_rewards)
        reversed_value = [self.state.get_reward() - latency for latency in latencies]
        # print(f"rev value is {reversed_value}, gae value is {values_target}")
        # # reversed_route_values = [
        #     self.state.get_reward() - route_value for route_value in route_values
        # ]
        # print(f"a2")
        # print(type(advantages))
        # print(type(route_advantages))
        placement_data = list(zip(states, ppo_probs, values_target, advantages, actions))
        routing_data = list(
            zip(
                route_states,
                route_ppo_probs,
                route_values_target,
                route_advantages,
                route_actions,
            )
        )
        # print(f"data size is {len(states)}, route data size is {len(route_states)}")
        # print(f"a3")
        return self.state.get_reward(), placement_data, routing_data

    def grpo_mapping(self, agent, print_action=False, max_steps=1000):
        self.env.reset()

        self.state = MappingState(self.env)
        self.state.mode = "placement"
        step, sum_latency = 0, 0
        rewards, values, actions, latencies, states, grpo_probs,qs = [], [], [], [], [], [],[]

        route_states, route_grpo_probs, route_values, route_rewards, route_actions = [], [], [], [], []

        step = 0
        policy_value_net = agent.mcts.agent_net.policy_value_fn_pure

        while step < max_steps:
            step += 1 

            try:
                act, probs, q,v = agent.get_action(self.state, return_prob=True)
            except Exception as e:
                print(f"Error getting action: {e}")
                break


            # Check if agent returned invalid move
            if act == -1:
                if print_action:
                    print("Stopping: Agent returned -1 move")
                break

            # Record pre-action state
            cur_state = copy.deepcopy(self.state)
            states.append(cur_state)
           
            # print(f"ppo probs is {probs}")
            pre_reward = self.state.get_reward()
            actions.append(act)
            latencies.append(pre_reward)

            self.state.take_action(action=act)
           
            self.state.mode = "route"
            route_buffer = self.start_route(agent)
            self.state.mode = "placement"
            sum_latency = self.state.get_reward()
            reward = sum_latency - pre_reward
            qs.append(q)
            rewards.append(reward)
            values.append(v)
            probs_, value = policy_value_net(cur_state)
            grpo_probs.append(probs)
            # values.append(value)
            print(f"pred val is {value}, reward is {reward}")
            if route_buffer == None:                    
                continue
            else:
                for (
                    route_state,
                    route_probs,
                    route_value,
                    route_reward,
                    route_action,
                ) in route_buffer:
                    route_states.append(route_state)
                    route_grpo_probs.append(route_probs)
                    route_values.append(route_value)
                    route_rewards.append(route_reward)
                    route_actions.append(route_action)
            if self.state.is_terminal():
                break
        # Log results
        if print_action:
            if self.state.mapping_state:
                print("The mapping is successful")
        # print(f"a1")
        reversed_value = [self.state.get_reward() - latency for latency in latencies]

        advantages, values_target = self.compute_grpo_gae(values,qs, rewards)
        for rv, vt, vs,a, p in zip(reversed_value, values_target, values, advantages, grpo_probs):
            print(f"mc value is {rv} and av value is {vt}, mcts value is {vs}. a is {a}")
            print(f"probs is {p[p>0]}")
        route_advantages, route_values_target = self.compute_gae(route_values, route_rewards)
        # print(f"rev value is {reversed_value}, gae value is {values_target}")
        # # reversed_route_values = [
        #     self.state.get_reward() - route_value for route_value in route_values
        # ]
        # print(f"a2")
        # print(type(advantages))
        # print(type(route_advantages))
        placement_data = list(zip(states, grpo_probs, values_target, advantages, actions))
        routing_data = list(
            zip(
                route_states,
                route_grpo_probs,
                route_values_target,
                route_advantages,
                route_actions,
            )
        )
        # print(f"data size is {len(states)}, route data size is {len(route_states)}")
        # print(f"a3")
        return self.state.get_reward(), placement_data, routing_data

    def grpo_mappingv2(self, agent, print_action=False, max_steps=1000):
        self.env.reset()

        self.state = MappingState(self.env)
        self.state.mode = "placement"
        step, sum_latency = 0, 0
        rewards, values, actions, latencies, states, grpo_probs,qs = [], [], [], [], [], [],[]
        advantages = []
        route_states, route_grpo_probs, route_values, route_rewards, route_actions = [], [], [], [], []

        step = 0
        policy_value_net = agent.mcts.agent_net.policy_value_fn_pure

        while step < max_steps:
            step += 1 

            try:
                act, acts, probs, avts= agent.get_actionv2(self.state, return_prob=True)
            except Exception as e:
                print(f"Error getting action: {e}")
                break


            # Check if agent returned invalid move
            if act == -1:
                if print_action:
                    print("Stopping: Agent returned -1 move")
                break

            # Record pre-action state
            cur_state = copy.deepcopy(self.state)
            states.append(cur_state)

            print(f"act size is {len(acts)}")
           
            # print(f"ppo probs is {probs}")
            pre_reward = self.state.get_reward()
            actions.append(acts)
            latencies.append(pre_reward)

            self.state.take_action(action=act)
           
            self.state.mode = "route"
            route_buffer = self.start_route(agent)
            self.state.mode = "placement"
            sum_latency = self.state.get_reward()
            reward = sum_latency - pre_reward
            rewards.append(reward)
            if len(avts) == 10:
                advantages.append(avts)
            values.append(agent.mcts.root.q)

            grpo_probs.append(probs)
            if route_buffer == None:                    
                continue
            else:
                for (
                    route_state,
                    route_probs,
                    route_value,
                    route_reward,
                    route_action,
                ) in route_buffer:
                    route_states.append(route_state)
                    route_grpo_probs.append(route_probs)
                    route_values.append(route_value)
                    route_rewards.append(route_reward)
                    route_actions.append(route_action)
            if self.state.is_terminal():
                break
        # Log results
        if print_action:
            if self.state.mapping_state:
                print("The mapping is successful")
        # print(f"a1")
        reversed_value = [self.state.get_reward() - latency for latency in latencies]

        # advantages, values_target = self.compute_grpo_gae(values,qs, rewards)
        # for rv, vt, vs,a, p in zip(reversed_value, values_target, values, advantages, grpo_probs):
        #     print(f"mc value is {rv} and av value is {vt}, mcts value is {vs}. a is {a}")
        #     print(f"probs is {p[p>0]}")
        route_advantages, route_values_target = self.compute_gae(route_values, route_rewards)
        
        placement_data = list(zip(states, grpo_probs, reversed_value, advantages, actions))
        routing_data = list(
            zip(
                route_states,
                route_grpo_probs,
                route_values_target,
                route_advantages,
                route_actions,
            )
        )
        return self.state.get_reward(), placement_data, routing_data

    def my_mapping(self, agent, print_action=False, max_steps=1000):
        self.env.reset()

        self.state = MappingState(self.env)
        self.state.mode = "placement"
        step, sum_latency = 0, 0

        rewards, values, actions, latencies, states, probs = [], [], [], [], [], []

        route_states, route_values, route_values, route_rewards, route_actions = [], [], [], [], []

        step = 0

        policy_value_net = agent.mcts.agent_net.policy_value_fn_pure

        while step < max_steps:
            step += 1

            try:
                act, _ = agent.get_new_action(self.state, return_prob=True)
            except Exception as e:
                print(f"Error getting action: {e}")
                break 

            if act == -1:
                if print_action:
                    print("Stopping: Agent returned -1 move")
                break 

            cur_state = copy.deepcopy(self.state)
            states.append(cur_state)

            pre_reward = self.state.get_reward()
            actions.append(act)
            latencies.append(pre_reward)

            self.state.take_action(action=act)

            self.state.mode = "route"
            route_buffer = self.start_route(agent)
            self.state.mode = "placement"
            sum_latency = self.state.get_reward()
            reward = sum_latency - pre_reward
            rewards.append(reward)
            probs, _ = policy_value_net(cur_state)

            probs.append(probs)

            if  route_buffer == None:
                continue
            else:
                for (
                    route_state,
                    route_probs,
                    route_value,
                    route_reward,
                    route_action,
                ) in route_buffer:
                    route_states.append(route_state)
                    route_probs.append(route_probs)
                    route_values.append(route_value)
                    route_rewards.append(route_reward)
                    route_actions.append(route_action)
            if self.state.is_terminal():
                break

        # Log results
        if print_action:
            if self.state.mapping_state:
                print("The mapping is successful")
        # print(f"a1")
        reversed_value = [self.state.get_reward() - latency for latency in latencies]
        advantages, values_target = self.compute_gae(values, rewards)
        route_advantages, route_values_target = self.compute_grpo_gae(route_values,latencies, route_rewards)
        # print(f"rev value is {reversed_value}, gae value is {values_target}")
        # # reversed_route_values = [
        #     self.state.get_reward() - route_value for route_value in route_values
        # ]
        # print(f"a2")
        # print(type(advantages))
        # print(type(route_advantages))
        placement_data = list(zip(states, probs, values_target, advantages, actions))
        routing_data = list(
            zip(
                route_states,
                route_probs,
                route_values_target,
                route_advantages,
                route_actions,
            )
        )
        # print(f"data size is {len(states)}, route data size is {len(route_states)}")
        # print(f"a3")
        return self.state.get_reward(), placement_data, routing_data


    def use_true_reward(self):
        self.env.heristic_reward = False
