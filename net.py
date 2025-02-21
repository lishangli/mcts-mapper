import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from GAT import GAT
from typing import List, Tuple, Union
from state import MappingState


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


class GraphEmbedding(nn.Module):
    def __init__(
        self,
        dfg_shape: int,
        adg_shape: int,
        hidden_dim: int,
        gat_heads: int,
        dropout: int,
        leaky_slope: int,
        device,
    ):
        super().__init__()
        self.device = device
        # self.fc128 = nn.Linear(128, 32)
        # self.fc32 = nn.Linear(32, 32)

        self.dfg_encoder = nn.Sequential(
            nn.Linear(dfg_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )

        self.adg_encoder = nn.Sequential(
            nn.Linear(adg_shape, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
        )
        # self.dfg_reduce_fc = nn.Linear(32, 32)
        # self.adg_reduce_fc = nn.Linear(32, 32)
        # self.leakyrelu = nn.LeakyReLU(0.2)

        self.dfg_gat = GAT(
            dfg_shape, gat_heads, hidden_dim, dropout, leaky_slope, gat_heads
        )
        self.adg_gat = GAT(
            adg_shape, gat_heads, hidden_dim, dropout, leaky_slope, gat_heads
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LeakyReLU(leaky_slope),
        )

    @staticmethod
    def _prepare_batch_data(
        batch_state: List[MappingState], device: torch.device
    ) -> Tuple[torch.Tensor, ...]:
        """Prepare batch data for processing"""
        batch_data = {
            "dfg_features": [],
            "adg_features": [],
            "dfg_adj": [],
            "adg_adj": [],
            "dfg_meta": [],
            "adg_meta": [],
        }

        max_dfg_nodes = 30


        for state in batch_state:
            # Clean adjacency matrices
            dfg_adj = state.env.dfg_adj.copy()
            adg_adj = state.env.adg_adj.copy()
            dfg_adj[dfg_adj == -1] = 0
            adg_adj[adg_adj == -1] = 0

            # 2 dfg_features_vec padding
            dfg_features_vec = state.env.dfg_features_vec
            num_dfg_nodes = dfg_features_vec.shape[0]

            dfg_features_vec_padded = dfg_features_vec

            if num_dfg_nodes < max_dfg_nodes:
                dfg_features_vec_padded = np.pad(
                    dfg_features_vec,
                    ((0, max_dfg_nodes - num_dfg_nodes), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            dfg_adj_padded = dfg_adj 
            num_dfg_nodes_adj = dfg_adj_padded.shape[0]
            if num_dfg_nodes_adj < max_dfg_nodes:
                padding_rows_adj = max_dfg_nodes - num_dfg_nodes_adj 
                dfg_adj_padded = np.pad(dfg_adj, 
                                        ((0, padding_rows_adj), (0, padding_rows_adj)), mode = 'constant',
                                        constant_values=0,
                                        ) # 使用 np.pad 进行填充

            # Collect batch data
            batch_data["dfg_features"].append(dfg_features_vec_padded)
            batch_data["adg_features"].append(state.env.adg_features_vec)
            batch_data["dfg_adj"].append(dfg_adj_padded)
            batch_data["adg_adj"].append(adg_adj)
            batch_data["dfg_meta"].append(state.get_dfg_meta())
            batch_data["adg_meta"].append(state.get_adg_meta())

        # Convert to numpy arrays
        return tuple(
            torch.FloatTensor(np.array(v)).to(device) for v in batch_data.values()
        )

    def _process_graph_embeddings(
        self, features: torch.Tensor, adj: torch.Tensor, gat_layer: nn.Module
    ) -> torch.Tensor:
        """Process graph embeddings through GAT and reduction"""
        x = gat_layer(features, adj)
        return torch.mean(x, dim=1, keepdim=True)

    def forward(self, batch_state, mode="placement"):
        # dfg embedding
        # batch_dfg_features_vec = []
        # batch_adg_features_vec = []
        # batch_dfg_adj = []
        # batch_adg_adj = []
        # batch_adg_meta = []
        # batch_dfg_meta = []
        # for s in batch_state:
        #     batch_dfg_features_vec.append(s.env.dfg_features_vec)
        #     batch_adg_features_vec.append(s.env.adg_features_vec)
        #     s.env.dfg_adj[s.env.dfg_adj == -1] = 0
        #     batch_dfg_adj.append(s.env.dfg_adj)
        #     s.env.adg_adj[s.env.adg_adj == -1] = 0
        #     batch_adg_adj.append(s.env.adg_adj)
        #     batch_adg_meta.append(s.get_adg_meta())
        #     batch_dfg_meta.append(s.get_dfg_meta())

        # batch_dfg_features_vec = np.array(batch_dfg_features_vec)
        # batch_adg_features_vec = np.array(batch_adg_features_vec)
        # batch_dfg_adj = np.array(batch_dfg_adj)
        # batch_adg_adj = np.array(batch_adg_adj)
        # batch_dfg_meta = np.array(batch_dfg_meta)
        # batch_adg_meta = np.array(batch_adg_meta)

        # Prepare batch data
        (dfg_features, adg_features, dfg_adj, adg_adj, dfg_meta_data, adg_meta_data) = (
            self._prepare_batch_data(batch_state, self.device)
        )

        # Process DFG and ADG through GAT
        dfg_embedding = self._process_graph_embeddings(
            dfg_features, dfg_adj, self.dfg_gat
        )
        adg_embedding = self._process_graph_embeddings(
            adg_features, adg_adj, self.adg_gat
        )

        # dfg_x = torch.mean(dfg_x, dim=1, keepdim=True)
        # # dfg_x = self.dfg_reduce_fc(dfg_x)

        # # adg embedding
        # adg_x = self.adg_gat(
        #     torch.FloatTensor(batch_adg_features_vec).to(self.device),
        #     torch.FloatTensor(batch_adg_adj).to(self.device),
        # )
        # adg_x = torch.mean(adg_x, dim=1, keepdim=True)

        dfg_meta = self.dfg_encoder(dfg_meta_data)
        adg_meta = self.adg_encoder(adg_meta_data)

        dfg_meta = torch.mean(dfg_meta, dim=1, keepdim=True)
        adg_meta = torch.mean(adg_meta, dim=1, keepdim=True)

        # current mapped node information
        # meta = self.fc(env.dfg_features[env.last_action])
        # adg_x = self.adg_reduce_fc(adg_x)
        combined_features = torch.cat(
            [dfg_embedding, adg_embedding, dfg_meta, adg_meta], dim=2
        )

        return self.fc_layers(combined_features)


"""set leraning rate for optimizer"""


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


class PolicyNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=32):
        super().__init__()
        self.fc = nn.Linear(state_size, hidden_size)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)
        self.softmax = nn.Softmax(dim=1)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x, mask):
        x = F.relu(self.fc(x))
        x = self.leakyrelu(x)

        # x = F.relu(self.fc(x))
        x = F.relu(self.fc1(x))
        x = self.leakyrelu(self.fc2(x))
        # x = F.normalize(x, p=1, dim=1)

        x = x + (mask - 1) * 1e2

        x = x - torch.max(x, dim=2, keepdim=True)[0]
        # x = x.masked_fill(~mask, -float('inf'))
        # print("before softmax : {}".format(x))
        x = F.log_softmax(x, dim=2)
        # print("after softmax : {}".format(x))
        # print("x shape {}".format(x[...,1:100]))
        # print(x[...,1:100])
        # x = x.reshape(-1)
        x = x.squeeze(1)
        return x


class ValueNet(nn.Module):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.fc = nn.Linear(32, 1)
        self.drop = nn.Dropout(p=0.5)
        self.mlp = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 32))
        self.leakyrelu = nn.LeakyReLU(alpha)

    def forward(self, x):
        x = self.mlp(x)
        x = self.drop(x)
        x = self.leakyrelu(x)
        x = self.fc(x)
        return x


class PolicyValueNet(nn.Module):
    def __init__(
        self,
        dfg_shape,
        adg_shape,
        state_size,
        action_size,
        route_action_size,
        hidden_size=32,
        device=torch.device("cpu"),
    ):
        super().__init__()
        self.embedding = GraphEmbedding(
            dfg_shape,
            adg_shape,
            hidden_dim=hidden_size,
            gat_heads=8,
            dropout=0.3,
            leaky_slope=0.2,
            device=device,
        )
        self.policyNet = PolicyNet(state_size, action_size, hidden_size)
        self.routePolicyNet = PolicyNet(state_size, route_action_size, hidden_size)
        self.ppoNet = PolicyNet(state_size, action_size, hidden_size)
        self.valueNet = ValueNet()
        self.rewardNet = RewardNet(action_size, hidden_size)

    def forward(self, state, mask, mode="placement"):
        x = self.embedding(state, mode)
        # print("embedding {}".format(x))
        if mode == "reward":
            return self.rewardNet(x, mask)
        if mode == "route":
            log_act_probs = self.routePolicyNet(x, mask)
        elif mode == "ppo":
            log_act_probs = self.ppoNet(x, mask)
        else:
            log_act_probs = self.policyNet(x, mask)
        value = self.valueNet(x)
        # print("value {}", value)

        return log_act_probs, value


class RewardNet(nn.Module):
    def __init__(self, action_size, hidden_size=32):
        super().__init__()
        # self.conv = conv3x3(4,3)
        # self.bn = torch.nn.BatchNorm1d(3)
        # self.resblocks = torch.nn.ModuleList([
        #     ResidualBlock(3) for _ in range(num_blocks)
        # ])

        self.action_fc = nn.Linear(action_size, hidden_size)

        self.fc1 = nn.Linear(2 * hidden_size, 64)

        self.fc2 = nn.Linear(64, 1)

        self.action_vec = np.zeros(action_size)

    def forward(self, state, action):
        # state_feat = self.state_fc(state)
        # state_feat = F.relu(state_feat)

        action_feat = self.action_fc(action)
        action_feat = F.relu(action_feat)

        x = torch.cat((state, action_feat), dim=2)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)
        return x.view(-1)


class AgentNetwork:
    def __init__(
        self,
        dfg_shape,
        adg_shape,
        state_size,
        action_size,
        route_size,
        hidden_size=32,
        use_gpu=False,
        model_file=None,
    ):
        # self.graph_embedding = GraphEmbedding()
        super().__init__()
        self.use_gpu = use_gpu
        self.device = torch.device("cuda" if use_gpu else "cpu")

        self.agentNet = PolicyValueNet(
            dfg_shape,
            adg_shape,
            state_size,
            action_size,
            route_size,
            hidden_size,
            device=self.device,
        ).to(self.device)
        self.action_size = action_size
        self.optimizer = torch.optim.Adam(self.agentNet.parameters(), lr=0.01)

        if model_file:
            self.agentNet = torch.load(
                model_file, map_location=self.device
            )  # load_state_dict(net_params)

    def policy_value(self, batch_state):
        # ! state is a  batch list of states
        # ! fix this for batch processing
        # x = self.embedding(state_embedding)
        batch_mask = [state.env.mask.flatten() for state in batch_state]
        log_act_probs, value = self.agentNet(
            batch_state, torch.FloatTensor(batch_mask).unsqueeze(1).to(self.device)
        )
        act_probs = np.exp(log_act_probs.cpu().data.numpy())
        # print("shape : {} , sum act_probs is {}, act_probs is {}".format(act_probs.shape, np.sum(act_probs), act_probs))

        value = value.cpu().data.numpy()

        return act_probs, value

    def route_policy_value(self, batch_state):
        batch_mask = [state.env.route_mask for state in batch_state]
        log_act_probs, value = self.agentNet(
            batch_state,
            torch.FloatTensor(batch_mask).unsqueeze(1).to(self.device),
            mode="route",
        )
        act_probs = np.exp(log_act_probs.cpu().data.numpy())
        value = value.cpu().data.numpy()
        return act_probs, value

    def policy_value_pure(self, batch_state):
        batch_mask = np.array([state.mask.flatten() for state in batch_state])
        log_act_probs, value = self.agentNet(
            batch_state,
            torch.FloatTensor(batch_mask).unsqueeze(1).to(self.device),
            mode="placement",
        )
        ##########  act_probs = np.exp(log_act_probs.cpu().data.numpy())
        return log_act_probs, value

    def route_policy_value_pure(self, batch_state):
        # batch_env = batch_state
        batch_mask = np.array([state.env.route_mask for state in batch_state])
        log_act_probs, value = self.agentNet(
            batch_state,
            torch.FloatTensor(batch_mask).unsqueeze(1).to(self.device),
            mode="route",
        )
        return log_act_probs, value

    def policy_value_fn(self, state):
        legal_actions = state.get_actions()
        mask = state.mask * state.env.dfg_mask
        log_act_probs, value = self.agentNet(
            [state], torch.FloatTensor(mask.flatten()).to(self.device)
        )
        log_act_probs = log_act_probs.reshape(-1)
        act_probs = np.exp(log_act_probs.cpu().data.numpy())
        value = value.cpu().data.numpy()[0][0]
        act_probs = zip(legal_actions, act_probs[legal_actions])
        return act_probs, value

    def ppo_policy_value_fn(self, state):
        legal_actions = state.get_actions()
        mask = np.zeros_like(state.mask).flatten()
        mask[legal_actions] = 1
        log_act_probs, value = self.agentNet(
            [state], torch.FloatTensor(mask).to(self.device), mode="ppo"
        )
        log_act_probs = log_act_probs.reshape(-1)
        act_probs = np.exp(log_act_probs.cpu().data.numpy())
        value = value.cpu().data.numpy()[0][0]
        print(np.sum(act_probs[legal_actions] / np.sum(act_probs[legal_actions])))
        return (
            legal_actions,
            act_probs[legal_actions] / np.sum(act_probs[legal_actions]),
            value,
        )

    def reward_model(self, state, action):
        if isinstance(action, list):
            action = torch.tensor(action, dtype=torch.long)
        elif isinstance(action, int):
            action = torch.tensor([action], dtype=torch.long)
        elif torch.is_tensor(action):
            action = action.long()
        else:
            # print("action type is {}".format(type(action)))
            action = torch.tensor([action], dtype=torch.long)
        action_vec = torch.nn.functional.one_hot(
            action, num_classes=self.action_size
        ).unsqueeze(1)

        if not isinstance(state, list):
            state = [state]
        # 转换为 float 并放到指定设备
        action_vec = action_vec.float().to(self.device)

        reward = self.agentNet(
            state,
            action_vec,
            mode="reward",
        )

        reward = reward.cpu().data.numpy()
        return reward

    def reward_model_pure(self, state, action):
        if isinstance(action, list):
            action = torch.tensor(action, dtype=torch.long)
        elif isinstance(action, int):
            action = torch.tensor([action], dtype=torch.long)
        elif torch.is_tensor(action):
            action = action.long()
        else:
            # print("action type is {}".format(type(action)))
            action = torch.tensor([action], dtype=torch.long)
        action_vec = torch.nn.functional.one_hot(
            action, num_classes=self.action_size
        ).unsqueeze(1)

        if not isinstance(state, list):
            state = [state]
        # 转换为 float 并放到指定设备
        action_vec = action_vec.float().to(self.device)

        reward = self.agentNet(
            state,
            action_vec,
            mode="reward",
        )

        return reward

    def route_policy_value_fn(self, state):
        legal_actions = state.get_route_actions()
        mask = state.env.route_mask
        log_act_probs, value = self.agentNet(
            [state], torch.FloatTensor(mask.flatten()).to(self.device), mode="route"
        )
        log_act_probs = log_act_probs.reshape(-1)
        act_probs = np.exp(log_act_probs.cpu().data.numpy())
        value = value.cpu().data.numpy()[0][0]
        act_probs = zip(legal_actions, act_probs[legal_actions])
        return act_probs, value

    # def route_policy_value(self, batch_state):
    #     batch_env = [state.env for state in batch_state]
    #     batch_mask = np.array([state.env.route_mask for state in batch_state])
    #     log_act_probs, value = self.agentNet(
    #         batch_env,
    #         torch.FloatTensor(batch_mask).unsqueeze(1).to(self.device),
    #         mode="route",
    #     )
    #     return log_act_probs, value

    # def train_step(
    #     self,
    #     state_batch,
    #     mcts_probs,
    #     value_batch,
    #     reward_batch,
    #     action_batch,
    #     lr,
    # ):
    #     """perform a training step"""
    #     if self.use_gpu:
    #         state_batch = state_batch
    #         mcts_probs = torch.FloatTensor(mcts_probs).to(self.device)
    #         value_batch = torch.FloatTensor(value_batch).to(self.device)
    #         action_batch = torch.LongTensor(action_batch).to(self.device)
    #         reward_batch = torch.FloatTensor(reward_batch).to(self.device)
    #     # reset gradients
    #     self.optimizer.zero_grad()

    #     # set learning rate
    #     set_learning_rate(self.optimizer, lr)

    #     act_probs, value = self.policy_value_pure(state_batch)
    #     rewards = self.reward_model_pure(state_batch, action_batch)

    #     value_loss = F.mse_loss(value.view(-1), value_batch)

    #     policy_loss = -torch.mean(torch.sum(mcts_probs * act_probs, 1))
    #     # policy_loss = -torch.mean(torch.sum(reward_batch * act_probs, 1))
    #     reward_loss = F.mse_loss(rewards, reward_batch)
    #     print(state_batch)
    #     print("reward {}, pred reward {}".format(reward_batch, rewards))
    #     print("value is {}, pred value {}".format(value_batch, value.view(-1)))

    #     print(
    #         "value loss is {}, policy loss is {}, reward loss is {}".format(
    #             value_loss, policy_loss, reward_loss
    #         )
    #     )

    #     # backward
    #     loss = value_loss + policy_loss + reward_loss
    #     loss.backward()
    #     self.optimizer.step()

    #     # calc policy entropy, for monitoring only

    #     entropy = -torch.mean(torch.sum(act_probs * torch.exp(act_probs), 2))
    #     return loss.item(), entropy.item()

    def train_step(
        self,
        state_batch,
        old_act_probs,  # Changed from mcts_probs to store old policy probabilities
        value_batch,
        reward_batch,
        action_batch,
        lr,
    ):
        """Perform a PPO training step"""
        if self.use_gpu:
            state_batch = state_batch
            old_act_probs = torch.FloatTensor(old_act_probs).to(self.device)
            value_batch = torch.FloatTensor(value_batch).to(self.device)
            action_batch = torch.LongTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device)

        # Reset gradients
        self.optimizer.zero_grad()

        # Set learning rate
        set_learning_rate(self.optimizer, lr)

        # Get current policy and value predictions
        new_act_probs, value = self.policy_value_pure(state_batch)

        new_act_probs = torch.exp(new_act_probs).squeeze(1)
        rewards = self.reward_model_pure(state_batch, action_batch)

        # Calculate advantages
        advantages = value_batch - value.detach()
        new_act_probs = new_act_probs.squeeze(1)

        # Get probabilities for actually taken actions
        old_probs = torch.sum(
            old_act_probs
            * torch.nn.functional.one_hot(
                action_batch, num_classes=new_act_probs.shape[-1]
            ),
            dim=-1,
        )
        new_probs = torch.sum(
            new_act_probs
            * torch.nn.functional.one_hot(
                action_batch, num_classes=new_act_probs.shape[-1]
            ),
            dim=-1,
        )
        # print("new probs is {}, old probs is {}".format(new_probs, old_probs))

        # Calculate ratio of new and old probabilities
        ratio = (new_probs + 1e-8) / (old_probs + 1e-8)
        # print("ratio is {}".format(ratio))
        # PPO clipped objective function
        clip_epsilon = 0.2  # Hyperparameter for clipping
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages
        policy_loss = -torch.mean(torch.min(surr1, surr2))

        value_target = value_batch.detach()  # 防止梯度传播
        # Value loss (can also be clipped if desired)
        value_loss = F.mse_loss(value.view(-1), value_target)
        print("pred value is {}, target value is {}".format(value, value_target))

        # Reward model loss
        reward_loss = F.mse_loss(rewards, reward_batch)

        # Optional: add entropy bonus to encourage exploration
        # 1. Add small epsilon to prevent log(0)
        eps = 1e-8
        # 2. Clamp probabilities to prevent numerical instability
        log_probs = torch.clamp(new_act_probs, eps, 1.0)
        # 3. Calculate entropy using a numerically stable method
        entropy = -torch.mean(torch.sum(log_probs * torch.log(log_probs), dim=-1))

        # Combined loss
        loss = policy_loss + 0.05 * value_loss + 0.1 * reward_loss - 0.01 * entropy

        # print(f"State batch shape: {state_batch.shape}")
        # print(
        #     f"Reward: {reward_batch.mean():.4f}, Predicted reward: {rewards.mean():.4f}"
        # )
        # print(
        #     f"Value: {value_batch.mean():.4f}, Predicted value: {value.view(-1).mean():.4f}"
        # )
        print(
            f"Value loss: {value_loss:.4f}, Policy loss: {policy_loss:.4f}, "
            f"Reward loss: {reward_loss:.4f}, Entropy: {entropy:.4f}"
        )

        # print("value is {}, pred value is {}".format(value_batch, value.view(-1)))

        # Backward pass and optimization
        loss.backward()

        # Optional: add gradient clipping
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)

        self.optimizer.step()

        return loss.item(), entropy.item()

    def train_step_pretrain(
        self,
        state_batch: List[MappingState],
        old_act_probs: List[List[float]],
        value_batch: List[List[float]],
        reward_batch: List[List[float]],
        action_batch: List[List[int]],
        lr: float,
    ):
        """Perform a pre training step"""
        if self.use_gpu:
            state_batch = state_batch
            old_act_probs = torch.FloatTensor(old_act_probs).to(self.device)
            value_batch = torch.FloatTensor(value_batch).to(self.device)
            action_batch = torch.LongTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        else:
            state_batch = state_batch
            old_act_probs = torch.FloatTensor(old_act_probs).to(self.device)
            value_batch = torch.FloatTensor(value_batch).to(self.device)
            action_batch = torch.LongTensor(action_batch).to(self.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device)

        # Reset gradients
        self.optimizer.zero_grad()
        # Set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        log_act_probs, value = self.policy_value_pure(state_batch)
        value_loss = F.mse_loss(value.view(-1), value_batch)
        policy_loss = -torch.mean(torch.sum(old_act_probs * log_act_probs, 1))
        print("value loss is {}, policy loss is {}".format(value_loss, policy_loss))
        loss = 0.05 * value_loss + policy_loss
        loss.backward()
        self.optimizer.step()

        entropy = -torch.mean(torch.sum(log_act_probs * torch.exp(log_act_probs), 1))

        return loss.item(), entropy.item()

    def train_route_step(self, state_batch, mcts_probs, sum_latency, lr):
        """perform a training step"""
        if self.use_gpu:
            state_batch = state_batch
            mcts_probs = torch.FloatTensor(mcts_probs).to(self.device)
            sum_latency = torch.FloatTensor(sum_latency).to(self.device)
        else:
            state_batch = state_batch
            mcts_probs = mcts_probs
            sum_latency = sum_latency

        self.optimizer.zero_grad()

        # set learning rate
        set_learning_rate(self.optimizer, lr)

        log_act_probs, value = self.route_policy_value_pure(state_batch)
        act_probs = torch.exp(log_act_probs)
        entropy = -torch.mean(torch.sum(act_probs * log_act_probs, 1))
        value_loss = F.mse_loss(value.view(-1), sum_latency)
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_act_probs, 1))
        # print("value loss is {}, policy loss is {}".format(value_loss, policy_loss))
        # print("value is {}, pred value is {}".format(sum_latency, value.view(-1)))
        # print("mcts probs is {}, perd probs is {}".format(mcts_probs, act_probs))
        loss = 0.05 * value_loss + policy_loss - 0.01 * entropy
        loss.backward()
        self.optimizer.step()
        return loss.item(), entropy.item()
