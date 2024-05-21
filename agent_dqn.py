from agent_base import BaseAgent
from tetris.env import TetrisEnv
from tetris.state import TetrisBoard
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

class DQNNetwork(nn.Module):
    def __init__(self):
        super(DQNNetwork, self).__init__()

        # 20 x 10 "image" tensor input of 1s/0s
        # the current falling piece will be represented with 0.5s on the grid
        # Mnih et al (2015) suggests mapping state to multiple actions to reduce need for multiple forward passes...
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.dropout1 = nn.Dropout2d(p=0.25)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3)
        self.dropout2 = nn.Dropout2d(p=0.25)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3)
        self.dropout3 = nn.Dropout2d(p=0.25)

        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.dropout4 = nn.Dropout2d(p=0.25)
        
        self.fc5 = nn.Linear(128 * 200, 256)
        self.dropout5 = nn.Dropout(p=0.25)

        self.fc6 = nn.Linear(256, 128)
        self.dropout6 = nn.Dropout(p=0.25)

        self.fc7 = nn.Linear(128, 1)
        
        # kaiming/he initialization
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def pad_boundary(self, x):
        # treat the sides/bottom as "filled" and the top as "open"
        x = F.pad(x, (1, 1, 1, 1), "constant", 1.0)
        x[:, :, 0, 1:11] = 0.0

        return x

    def forward(self, x):
        x = self.pad_boundary(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.pad_boundary(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.pad_boundary(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = self.pad_boundary(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.dropout4(x)

        x = x.view(x.size(0), -1)  # flatten the output
        x = self.fc5(x)
        x = F.relu(x)
        x = self.dropout5(x)

        x = self.fc6(x)
        x = F.relu(x)
        x = self.dropout6(x)

        x = self.fc7(x)
        return x


class ReplayBuffer:
    # cyclical buffer
    def __init__(self, buffer_size):
        self.transitions = [None for _ in range(buffer_size)]
        self.index = 0
        self.max_size = buffer_size
        self.is_filled = False

    def addTransition(self, state_action, reward, next_state_action_tensor, is_terminal):
        self.transitions[self.index] = (state_action, reward, next_state_action_tensor, is_terminal)

        self.index += 1
        if self.index == self.max_size:
            self.is_filled = True
        self.index %= self.max_size

    def sampleMiniBatch(self, batch_size):
        num_entries = self.max_size if self.is_filled else self.index
        batch_size = np.min((num_entries, batch_size))

        # select random entries
        # TODO use random generator with seeding...
        chosen_entries = np.random.choice(num_entries, size=batch_size, replace=False)
        batch = [self.transitions[i] for i in chosen_entries]
        state_actions, rewards, next_state_actions_tensor_list, terminals = zip(*batch)

        # tensor shape: (batch, 1, 20, 10)
        state_actions_batch = torch.from_numpy(np.array(state_actions)).unsqueeze(1)

        return state_actions_batch, rewards, next_state_actions_tensor_list, terminals


class DQNAgent(BaseAgent):
    def __init__(self, num_episodes=20_000, data_file="./data/dqn.csv", model_file="./trained_models/dqn.model",
                 eps_start=1.0, eps_end=0.01, eps_scale=250_000, 
                 alpha_base=0.01, alpha_factor_start=1.0, alpha_factor_end=0.1, 
                 alpha_factor_num_episodes=10_000,
                 gamma=0.999, batch_size=64, buffer_capacity=500_000, target_update=2_500,
                 random_seed=69420, show_gui=True):
        super().__init__(num_episodes, data_file, show_gui)
        self.model_file = model_file

        # hyperparameters
        self.EPSILON_START = eps_start # epsilon-greedy
        self.EPSILON = eps_start
        self.EPSILON_END = eps_end
        self.EPSILON_SCALE = eps_scale
        self.GAMMA = gamma # discount factor
        self.BATCH_SIZE = batch_size # minibatch size from replaybuffer
        self.TARGET_UPDATE = target_update # number of updates before updating target network

        # state variable
        self.total_actions_survived = 0 # used for updating EPSILON

        # device setup
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("DQNAgent | Using GPU device:", torch.cuda.get_device_name(0))
            torch.cuda.manual_seed(random_seed)
        else:
            self.device = torch.device("cpu")
            print("DQNAgent | GPU not available, using CPU instead.")
            torch.manual_seed(random_seed)

        # q-value network and buffer
        self.q_model = DQNNetwork().to(self.device); self.q_model.train()
        self.q_target = deepcopy(self.q_model).to(self.device); self.q_target.eval()
        self.buffer = ReplayBuffer(buffer_capacity)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_model.parameters(), lr=alpha_base)
        self.alpha_scheduler = optim.lr_scheduler.LinearLR(self.optimizer, alpha_factor_start,
                                                        alpha_factor_end, alpha_factor_num_episodes)

        total_params = sum(p.numel() for p in self.q_model.parameters() if p.requires_grad)
        print(f"DQNAgent | # of trainable parameters: {total_params}")


    def run_episode(self):
        board, piece, _ = self.env.reset()
        
        if self.gui: self.gui.runOnce()

        done = False
        action_count = 0
        lines_cleared = 0
        while not done:
            self.total_actions_survived += 1
            action_count += 1

            # pick action (eps-greedy)
            #     state: grid only 1s/0s
            #     state-action: grid with 1s/0s and 0.5s at the bottom in landing positions
            state_action = None
            action_index = -1
            landing_positions = TetrisEnv.onePieceSearch(piece, board)
            num_group_actions = len(landing_positions)
            if np.random.random() < self.EPSILON:
                action_index = int(np.random.choice(num_group_actions))
                tetrimino, _ = landing_positions[action_index]
                state_action = TetrisBoard.placePieceInBoard(board.getBoard(), tetrimino)
            else:
                state_action_array = np.zeros((num_group_actions, 20, 10), dtype=np.float32)
                for landing_index in range(num_group_actions):
                    tetrimino, _ = landing_positions[landing_index]
                    state_action_array[landing_index, :, :] = TetrisBoard.placePieceInBoard(board.getBoard(), tetrimino)

                state_actions_tensor = torch.from_numpy(state_action_array) \
                                            .unsqueeze(1).to(self.device)

                self.q_model.eval()
                with torch.no_grad():
                    predictions = self.q_model(state_actions_tensor)
                self.q_model.train()

                action_index = torch.argmax(predictions).item()
                state_action = state_action_array[action_index, :, :]

            # update EPSILON
            self.EPSILON = np.max((self.EPSILON_START - self.total_actions_survived / self.EPSILON_SCALE, 
                                   self.EPSILON_END))
            
            # progress environment with chosen action
            (next_board, next_piece, _), reward, done, info = self.env.group_step(landing_positions[action_index])
            lines_cleared += info["cleared_lines"]
            if self.gui: self.gui.draw()

            # store the next state-actions
            next_landing_positions = TetrisEnv.onePieceSearch(next_piece, next_board)
            num_next_landing_positions = len(next_landing_positions)
            next_state_actions_array = np.zeros((num_next_landing_positions, 20, 10), dtype=np.float32)
            for next_land_index in range(num_next_landing_positions):
                tetrimino, _ = next_landing_positions[next_land_index]
                next_state_actions_array[next_land_index, :, :] = TetrisBoard.placePieceInBoard(next_board.getBoard(), tetrimino)

            next_state_actions_tensor = torch.tensor(next_state_actions_array, dtype=torch.float32) \
                                                    .unsqueeze(1) \
                                                    .to(self.device)
            
            # store transition in replay buffer
            self.buffer.addTransition(state_action, reward, next_state_actions_tensor, done)

            # sample random minibatch from replay buffer
            state_actions_batch, \
                rewards, \
                next_state_actions_tensor_list, \
                terminals = self.buffer.sampleMiniBatch(self.BATCH_SIZE)
            state_actions_batch = state_actions_batch.to(self.device)

            # calculate target update and q
            q_values = self.q_model(state_actions_batch)
            q_values = q_values.squeeze()

            max_q_targets = [] # array of batch_size length
            with torch.no_grad():
                for next_state_actions_tensor in next_state_actions_tensor_list:
                    predictions = self.q_target(next_state_actions_tensor) 
                    max_q_targets.append(torch.max(predictions).item())

            td_increments = []
            for reward, max_q, terminal in zip(rewards, max_q_targets, terminals):
                if terminal:
                    td_increments.append(reward)
                else:
                    td_increments.append(reward + self.GAMMA * max_q)
            target_val = torch.tensor(td_increments, dtype=torch.float32) \
                                .squeeze().to(self.device)
            
            # stochastic gradient update
            self.optimizer.zero_grad()
            loss = self.criterion(q_values, target_val)
            loss.backward()
            self.optimizer.step()

            # update state variable for buffer logic
            piece, board = next_piece, next_board

            # periodically update target 
            if self.total_actions_survived % self.TARGET_UPDATE == 0:
                print("DQNAgent |     Updating Target Model,", target_val)
                q_target = deepcopy(self.q_model).to(self.device)
                torch.save(q_target, self.model_file)
                q_target.eval()

        self.alpha_scheduler.step()
        return lines_cleared, action_count


    def save_network(self, file=None):
        if file is None:
            torch.save(self.q_model, self.model_file)
        else:
            torch.save(self.q_model, file)
    

if __name__ == "__main__":
    agent = DQNAgent()
    agent.train()
    agent.save_network()
