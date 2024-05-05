import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from agent import BaseAgent
from tetris import TetrisEnv, TetrisBoard, ACTION
from tetrisgui import TetrisGUI
from copy import deepcopy
from feature import Features
import pandas as pd

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

        self.fc7 = nn.Linear(128, 1) # estimating Q(s, a). expensive forward passes, but... yeah
        
        # kaiming/he initialization
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.pad(x, (1, 1, 1, 1), "constant", 1.0)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = F.pad(x, (1, 1, 1, 1), "constant", 1.0)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = F.pad(x, (1, 1, 1, 1), "constant", 1.0)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout3(x)

        x = F.pad(x, (1, 1, 1, 1), "constant", 1.0)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.dropout4(x)

        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc5(x)
        x = F.relu(x)
        x = self.dropout5(x)

        x = self.fc6(x)
        x = F.relu(x)
        x = self.dropout6(x)

        x = self.fc7(x)
        return x
    
class ReplayBuffer:
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
        chosen_entries = np.random.choice(num_entries, size=batch_size, replace=False)
        batch = [self.transitions[i] for i in chosen_entries]
        state_actions, rewards, next_state_actions_tensor_list, terminals = zip(*batch)

        # tensor shape: (batch, 1, 20, 10)
        state_actions_batch = torch.from_numpy(np.array(state_actions)).unsqueeze(1)

        return state_actions_batch, rewards, next_state_actions_tensor_list, terminals

# def get_scoring_function(board, piece):
#     features = Features(board, piece)
#     landing_height = -12.63 * features.landing_height(piece)
#     eroded_piece_cells = 6.60 * features.eroded_piece_cells(piece)
#     row_transitions = -9.92 * features.row_transitions()
#     column_transitions = -19.77 * features.column_transitions()
#     num_holes = -13.08 * features.num_holes()
#     cumulative_wells = -10.49 * features.cumulative_wells()
#     hole_depth = -1.61 * features.hole_depth()
#     rows_with_holes = -24.04 * features.rows_with_holes()
#     score = landing_height + eroded_piece_cells + row_transitions + column_transitions + num_holes + cumulative_wells + hole_depth + rows_with_holes
#     return 0.1 * score

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
        torch.cuda.manual_seed()
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU instead.")
        torch.manual_seed()

    EPSILON_START = 1.0 # epsilon-greedy
    EPSILON = 1.0
    EPSILON_END = 0.001
    ALPHA = 0.001 # learning rate
    GAMMA = 0.999 # discount factor
    BATCH_SIZE = 64 # minibatch size from replaybuffer
    TARGET_UPDATE = 2_000 # number of updates before updating target network

    env = TetrisEnv()
    # gui = TetrisGUI()
    # gui.linkGameEnv(env)

    q_model = DQNNetwork().to(device); q_model.train()
    q_target = deepcopy(q_model).to(device); q_target.eval()
    buffer = ReplayBuffer(1_000_000)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(q_model.parameters(), lr=ALPHA)

    total_params = sum(p.numel() for p in q_model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")
    cumulative_lines_per_epoch = [0]
    cumulative_frames_per_epoch = [0]
    total_lines_cleared = 0
    total_frames = 0
    for episode in range(10_000):
        board, piece, _ = env.reset()
        # state = board.getBoardWithPiece(piece)
        # gui.runOnce()

        print("Episode:", episode)
        done = False
        
        # prev_bcts_score = None
        frame_count = 0
        while not done:
            total_frames += 1
            frame_count += 1

            # pick action (eps-greedy)
            #     state: grid only 1s/0s
            #     state-action: grid with 1s/0s and 0.5s at the bottom in landing positions
            state_action = None
            action_index = -1
            landing_positions = TetrisEnv.onePieceSearch(piece, board)
            num_group_actions = len(landing_positions)
            if np.random.random() < EPSILON:
                action_index = int(np.random.choice(num_group_actions))
                tetrimino, _ = landing_positions[action_index]
                state_action = TetrisBoard.placePieceInBoard(board.getBoard(), tetrimino)
            else:
                state_action_array = np.zeros((num_group_actions, 20, 10), dtype=np.float32)
                for landing_index in range(num_group_actions):
                    tetrimino, _ = landing_positions[landing_index]
                    state_action_array[landing_index, :, :] = TetrisBoard.placePieceInBoard(board.getBoard(), tetrimino)

                state_actions_tensor = torch.from_numpy(state_action_array).unsqueeze(1).to(device)

                q_model.eval()
                with torch.no_grad():
                    predictions = q_model(state_actions_tensor)
                q_model.train()

                action_index = torch.argmax(predictions).item()
                state_action = state_action_array[action_index, :, :]

            # update EPSILON
            EPSILON = np.max((EPSILON_START - total_frames / 100_000.0, EPSILON_END))
            
            # progress environment with chosen action
            (next_board, next_piece, _), reward, done, info = env.group_step(landing_positions[action_index])
            # gui.draw()
            # bcts_score = get_scoring_function(next_board, next_piece)
            # if prev_bcts_score == None:
            #     score_diff = 0.0
            # else:
            #     score_diff = bcts_score - prev_bcts_score
            # prev_bcts_score = bcts_score

            total_lines_cleared += info["cleared_lines"]
            
            # need to store the next state-actions
            next_landing_positions = TetrisEnv.onePieceSearch(next_piece, next_board)
            num_next_group_actions = len(next_landing_positions)
            next_state_actions_array = np.zeros((num_next_group_actions, 20, 10), dtype=np.float32)
            for next_land_index in range(num_next_group_actions):
                tetrimino, _ = next_landing_positions[next_land_index]
                next_state_actions_array[next_land_index, :, :] = TetrisBoard.placePieceInBoard(next_board.getBoard(), tetrimino)

            next_state_actions_tensor = torch.tensor(next_state_actions_array, dtype=torch.float32).unsqueeze(1).to(device)

            # store transition in replay buffer
            buffer.addTransition(state_action, reward, next_state_actions_tensor, done)

            # sample random minibatch from replay buffer
            state_actions_batch, rewards, next_state_actions_tensor_list, terminals = buffer.sampleMiniBatch(BATCH_SIZE)
            state_actions_batch = state_actions_batch.to(device)

            # calculate target update and q
            q_values = q_model(state_actions_batch)
            q_values = q_values.squeeze()
            
            max_q_targets = [] # array of batch_size length
            with torch.no_grad():
                for next_state_actions_tensor in next_state_actions_tensor_list:
                    predictions = q_target(next_state_actions_tensor) 
                    max_q_targets.append(torch.max(predictions).item())

            td_increments = []
            for reward, max_q, terminal in zip(rewards, max_q_targets, terminals):
                if terminal:
                    td_increments.append(reward) # -10000 reward
                else:
                    td_increments.append(reward + GAMMA * max_q)
            target_val = torch.tensor(td_increments, dtype=torch.float32).squeeze().to(device)

            # stochastic gradient update
            optimizer.zero_grad()
            loss = criterion(q_values, target_val)
            # print("    Loss:", loss.item())
            loss.backward()
            optimizer.step()

            # update state variable for buffer logic
            piece, board = next_piece, next_board
            
            # periodically update target 
            if total_frames % TARGET_UPDATE == 0:
                print("    Updating Target Model,", target_val)
                q_target = deepcopy(q_model).to(device)
                torch.save(q_target, "./trained_models/dqn.model")
                q_target.eval()

        print("    Length:", frame_count, "Total Length:", total_frames, "\tTotal Lines Cleared:", total_lines_cleared)
        cumulative_lines_per_epoch.append(total_lines_cleared)
        cumulative_frames_per_epoch.append(total_frames)
    
    torch.save(q_target, "./trained_models/dqn.model")
    file_dict = dict()
    file_dict["Cumulative Frames"] = cumulative_frames_per_epoch
    file_dict["Cumulative Lines"] = cumulative_lines_per_epoch
    df = pd.DataFrame(file_dict)
    df.to_csv("./data/dqn_data.csv")
    