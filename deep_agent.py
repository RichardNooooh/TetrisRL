import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from agent import BaseAgent
from tetris import TetrisEnv, ACTION
from tetrisgui import TetrisGUI
from copy import deepcopy

class DQNNetwork(nn.Module):
    def __init__(self):
        super(DQNNetwork, self).__init__()

        # 20 x 10 "image" tensor input of 1s/0s
        # the current falling piece will be represented with 0.5s on the grid
        # Mnih et al (2015) suggests mapping state to multiple actions to reduce need for multiple forward passes...
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.dropout1 = nn.Dropout2d(p=0.25)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout2d(p=0.25)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.dropout3 = nn.Dropout2d(p=0.25)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dropout4 = nn.Dropout2d(p=0.25)
        
        self.fc5 = nn.Linear(64 * 200, 128)
        self.dropout5 = nn.Dropout(p=0.25)

        self.fc6 = nn.Linear(128, 128)
        self.dropout6 = nn.Dropout(p=0.25)

        self.fc7 = nn.Linear(128, 5) # 5 actions
        
        # kaiming/he initialization
        self.init_weights()
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.dropout3(x)

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

    def addTransition(self, state, action, reward, next_state, is_terminal):
        self.transitions[self.index] = (state, action, reward, next_state, is_terminal)

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
        states, actions, rewards, next_states, terminals = zip(*batch)

        # tensor shape: (batch, 1, 20, 10)
        state_batch = torch.from_numpy(np.array(states)).unsqueeze(1)
        next_state_batch = torch.from_numpy(np.array(next_states)).unsqueeze(1)

        actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1)

        return state_batch, actions, rewards, next_state_batch, terminals

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using GPU:", torch.cuda.get_device_name(0))
        torch.cuda.manual_seed(69420)
    else:
        device = torch.device("cpu")
        print("GPU not available, using CPU instead.")
        torch.manual_seed(69420)

    EPSILON_START = 1.0 # epsilon-greedy
    EPSILON = 1.0
    EPSILON_END = 0.001
    ALPHA = 0.001 # learning rate
    GAMMA = 0.9999 # discount factor
    BATCH_SIZE = 32 # minibatch size from replaybuffer
    TARGET_UPDATE = 10_000 # number of updates before updating target network

    env = TetrisEnv()
    gui = TetrisGUI()
    gui.linkGameEnv(env)
    gui.runOnce()

    q_model = DQNNetwork().to(device)
    q_target = deepcopy(q_model).to(device); q_target.eval()
    buffer = ReplayBuffer(1_000_000)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(q_model.parameters(), lr=ALPHA)

    total_params = sum(p.numel() for p in q_model.parameters() if p.requires_grad)
    print(f"Total number of parameters: {total_params}")

    total_lines_cleared = 0
    for episode in range(1000):
        board, piece, _ = env.reset()
        state = board.getBoardWithPiece(piece)

        print("Episode:", episode)
        done = False
        total_frames = 0
        i = 0
        while not done:
            total_frames += 1
            i += 1

            # pick action (eps-greedy)
            # next_state_actions = env.getNextStates()
            action = -1
            if np.random.random() < EPSILON:
                action = int(np.random.choice(len(ACTION)))
            else:
                # Given state, we calculate Q(s, actions) with network...
                state_tensor = torch.from_numpy(state).unsqueeze(0).unsqueeze(0).to(device)

                q_model.eval()
                with torch.no_grad():
                    predictions = q_model(state_tensor)
                q_model.train()
                action = torch.argmax(predictions).item()
            
            # update EPSILON
            EPSILON = np.max((EPSILON_START - total_frames / 1_000_000.0, EPSILON_END))
            
            # progress environment with chosen action
            (next_board, next_piece, _), reward, done, info = env.step(action)
            next_state = next_board.getBoardWithPiece(next_piece)
            total_lines_cleared += info["cleared_lines"]
            gui.draw()

            # store transition in replay buffer
            buffer.addTransition(state, action, reward, next_state, done)

            # sample random minibatch from replay buffer
            state_batch, actions, rewards, next_state_batch, terminals = buffer.sampleMiniBatch(BATCH_SIZE)
            state_batch, next_state_batch = state_batch.to(device), next_state_batch.to(device)
            actions = actions.to(device)

            # calculate target update and q
            q_values = q_model(state_batch)
            q_values = q_values.gather(1, actions)
            q_values = q_values.squeeze()
            
            # q_target.eval()
            with torch.no_grad(): # we only ever evaluate on q_target
                next_prediction_batch = q_target(next_state_batch) 
            # q_target.train()

            td_increments = []
            for reward, prediction, terminal in zip(rewards, next_prediction_batch, terminals):
                if terminal:
                    td_increments.append(reward)
                else:
                    td_increments.append(reward + GAMMA * torch.max(prediction).item())
            target_val = torch.tensor(td_increments).squeeze().to(device)

            # stochastic gradient update
            optimizer.zero_grad()
            loss = criterion(q_values, target_val)
            loss.backward()
            optimizer.step()

            # update state variable for buffer logic
            state = next_state
            
            # periodically update target 
            if i % TARGET_UPDATE == 0:
                q_target = deepcopy(q_model).to(device)
        print("    Length:", i, "Total Length:", total_frames, "\tTotal Lines Cleared:", total_lines_cleared)
