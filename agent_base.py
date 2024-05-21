import numpy as np
import pandas as pd
from tetris.env import TetrisEnv
from tetris.gui import TetrisGUI
from tetris.feature import Features

class BaseAgent:
    def __init__(self, num_episodes, file_name, show_gui):
        self.num_episodes = num_episodes
        # self.num_runs = num_runs # for averaging
        self.num_lines_cleared = np.zeros(num_episodes)
        self.actions_survived = np.zeros(num_episodes)
        self.file_name = file_name

        self.env = TetrisEnv()
        self.gui = None
        if show_gui:
            self.gui = TetrisGUI()
            self.gui.linkGameEnv(self.env)

    def train(self):
        for episode in range(self.num_episodes):
            print("Episode:", episode)
            cleared_lines, survived_actions = self.run_episode()
            print("    Lines Cleared:", cleared_lines, "\tSurvived Frames:", survived_actions)
            self.num_lines_cleared[episode] = cleared_lines
            self.actions_survived[episode] = survived_actions

    def run_episode(self):
        raise NotImplementedError
    
    def record_data(self):
        data_dict = dict()
        data_dict["Frames Survived"] = self.actions_survived
        data_dict["Num Lines Cleared"] = self.num_lines_cleared

        df = pd.DataFrame(data_dict)
        df.to_csv(self.file_name)

