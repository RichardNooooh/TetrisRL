from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import MOVEMENT
import matplotlib.pyplot as plt
import numpy as np

env = gym_tetris.make('TetrisA-v3')
env = JoypadSpace(env, MOVEMENT)

# def read_row(start_address):
#     return gym_tetris.TetrisEnv._read_bcd(start_address, 10)

print(env.unwrapped)

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    
    # data = env.unwrapped._board  
    # curr_x = env.unwrapped.ram[0x40]
    # curr_y = env.unwrapped.ram[0x41]

    play_field = state[48:208, 95:175, :] # height of 160 and width of 80
    # tetris is 20 high and 10 wide => grid of 8x8 squares
    grid = []
    for h in range(20):
        temp = []
        for w in range(10):
            height_offset = h * 8
            width_offset = w * 8
            avg = np.sum(play_field[height_offset:height_offset + 8, width_offset: width_offset + 8, :])
            temp.append(1 if avg else 0)
        grid.append(temp)
    if step >= 3000:
        plt.imshow(play_field)
        plt.show()
    env.render()

env.close()