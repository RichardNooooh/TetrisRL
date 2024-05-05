from agent import LinearSarsaLambdaAgent
import multiprocessing
from tqdm import tqdm

# runs and also trains
if __name__ == '__main__':  
    num_episodes = 10_000
    agent = LinearSarsaLambdaAgent(num_episodes, "./data/sarsa_data.csv", 
                                   0.999, 0.8, 0.001, 0.001, show_gui=True)

    # agent.load_weights()
    agent.train()
    agent.print_weights()
    agent.record_data()
