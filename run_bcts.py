from agent import HandwrittenBCTSAgent
import multiprocessing
from tqdm import tqdm

if __name__ == '__main__':  
    agent = HandwrittenBCTSAgent("./data/bcts_data.csv", show_gui=True)

    agent.train() # not actually training, just running.
    agent.record_data()
