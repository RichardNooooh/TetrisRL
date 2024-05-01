from agent import HandwrittenBCTSAgent
import multiprocessing
from tqdm import tqdm

agent = HandwrittenBCTSAgent(2, 3, "bcts_testdata.csv")
# num_episodes = 1
# num_runs = 32
# file_name = "test.csv"

# threads = []
# for run_id in range(num_runs):
#     num_processes = 4

#     thread = multiprocessing.Process(target=agent.train, args=(run_id,))
#     threads.append(thread)
#     thread.start()

# # Wait for all threads to complete
# for thread in threads:
#     thread.join()
# for i in tqdm(range(3)):
#     agent.train(i)

# agent.record_data()

agent.train(1)
