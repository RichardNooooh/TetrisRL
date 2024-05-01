from agent import SarsaLambdaAgent
import multiprocessing
from tqdm import tqdm

if __name__ == '__main__':  
    num_episodes = 2
    num_runs = 3
    agent = SarsaLambdaAgent(num_episodes, num_runs, "sarsa_testdata.csv", 0.99, 0.8, 0.1, show_gui=False)

    file_name = "test.csv"

    threads = []
    for run_id in range(num_runs):
        num_processes = 4

        thread = multiprocessing.Process(target=agent.train, args=(run_id,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    for i in tqdm(range(3)):
        agent.train(i)

    agent.record_data()
