import dask
import os
import numpy as np
from time import sleep
from dask.distributed import LocalCluster, Client
@dask.delayed
def inc(x):
    sleep(1)
    print(x)
    return x + 1

def simulate(seed, count):
    np.random.seed(seed)
    xy = np.random.uniform(size=(count, 2))
    hits = ((xy * xy).sum(1) < 1.0).sum()
    return hits, count

def reduce(results):
    total_hits = 0
    total_count = 0
    for hits, count in results:
        total_hits += hits #.get() #need to pull our data off the GPU
        total_count += count
    return 4.0 * total_hits / total_count

if __name__=="__main__":
    #cluster = LocalCluster()
    #client = Client(cluster)
    scheduler_file = os.path.join(os.environ["SCRATCH"], "scheduler_file.json")

    #note, we're not going to connect to the dashboard since this job is not running interactively

    client = Client(scheduler_file=scheduler_file)
    total = 5000000
    tasks = 8
    count = total // tasks

    #futures = client.map(simulate, list(9876543 + np.arange(tasks, dtype=int)),count=count)
    futures = client.map(inc,np.arange(20))
    total = client.submit(sum, futures).result()
    print("After computing :", total.compute())

    """results = []
    data = np.arange(20)
    for x in data:
        y = inc(x)
        results.append(y)

    total = sum(results)
    print("Before computing:", total)  # Let's see what type of thing total is
    result = total.compute()
    print("After computing :", result)  # After it's computed"""
