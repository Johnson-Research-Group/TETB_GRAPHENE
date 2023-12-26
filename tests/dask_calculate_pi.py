#!/usr/bin/python

import time
import dask
from dask.distributed import Client
import os
import numpy as np
#import cupy as cp

def simulate(seed, count):
    np.random.seed(seed)
    xy = np.random.uniform(size=(count, 2))
    hits = ((xy * xy).sum(1) < 1.0).sum()
    return hits, count

def reduce(results):
    total_hits = 0
    total_count = 0
    for hits, count in results: 
        total_hits += hits
        total_count += count
    return 4.0 * total_hits / total_count

scheduler_file = os.path.join(os.environ["SCRATCH"], "scheduler_file.json")

#note, we're not going to connect to the dashboard since this job is not running interactively
print("in python script")
client = Client(scheduler_file=scheduler_file)
print(client)

total = 5000000
tasks = 10000
count = total // tasks
futures = client.map(simulate, list(9876543 + np.arange(tasks, dtype=int)), count=count)
#this will return a list of futures
futures[0]

print("value of pi is")
start = time.time()
print(client.submit(reduce, futures).result())
end = time.time()

print("calculated pi in {} seconds".format(end - start))


