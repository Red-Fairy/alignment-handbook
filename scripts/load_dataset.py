import json
import os
from datasets import Dataset, DatasetDict, IterableDataset
from torch.utils.data import DataLoader

'''
data are stored in json files, each json file contains a "data" field which is a list of json objects
with following fields: 'text', 'input_visual', 'output_visual', 'input_action', 'output_action'
'''

def gen(shards):
    for shard in shards:
        with open(shard, "r") as f:
            data = json.load(f)["data"]
            for d in data:
                yield d

root = 'example_files'
shards = [f"{root}/data_{i}.json" for i in range(1)]
ds = IterableDataset.from_generator(gen, gen_kwargs={"shards": shards})

dataloader = DataLoader(ds.with_format("torch"), num_workers=4)  # give each worker a subset of 32/4=8 shards

# load the dataset
for data in dataloader:
    print(data)

