import json
import os
from datasets import Dataset, DatasetDict, IterableDataset
from torch.utils.data import DataLoader
import random
import numpy as np

'''
data are stored in json files, each json file contains a "data" field which is a list of json objects
with following fields: 'text', 'Visual', 'Action', "ID", "Frame_number"
output: a list of dictionaries, each dictionary contains the following fields:
    'input_visual', 'output_visual', 'input_action', 'output_action', 'text'
    'input_visual', 'output_visual', 'input_action', 'output_action' are torch tensors
'''

def gen(shards, num_input, num_output, vocab_size, num_visual_tokens, num_action_tokens):
    for shard in shards:
        with open(shard, "r") as f:
            for line in f:
                instance_info = json.loads(line)
                num_frames = instance_info["Frame_number"]
                ret = {}
                # randomly select num_input+num_output consecutive frames
                if num_frames < num_input + num_output:
                    continue
                start_frame = random.randint(0, num_frames - num_input - num_output)
                ret['input_visual'] = np.array(instance_info['Visual'][start_frame:start_frame+num_input], dtype=np.int32) + vocab_size
                ret['output_visual'] = np.array(instance_info['Visual'][start_frame+num_input:start_frame+num_input+num_output], dtype=np.int32) + vocab_size
                ret['input_action'] = np.array(instance_info['Action'][start_frame:start_frame+num_input], dtype=np.int32) + num_visual_tokens + vocab_size
                ret['output_action'] = np.array(instance_info['Action'][start_frame+num_input:start_frame+num_input+num_output], dtype=np.int32) + num_visual_tokens + vocab_size
                ret['text'] = instance_info['Text']
                yield ret

def get_VLA_dataset(args, split='train'):
    root = args.data_root
    file_format = 'data_bridge2_processed_{}.jsonl'
    shards = [os.path.join(root, split, file_format.format(i)) for i in range(len(os.listdir(root)))]
    ds = IterableDataset.from_generator(gen, gen_kwargs={"shards": shards, "vocab_size": args.vocab_size, 
                                                         "num_visual_tokens": args.num_visual_tokens, "num_action_tokens": args.num_action_tokens,
                                                        "num_input": args.num_input_frames, "num_output": args.num_output_frames})
    return ds

def get_VLA_dataloader(root, num_input=6, num_output=1):
    file_format = 'data_bridge2_processed_{}.jsonl'
    shards = [os.path.join(root, file_format.format(i)) for i in range(len(os.listdir(root)))]
    ds = IterableDataset.from_generator(gen, gen_kwargs={"shards": shards, "num_input": num_input, "num_output": num_output})
    dataloader = DataLoader(ds.with_format("torch"), num_workers=4)
    return dataloader

if __name__ == "__main__":
    root = '/mnt/azureml/cr/j/994c153d01ba4915bd5a5b70faa5fc4d/exe/wd/VQ-Diffusion/bridge2_tokenized_sample'
    dataloader = get_VLA_dataloader(root)
    for batch in dataloader:
        for k, v in batch.items():
            print(k)
            if k in ['input_visual', 'output_visual', 'input_action', 'output_action']:
                print(v.shape)
            else:
                print(v)
