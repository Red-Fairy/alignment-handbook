import json
import os
from datasets import Dataset, DatasetDict, IterableDataset
from torch.utils.data import DataLoader
import random
import numpy as np

def VLA_dataset_generator(shards, eos_token, num_input=6, num_output=6):
    '''
    each shard is a jsonl file, with each line containing a json object
    the json object contains the following fields:
    - frame_number: number of frames in the video, we guarantee frame_number % 6 == 0
    - vision_tokens: a 2D array of size (frame_number // 6 + 1, 256 * 3), 
        256 * 3 is because each clip has 6 frames and downsamples by factor 2
        index 0 is for 6 identical first frames
        index k is for the k-th clip of 6 frames (frame 6k-6 to frame 6k-1)
    - action_tokens: a 2D array of size (frame_number+1, 1),
        the last action token is for the null action (no action)
    - trajectory_language: a string that describes the trajectory
    - descriptions: 
        a dictionary with 0 and 6k-1 as keys, and strings as values
        0: describes the initial stage
        6k-1: describes the frame difference between frame 6k-6 to frame 6k-1
    - now we assume num_input == num_output == 6
    
    output:
    a generator that yields a dictionary with only the 'text' field
    each time we random sample timestamp t in 0, 1, ..., k-1 (k = frame_number // 6)
    if t == 0, we use the vision_tokens[0] and null_action * 5 as input
        and vision_tokens[1] and null_action + action_tokens[0..4] as output
    else, we use the vision_tokens[t] and action_tokens[6k-6, ..., 6k-2] as input
        and vision_tokens[t+1] and action_tokens[6k-1, ..., 6k+4] as output

    text = '<bot_i>' + data['trajectory_language'] + data['input_plan_description'] + '<eot_i>' + \
            '<bov_i>' + ''.join([f'<v{str(x)}>' for x in data['input_visual']]) + '<eov_i>' + \
            '<boa_i>' + ''.join([f'<a{str(x)}>' for x in data['input_action']]) + '<eoa_i>' + \
            '<bot_o>' + data['output_plan_description'] + '<eot_o>' + \
            '<bov_o>' + ''.join([f'<v{str(x)}>' for x in data['output_visual']]) + '<eov_o>' + \
            '<boa_o>' + ''.join([f'<a{str(x)}>' for x in data['output_action']]) + '<eoa_o>' + eos_token
    '''

    for shard in shards:
        with open(shard, "r") as f:
            for line in f:
                instance_info = json.loads(line)
                num_frames = instance_info["frame_number"]
                if num_frames < num_input + num_output:
                    continue
                # randomly sample a start frame from 0 ... num_frames // 6 - 1
                start_frame = random.randint(0, num_frames // 6 - 1)
                out = {}
                if start_frame == 0:
                    out['input_visual'] = np.array(instance_info['vision_tokens'][0], dtype=np.int32).flatten()
                    out['output_visual'] = np.array(instance_info['vision_tokens'][1], dtype=np.int32).flatten()
                    out['input_action'] = np.array([instance_info['action_tokens'][-1]] * 5, dtype=np.int32).flatten()
                    out['output_action'] = np.array([instance_info['action_tokens'][-1]] + instance_info['action_tokens'][0:5], dtype=np.int32).flatten()
                else:
                    out['input_visual'] = np.array(instance_info['vision_tokens'][start_frame], dtype=np.int32).flatten()
                    out['output_visual'] = np.array(instance_info['vision_tokens'][start_frame+1], dtype=np.int32).flatten()
                    out['input_action'] = np.array(instance_info['action_tokens'][6*start_frame-6:6*start_frame-2], dtype=np.int32).flatten()
                    out['output_action'] = np.array(instance_info['action_tokens'][6*start_frame-1:6*start_frame+4], dtype=np.int32).flatten()
                out['plan_description'] = instance_info['descriptions'][0] if start_frame == 0 else instance_info['descriptions'][6*start_frame-1]
                
                ret = {}
                ret['text'] = '<bot_i>' + instance_info['trajectory_language'] + out['plan_description'] + '<eot_i>' + \
                        '<bov_i>' + ''.join([f'<v{str(x)}>' for x in out['input_visual']]) + '<eov_i>' + \
                        '<boa_i>' + ''.join([f'<a{str(x)}>' for x in out['input_action']]) + '<eoa_i>' + \
                        '<bot_o>' + instance_info['descriptions'][start_frame+1] + '<eot_o>' + \
                        '<bov_o>' + ''.join([f'<v{str(x)}>' for x in out['output_visual']]) + '<eov_o>' + \
                        '<boa_o>' + ''.join([f'<a{str(x)}>' for x in out['output_action']]) + '<eoa_o>' + eos_token

def get_preprocessed_VLA_dataset(args, eos_token, split='train'):
    root = args.data_root
    file_format = 'data_bridge2_processed_{}.jsonl'
    shards = [os.path.join(root, split, file_format.format(i)) for i in range(len(os.listdir(os.path.join(root, split))))]
    ds = IterableDataset(VLA_dataset_generator, gen_kwargs={"shards": shards, "eos_token": eos_token})
    return ds