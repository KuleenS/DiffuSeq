import os

import json

import math

from typing import List

import cv2

import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

import torch
import json
import datasets
from datasets import Dataset as Dataset2

def load_data_text(
    batch_size, 
    seq_len, 
    folder,
    deterministic=False, 
    data_args=None, 
    model_emb=None,
    split='train', 
    loaded_vocab=None,
    loop=True,
):
    """
    For a dataset, create a generator over (seqs, kwargs) pairs.

    Each seq is an (bsz, len, h) float tensor, and the kwargs dict contains zero or
    more keys, each of which map to a batched Tensor of their own.
    The kwargs dict can be used for some meta information.

    :param batch_size: the batch size of each returned pair.
    :param seq_len: the max sequence length (one-side).
    :param deterministic: if True, yield results in a deterministic order.
    :param data_args: including dataset directory, num of dataset, basic settings, etc.
    :param model_emb: loaded word embeddings.
    :param loaded_vocab: loaded word vocabs.
    :param loop: loop to get batch data or not.
    """

    print('#'*30, '\nLoading text data...')

    training_data = get_corpus(data_args, seq_len, folder=os.path.join(folder, split), loaded_vocab=loaded_vocab)

    dataset = TextDataset(
        training_data,
        data_args,
        model_emb=model_emb
    )

    if split != 'test':
        sampler = DistributedSampler(dataset)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            # drop_last=True,
            sampler=sampler,
            # shuffle=not deterministic,
            # num_workers=4,
        )
    else:
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,  # 20,
            # drop_last=True,
            # sampler=sampler,
            shuffle=not deterministic,
            # num_workers=4,
        )

    if loop:
        return infinite_loader(data_loader)
    else:
        # print(data_loader)
        return iter(data_loader)

def infinite_loader(data_loader):
    while True:
        yield from data_loader

def helper_tokenize(sentence_lst, vocab_dict, seq_len):
    # Process.memory_info is expressed in bytes, so convert to megabytes
    raw_datasets = Dataset2.from_dict(sentence_lst)

    def tokenize_function(examples):
        input_id_x = vocab_dict.encode_token(examples['src'])
        input_id_y = vocab_dict.encode_token(examples['trg'])
        result_dict = {'input_id_x': input_id_x, 'input_id_y': input_id_y, 'video_path': examples['video_path'], 'start': examples['start']}

        return result_dict

    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        remove_columns=['src', 'trg'],
        load_from_cache_file=True,
        desc="Running tokenizer on dataset",
    )

    def merge_and_mask(group_lst):
        lst = []
        mask = []
        for i in range(len(group_lst['input_id_x'])):
            end_token = group_lst['input_id_x'][i][-1]
            src = group_lst['input_id_x'][i][:-1]
            trg = group_lst['input_id_y'][i][:-1]
            while len(src) + len(trg) > seq_len - 3:
                if len(src)>len(trg):
                    src.pop()
                elif len(src)<len(trg):
                    trg.pop()
                else:
                    src.pop()
                    trg.pop()
            src.append(end_token)
            trg.append(end_token)

            lst.append(src + [vocab_dict.sep_token_id] + trg)
            mask.append([0]*(len(src)+1))
        group_lst['input_ids'] = lst
        group_lst['input_mask'] = mask
        return group_lst
    
    tokenized_datasets = tokenized_datasets.map(
        merge_and_mask,
        batched=True,
        desc=f"merge and mask",
    )
    
    def pad_function(group_lst):
        max_length = seq_len
        group_lst['input_ids'] = _collate_batch_helper(group_lst['input_ids'], vocab_dict.pad_token_id, max_length)
        group_lst['input_mask'] = _collate_batch_helper(group_lst['input_mask'], 1, max_length)
        return group_lst

    lm_datasets = tokenized_datasets.map(
        pad_function,
        batched=True,
        desc=f"padding",
    )

    raw_datasets = datasets.DatasetDict()
    raw_datasets['train'] = lm_datasets
    return raw_datasets

def read_folder(folder: str, ext: str) -> List[str]:
    total_files = [] 

    for root, dirs, files in os.walk(folder):
        for name in files:
            if ext in name:
                total_files.append(os.path.join(root, name))
    
    return total_files

def get_corpus(data_args, seq_len, folder, loaded_vocab=None):

    print('#'*30, '\nLoading dataset {} from {}...'.format(data_args.dataset, data_args.data_dir))

    sentence_lst = {'src':[], 'video_path': [], 'start': [], 'trg': []}

    video_files = sorted(read_folder(folder, ".mp4"))

    caption_files = [x.split(".")[0]+"_cleaned.json" for x in video_files]

    for video_file, caption_file in zip(video_files, caption_files):
        with open(caption_file, "r") as f:
            data = json.load(f)
        for i in range(1, len(data)):
            sentence_lst["src"].append(" ".join([x["text"] for x in data[:i]]))
            sentence_lst["trg"].append(data[i]["text"])
            sentence_lst["start"].append(data[i]["start"])
            sentence_lst["video_path"].append(video_file)

    vocab_dict = loaded_vocab
    
    train_dataset = helper_tokenize(sentence_lst, vocab_dict, seq_len)
    return train_dataset


class TextDataset(Dataset):
    def __init__(self, text_datasets, data_args, model_emb=None):
        super().__init__()
        self.text_datasets = text_datasets
        self.length = len(self.text_datasets['train'])
        self.data_args = data_args
        self.model_emb = model_emb

        self.video_container = None
        self.video_container_path = None

    def __len__(self):
        return self.length

    def sample_frame_indices(self, frame_length_of_clip, FPS):
        initial_rate = FPS // 5

        ret = [i*initial_rate for i in range(50)]

        new_tot = frame_length_of_clip - math.ceil(10 * FPS)
        
        first_third = np.ceil(np.linspace(0, new_tot // 3, num=90) + math.ceil(10 * FPS))

        second_third = np.ceil(np.linspace(new_tot//3, new_tot, num=90) + math.ceil(10 * FPS))

        return ret + list(first_third) + list(second_third)

    def __getitem__(self, idx):
        with torch.no_grad():

            input_ids = self.text_datasets['train'][idx]['input_ids']
            hidden_state = self.model_emb(torch.tensor(input_ids))

            # obtain the input vectors, only used when word embedding is fixed (not trained end-to-end)
            arr = np.array(hidden_state, dtype=np.float32)

            out_kwargs = {}
            out_kwargs['input_ids'] = np.array(self.text_datasets['train'][idx]['input_ids'])
            out_kwargs['input_mask'] = np.array(self.text_datasets['train'][idx]['input_mask'])
        
        if self.text_datasets['train'][idx]["video_path"] != self.video_container_path:
        
            self.video_container = cv2.VideoCapture(self.text_datasets['train'][idx]["video_path"])
            self.video_container_path = self.text_datasets['train'][idx]["video_path"]

        FPS = self.video_container.get(cv2.CAP_PROP_FPS)

        sentence_start_frame = math.ceil(self.text_datasets['train'][idx]["start"] * FPS)

        if sentence_start_frame < 230:
            num_copies = 230//sentence_start_frame

            left_over = 230 % sentence_start_frame

            indicies = [0]*(num_copies + left_over)

            for i in range(1, sentence_start_frame):
                indicies += [i]*num_copies

        elif sentence_start_frame < FPS * 60:
            indicies = list(np.linspace(0, sentence_start_frame, num=230))
        else:
            indicies = self.sample_frame_indices(sentence_start_frame, FPS)

        indicies = [sentence_start_frame-x for x in indicies][::-1]

        frames = []

        for index in indicies:
            self.video_container.set(cv2.CAP_PROP_POS_FRAMES, index)

            _, frame = self.video_container.read()

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        out_kwargs["video"] = frames

        return arr, out_kwargs

def _collate_batch_helper(examples, pad_token_id, max_length, return_mask=False):
    result = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    mask_ = torch.full([len(examples), max_length], pad_token_id, dtype=torch.int64).tolist()
    for i, example in enumerate(examples):
        curr_len = min(len(example), max_length)
        result[i][:curr_len] = example[:curr_len]
        mask_[i][:curr_len] = [1] * curr_len
    if return_mask:
        return result, mask_
    return result