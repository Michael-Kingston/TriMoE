# src/dataset.py
# This file contains the logic for loading, processing, and batching the data.
# It includes the main PyTorch Dataset class and the collate functions.

import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


def load_data(file_path, mode, debug=False):
    """Loads the pre-processed data from a pickle file."""
    dataPath = os.path.join(file_path, f"{mode}p2x_data_with_dem.pkl")
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[ '{mode}' from: {dataPath}")

    if not os.path.isfile(dataPath):
        if int(os.environ.get("RANK", "0")) == 0:
            print(f"[data file not found at {dataPath}")
        return None

    with open(dataPath, 'rb') as f:
        data = pickle.load(f)

    if int(os.environ.get("RANK", "0")) == 0:
        print(f"dloaded {len(data)} samples for mode '{mode}'.")

    return data[:200] if debug else data

class TSNote_Irg(Dataset):
    """
    PyTorch Dataset for loading trimodal EHR data 
    Handles tokenisation and tensor conversion.
    
    """
    def __init__(self, args, mode, tokenizer):
        self.tokenizer = tokenizer
        self.max_len = args.max_length
        self.num_of_notes = args.num_of_notes
        self.tt_max = args.tt_max
        self.dem_dim = None
        self.ts_dim = None
        self.data = load_data(file_path=args.file_path, mode=mode, debug=args.debug)
        if self.data:
            self._find_dims()

    def _find_dims(self):
        """Finds the input dimensions for demographics and time-series data."""
        for item in self.data:
            dem_ok = 'dem' in item and len(item.get('dem', [])) > 0
            ts_ok = 'irg_ts' in item and len(item.get('irg_ts', [])) > 0
            if dem_ok and ts_ok:
                self.dem_dim = len(item['dem'])
                self.ts_dim = len(item['irg_ts'][0])
                
                if int(os.environ.get("RANK", "0")) == 0:
                    print(f"found dims: Demographics={self.dem_dim}, Time-Series={self.ts_dim}")
                return
        if int(os.environ.get("RANK", "0")) == 0:
            print("could not find a valid sample to determine input dimensions.")

    def __getitem__(self, idx):
        if self.dem_dim is None or self.ts_dim is None:
            return None
        
        item = self.data[idx]
        
        # dem
        dem_tensor = torch.tensor(item.get('dem', []), dtype=torch.float)
        if dem_tensor.numel() == 0:
            dem_tensor = torch.zeros(self.dem_dim, dtype=torch.float)

        # t-s
        if 'irg_ts' in item and len(item.get('irg_ts', [])) > 0:
            ts_tensor = torch.tensor(item['irg_ts'], dtype=torch.float)
            ts_mask_tensor = torch.tensor(item['irg_ts_mask'], dtype=torch.long)
            ts_tt_tensor = torch.tensor([t / self.tt_max for t in item["ts_tt"]], dtype=torch.float)
        else:
            ts_tensor = torch.zeros(1, self.ts_dim, dtype=torch.float)
            ts_mask_tensor = torch.zeros(1, self.ts_dim, dtype=torch.long) # mask should have same shape as value
            ts_tt_tensor = torch.zeros(1, dtype=torch.float)
        
        # text
        text = item.get('text_data', [])
        tokens, masks = [], []
        if not text:
            for _ in range(self.num_of_notes):
                tokens.append(torch.zeros(self.max_len, dtype=torch.long))
                masks.append(torch.zeros(self.max_len, dtype=torch.long))
        else:
            for t in text:
                inputs = self.tokenizer.encode_plus(
                    t, padding="max_length", max_length=self.max_len,
                    add_special_tokens=True, return_attention_mask=True, truncation=True
                )
                tokens.append(torch.tensor(inputs['input_ids'], dtype=torch.long))
                masks.append(torch.tensor(inputs['attention_mask'], dtype=torch.long))
            
            # pad or truncate the list of notes
            if len(tokens) > self.num_of_notes:
                tokens = tokens[-self.num_of_notes:]
                masks = masks[-self.num_of_notes:]
            while len(tokens) < self.num_of_notes:
                tokens.insert(0, torch.zeros(self.max_len, dtype=torch.long))
                masks.insert(0, torch.zeros(self.max_len, dtype=torch.long))

        # create a mask for the notes themselves
        note_mask = torch.tensor([1] * len(text) + [0] * (self.num_of_notes - len(text)), dtype=torch.long)
        if len(note_mask) > self.num_of_notes:
             note_mask = note_mask[-self.num_of_notes:]

        return {
            'ts': ts_tensor, 'ts_mask': ts_mask_tensor, 'ts_tt': ts_tt_tensor,
            'dem': dem_tensor, 'label': torch.tensor(item["label"], dtype=torch.long),
            'input_ids': torch.stack(tokens), 'attention_mask': torch.stack(masks),
            'note_time_mask': note_mask
        }

    def __len__(self):
        return len(self.data) if hasattr(self, 'data') and self.data is not None else 0


def robust_collate_fn(batch):
    """
    Robustly collates a batch of samples, filtering out Nones and padding sequences.
    """
    batch = [b for b in batch if b is not None]
    if not batch:
        return None

    # pad the t-s tensors
    ts_i = pad_sequence([b['ts'] for b in batch], batch_first=True, padding_value=0.0)
    ts_m = pad_sequence([b['ts_mask'] for b in batch], batch_first=True, padding_value=0)
    ts_t = pad_sequence([b['ts_tt'] for b in batch], batch_first=True, padding_value=0.0)

    # stack other tensors
    dem = torch.stack([b['dem'] for b in batch])
    lab = torch.stack([b["label"] for b in batch])
    i_ids = torch.stack([b['input_ids'] for b in batch])
    a_m = torch.stack([b['attention_mask'] for b in batch])
    n_t_m = torch.stack([b['note_time_mask'] for b in batch])

    return (ts_i, ts_m, ts_t, dem, i_ids, a_m, n_t_m, lab.float().unsqueeze(1))

def train_collate_wrapper(batch):
    """Wrapper for training."""
    processed_batch = robust_collate_fn(batch)
    if processed_batch is None:
        return None, len(batch)
    return processed_batch, 0

def eval_collate_wrapper(batch):
    """Wrapper for evaluation."""
    return robust_collate_fn(batch)

def data_prepare(args, mode, tokenizer):
    """Creates a DataLoader for a given data split"""
    dataset = TSNote_Irg(args, mode, tokenizer)
    if not hasattr(dataset, 'data') or dataset.data is None:
        return None
    
    collate_fn = train_collate_wrapper if mode == 'train' else eval_collate_wrapper
    
    return DataLoader(
        dataset, 
        shuffle=(mode == 'train'), 
        batch_size=args.batch_size, 
        collate_fn=collate_fn, 
        num_workers=4, 
        pin_memory=True)