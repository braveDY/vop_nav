from bisect import bisect_right
from collections import OrderedDict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class VOPDataset(Dataset):
    def __init__(self, directory, split='train', verify=False, check_stop=None):
        self.directory = Path(directory).resolve()
        payload = (self.directory / 'manifest.json').read_bytes()
        self.fingerprint = hashlib.sha256(payload).hexdigest()
        self.manifest = json.loads(payload)
        if self.manifest['schema_version'] != 1 or split not in ('train', 'val'):
            raise ValueError('unsupported prepared schema or split')
        if self.manifest.get('label_config', {}).get('geometry') != 'rounded_box':
            raise ValueError('Legacy VO geometry: regenerate prepared labels from raw with rounded_box')
        self.shards = [s for s in self.manifest['shards'] if s['split'] == split]
        self.ends, total = [], 0
        for shard in self.shards:
            root = (self.directory / shard['path']).resolve()
            if root.parent != self.directory or shard['samples'] <= 0:
                raise ValueError(f'invalid shard: {root}')
            for name, shape in (('history.npy', (shard['samples'], 5, 40)),
                                ('targets.npy', (shard['samples'], 360, 2)),
                                ('sample_ids.npy', (shard['samples'], 3))):
                if check_stop:
                    check_stop()
                path = root / name
                if verify and hashlib.sha256(path.read_bytes()).hexdigest() != shard['sha256'][name]:
                    raise ValueError(f'checksum mismatch: {path}')
                array = np.load(path, mmap_mode='r', allow_pickle=False)
                if array.shape != shape or (name != 'sample_ids.npy' and array.dtype != np.float32):
                    raise ValueError(f'invalid array: {path}')
            total += shard['samples']
            self.ends.append(total)
        if total != self.manifest['samples'][split] or total == 0:
            raise ValueError(f'invalid or empty split: {split}')
        self.cache = OrderedDict()

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        shard = bisect_right(self.ends, index)
        offset = index - (self.ends[shard - 1] if shard else 0)
        if shard not in self.cache:
            root = self.directory / self.shards[shard]['path']
            self.cache[shard] = tuple(np.load(root / name, mmap_mode='r', allow_pickle=False)
                                      for name in ('history.npy', 'targets.npy'))
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        self.cache.move_to_end(shard)
        history, target = self.cache[shard]
        return torch.from_numpy(history[offset].copy()), torch.from_numpy(target[offset].copy())

    def __getstate__(self):
        state = self.__dict__.copy()
        state['cache'] = OrderedDict()
        return state


class ShardBatchSampler(Sampler):
    """Shuffle shards and rows, while each minibatch reads a single shard."""
    def __init__(self, dataset, batch_size, shuffle=True, seed=123):
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        self.dataset, self.batch_size = dataset, batch_size
        self.shuffle, self.seed, self.epoch = shuffle, seed, 0

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.dataset.shards), generator=generator).tolist() if self.shuffle else range(len(self.dataset.shards))
        for shard in order:
            n = self.dataset.shards[shard]['samples']
            start = self.dataset.ends[shard - 1] if shard else 0
            rows = torch.randperm(n, generator=generator) if self.shuffle else torch.arange(n)
            for offset in range(0, n, self.batch_size):
                yield (rows[offset:offset + self.batch_size] + start).tolist()

    def __len__(self):
        return sum(math.ceil(s['samples'] / self.batch_size) for s in self.dataset.shards)
