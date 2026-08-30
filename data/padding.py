from torch.utils.data import Sampler, default_collate
import torch

class PaddingSampler(Sampler):
    def __init__(self, lengths):
        # HuggingFace Dataset column access returns a Column/list, not a Tensor.
        # torch.vstack requires tensors, so normalize first.
        lengths = torch.as_tensor(lengths, dtype=torch.long)
        indices = torch.arange(len(lengths))
        self.length_idx = torch.stack((indices, lengths), dim=1)

    def __iter__(self):
        # Shuffle the indices
        self.length_idx = self.length_idx[torch.randperm(len(self.length_idx))]
        # Resort, this will get a random order within each length group
        self.length_idx = self.length_idx[self.length_idx[:, 1].argsort()]
        inds = self.length_idx[:, 0].tolist()
        yield from iter(inds)

    def __len__(self) -> int:
        return len(self.length_idx)

def pad_collate(batch):
    # Get max lengths in the batch
    max_es_len = max([len(x["es_ids"]) for x in batch])
    max_en_len = max([len(x["en_ids"]) for x in batch])
    # Pad batch to max length
    for i in range(len(batch)):
        batch[i]["es_ids"] = torch.cat(
            [batch[i]["es_ids"], torch.zeros(max_es_len - len(batch[i]["es_ids"]))]
        )
        batch[i]["en_ids"] = torch.cat(
            [batch[i]["en_ids"], torch.zeros(max_en_len - len(batch[i]["en_ids"]))]
        )
    return default_collate(batch)