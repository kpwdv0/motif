import os
import json
import torch
from torch.utils.data import Dataset


class MotifDataset(Dataset):


    def __init__(self, precomputed_dir):
        self.precomputed_dir = precomputed_dir

        index_path = os.path.join(precomputed_dir, "index.json")
        with open(index_path) as f:
            self.piece_index = json.load(f)

        self.flat_index = []
        for entry in self.piece_index:
            for local_i in range(entry["num_examples"]):
                self.flat_index.append((entry["file"], local_i))

    def __len__(self):
        return len(self.flat_index)

    def __getitem__(self, idx):
        filename, local_i = self.flat_index[idx]
        path = os.path.join(self.precomputed_dir, filename)

        with open(path) as f:
            piece_examples = json.load(f)

        example = piece_examples[local_i]

     
        context_pitches = torch.tensor([note[2] for note in example["context"]], dtype=torch.long)
        target_pitches = torch.tensor([note[2] for note in example["target"]], dtype=torch.long)

        return {
            "context": context_pitches,
            "target": target_pitches,
            "motif_patterns": list(example["motifs"].keys()),
        }

    


#ok so i added this basically it pads everythign so i can feed everything in one batch cs eveerythings different
import torch
from torch.nn.utils.rnn import pad_sequence

PAD_VALUE = 128

def motif_collate_fn(batch):

    contexts = [item["context"] for item in batch]
    targets = [item["target"] for item in batch]

    context_lengths = torch.tensor([len(c) for c in contexts])
    target_lengths = torch.tensor([len(t) for t in targets])

    padded_contexts = pad_sequence(contexts, batch_first=True, padding_value=PAD_VALUE)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=PAD_VALUE)

    return {
        "context": padded_contexts,
        "context_lengths": context_lengths,
        "target": padded_targets,
        "target_lengths": target_lengths,
        "motif_patterns": [item["motif_patterns"] for item in batch],
    }
