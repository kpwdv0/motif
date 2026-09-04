r"""

from doingstuff import loadnotes, skyline_melody, note_transitions, find_motifs

path = r"C:\Users\kira\data\maestro-v3.0.0\2018\MIDI-Unprocessed_Recital1-3_MID--AUDIO_01_R1_2018_wav--1.midi"
notes = loadnotes(path)
melody = skyline_melody(notes)

motifs = find_motifs(melody, window_size=6)
print(len(motifs), "repeated patterns found")

for pattern, positions in list(motifs.items())[:10]:
    print(len(positions), "occurrences ->", pattern, "at", positions)

"""

r"""
from doingstuff import loadnotes, skyline_melody, note_transitions, find_motifs

path = r"C:\Users\kira\data\maestro-v3.0.0\2018\MIDI-Unprocessed_Recital1-3_MID--AUDIO_01_R1_2018_wav--1.midi"
notes = loadnotes(path)
melody = skyline_melody(notes)

motifs = find_motifs(melody, window_size=6)
print(len(motifs), "repeated patterns found")


sorted_motifs = sorted(motifs.items(), key=lambda item: len(item[1]), reverse=True)

for pattern, positions in sorted_motifs[:10]:
    print(len(positions), "occurrences ->", pattern, "at", positions)

"""


r"""

from motifstuff import MotifDataset

dataset = MotifDataset(r"C:\Users\kira\data\maestro-precomputed")
print(len(dataset), "total examples")

example = dataset[0]
print("context shape:", example["context"].shape)
print("target shape:", example["target"].shape)
print("motif patterns:", example["motif_patterns"][:3])

"""

r"""
from torch.utils.data import DataLoader
from motifstuff import MotifDataset, motif_collate_fn

dataset = MotifDataset(r"C:\Users\kira\data\maestro-precomputed")
loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=motif_collate_fn)

batch = next(iter(loader))
print("context batch shape:", batch["context"].shape)
print("target batch shape:", batch["target"].shape)
print("context lengths:", batch["context_lengths"])

"""

r"""
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("loubb/aria-medium-base", trust_remote_code=True)
print(model)
"""

r"""
some output:
'
(motif) PS C:\Users\kira\motif> python testingstuff.py
AriaForCausalLM(
  (model): AriaModel(
    (tok_embeddings): Embedding(17727, 1536)
    (out_layer_norm): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
    (encode_layers): ModuleList(
      (0-15): 16 x TransformerBlock(
        (mixed_qkv): Linear(in_features=1536, out_features=4608, bias=False)
        (att_proj_linear): Linear(in_features=1536, out_features=1536, bias=False)
        (ff_gate_proj): Linear(in_features=1536, out_features=6144, bias=False)
        (ff_up_proj): Linear(in_features=1536, out_features=6144, bias=False)
        (ff_down_proj): Linear(in_features=6144, out_features=1536, bias=False)
        (norm1): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
        (norm2): LayerNorm((1536,), eps=1e-05, elementwise_affine=True)
      )
    )
  )
  (lm_head): Linear(in_features=1536, out_features=17727, bias=False)
)


okay so freeze tok_embeddings obviously

and then do stuff on the 16 transformer blocks!!
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from motifattention import MotifCrossAttention, motif_notes_to_vector, BlockWithMotifAttention, train_one_epoch, generate_continuation, notes_to_token_ids, run_ablation_sweep, load_maestro_splits, train_on_split_only, eval_on_split_only, group_index_by_split, train_with_epochs
from doingstuff import loadnotes, skyline_melody, find_motifs
import torch
import torch.nn.functional as F
import time
import json
import os
import math
import inspect
import motifattention

model = AutoModelForCausalLM.from_pretrained("loubb/aria-medium-base", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("loubb/aria-medium-base", trust_remote_code=True)

path = r"C:\Users\kira\data\maestro-v3.0.0\2018\MIDI-Unprocessed_Recital1-3_MID--AUDIO_01_R1_2018_wav--1.midi"
notes = loadnotes(path)
melody = skyline_melody(notes)
motifs = find_motifs(melody, window_size=6)

first_pattern, positions = next(iter(motifs.items()))
first_occurrence_start = positions[0]
motif_notes = melody[first_occurrence_start : first_occurrence_start + 7]

vector = motif_notes_to_vector(motif_notes, tokenizer, model)
print("motif vector shape:", vector.shape)

hidden_dim = model.config.hidden_size
print("hidden_dim:", hidden_dim)
model.model.encode_layers[14] = BlockWithMotifAttention(model.model.encode_layers[14], hidden_dim)
print("swapped in successfully")


r"""
(motif) PS C:\Users\kira\motif> python testingstuff.py
motif vector shape: torch.Size([1536])
hidden_dim: 1536
swapped in successfully
example 10: loss = 2.4401, gate = 0.5000
example 20: loss = 1.5816, gate = 0.5000
example 30: loss = 2.8027, gate = 0.5000
example 40: loss = 2.1526, gate = 0.5000
example 50: loss = 3.1238, gate = 0.5000

done: 50 examples, avg loss = 2.2513
(motif) PS C:\Users\kira\motif>     

"""

patched_block = model.model.encode_layers[14]

model.load_state_dict(torch.load("checkpoint.pt"))

start = time.time()
with_motifs = generate_continuation(model, tokenizer, patched_block, path, use_motifs=True, motif_vector=vector)
print(f"with-motifs generation took {time.time() - start:.1f}s")

start = time.time()
without_motifs = generate_continuation(model, tokenizer, patched_block, path, use_motifs=False)
print(f"without-motifs generation took {time.time() - start:.1f}s")

midi_dict_with = tokenizer.decode(with_motifs[0].tolist())
midi_dict_without = tokenizer.decode(without_motifs[0].tolist())

midi_dict_with.to_midi().save("generated_with_motifs.mid")
midi_dict_without.to_midi().save("generated_without_motifs.mid")

print("saved generated_with_motifs.mid and generated_without_motifs.mid")

r"""
(motif) PS C:\Users\kira\motif> python testingstuff.py
motif vector shape: torch.Size([1536])
hidden_dim: 1536
swapped in successfully
with-motifs generation took 5.3s
without-motifs generation took 4.7s
saved generated_with_motifs.mid and generated_without_motifs.mid
"""

model.eval()
total_loss = 0.0
count = 0

with open(r"C:\Users\kira\data\maestro-precomputed\index.json") as f:
    piece_index = json.load(f)

for entry in piece_index[:20]:
    with open(os.path.join(r"C:\Users\kira\data\maestro-precomputed", entry["file"])) as f:
        examples = json.load(f)

    for example in examples[:2]:
        motif_vecs = []
        for pattern_str, positions in list(example["motifs"].items())[:3]:
            pos = positions[0]
            motif_notes = example["context"][pos:pos + 7]
            if len(motif_notes) < 2:
                continue
            motif_vecs.append(motif_notes_to_vector(motif_notes, tokenizer, model))

        if not motif_vecs:
            continue

        patched_block.motif_vectors = torch.stack(motif_vecs).unsqueeze(0)

        full_notes = example["context"] + example["target"]
        try:
            input_ids = notes_to_token_ids(full_notes, tokenizer)[:, :512]
        except Exception:
            continue
        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            output = model(input_ids)
            logits = output[0]
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
            )
        total_loss += loss.item()
        count += 1

print(f"quick eval: {count} examples, avg loss = {total_loss / max(count,1):.4f}")


r"""
reran this so i could get some more stuff

(motif) PS C:\Users\kira\motif> python testingstuff.py
motif vector shape: torch.Size([1536])
hidden_dim: 1536
swapped in successfully
with-motifs generation took 5.3s
without-motifs generation took 4.7s
saved generated_with_motifs.mid and generated_without_motifs.mid
quick eval: 40 examples, avg loss = 2.7282
(motif) PS C:\Users\kira\motif> 

"""

gate_value = torch.sigmoid(patched_block.motif_attn.gate).item()
print(f"trained gate value: {gate_value:.4f}")

def quick_eval(use_motifs, num_pieces=20):
    model.eval()
    total_loss = 0.0
    count = 0

    with open(r"C:\Users\kira\data\maestro-precomputed\index.json") as f:
        piece_index = json.load(f)

    for entry in piece_index[:num_pieces]:
        with open(os.path.join(r"C:\Users\kira\data\maestro-precomputed", entry["file"])) as f:
            examples = json.load(f)

        for example in examples[:2]:
            if use_motifs:
                motif_vecs = []
                for pattern_str, positions in list(example["motifs"].items())[:3]:
                    pos = positions[0]
                    motif_notes = example["context"][pos:pos + 7]
                    if len(motif_notes) < 2:
                        continue
                    motif_vecs.append(motif_notes_to_vector(motif_notes, tokenizer, model))
                if not motif_vecs:
                    continue
                patched_block.motif_vectors = torch.stack(motif_vecs).unsqueeze(0)
            else:
                patched_block.motif_vectors = None

            full_notes = example["context"] + example["target"]
            try:
                input_ids = notes_to_token_ids(full_notes, tokenizer)[:, :512]
            except Exception:
                continue
            if input_ids.shape[1] < 2:
                continue

            with torch.no_grad():
                output = model(input_ids)
                logits = output[0]
                loss = F.cross_entropy(
                    logits[:, :-1, :].reshape(-1, logits.size(-1)),
                    input_ids[:, 1:].reshape(-1),
                )
            total_loss += loss.item()
            count += 1

    avg_loss = total_loss / max(count, 1)
    perplexity = math.exp(avg_loss)
    return count, avg_loss, perplexity


count_with, loss_with, ppl_with = quick_eval(use_motifs=True)
count_without, loss_without, ppl_without = quick_eval(use_motifs=False)

print(f"\nWITH motifs:    {count_with} examples, avg loss = {loss_with:.4f}, perplexity = {ppl_with:.2f}")
print(f"WITHOUT motifs: {count_without} examples, avg loss = {loss_without:.4f}, perplexity = {ppl_without:.2f}")
print(f"difference: {loss_without - loss_with:+.4f} loss ({'motifs help' if loss_with < loss_without else 'motifs do not help on this sample'})")


r"""

(motif) PS C:\Users\kira\motif> python testingstuff.py
motif vector shape: torch.Size([1536])
hidden_dim: 1536
swapped in successfully
with-motifs generation took 5.2s
without-motifs generation took 4.8s
saved generated_with_motifs.mid and generated_without_motifs.mid
quick eval: 40 examples, avg loss = 2.7282
trained gate value: 0.5001

WITH motifs:    40 examples, avg loss = 2.7282, perplexity = 15.30
WITHOUT motifs: 40 examples, avg loss = 2.7377, perplexity = 15.45
difference: +0.0095 loss (motifs help)
errrr kinda help idk 
"""


r"""
print(model.config)

output:
AriaConfig {
  "architectures": [
    "AriaForCausalLM"
  ],
  "auto_map": {
    "AutoConfig": "configuration_aria.AriaConfig",
    "AutoModel": "modeling_aria.AriaModel",
    "AutoModelForCausalLM": "modeling_aria.AriaForCausalLM"
  },
  "dtype": "float32",
  "embedding_size": null,
  "eos_token_id": 1,
  "hidden_size": 1536,
  "intermediate_size": 6144,
  "max_seq_len": 8192,
  "model_type": "aria",
  "num_attention_heads": 24,
  "num_hidden_layers": 16,
  "pad_token_id": 2,
  "return_dict": false,
  "tie_word_embeddings": false,
  "transformers_version": "4.57.6",
  "use_cache": true,
  "vocab_size": 17727
}

no dropout field present - checked the actual model source, found:
self.drop_p = 0.0  in TransformerBlock.__init__, but never actually
used anywhere in the forward pass. Aria's own attention/ff layers run
with effectively zero dropout, even though the attribute exists.
"""

r"""
print(inspect.getsourcefile(type(model)))

output:
C:\Users\kira\.cache\huggingface\modules\transformers_modules\loubb\aria_hyphen_medium_hyphen_base\5a1ef36c5007ee10fbe57d61d96725a7244bd2b2\modeling_aria.py
"""


splits = load_maestro_splits(r"C:\Users\kira\data\maestro-v3.0.0\maestro-v3.0.0.csv")
grouped = group_index_by_split(r"C:\Users\kira\data\maestro-precomputed", splits)

test_grouped = {"train": grouped["train"][:5]}

original_group_fn = motifattention.group_index_by_split
motifattention.group_index_by_split = lambda *args, **kwargs: test_grouped

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)

train_with_epochs(model, patched_block, tokenizer, r"C:\Users\kira\data\maestro-precomputed", optimizer, splits, num_epochs=2, use_wandb=True, wandb_run_name="epoch-test")

motifattention.group_index_by_split = original_group_fn
