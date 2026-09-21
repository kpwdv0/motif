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
from motifstuff import MotifDataset, motif_collate_fn  # adjust import if collate_fn lives elsewhere

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
from motifattention import MotifCrossAttention, motif_notes_to_vector, BlockWithMotifAttention, train_one_epoch, generate_continuation, notes_to_token_ids, run_ablation_sweep, load_maestro_splits, train_on_split_only, eval_on_split_only, group_index_by_split, train_with_epochs, build_batch, select_fixed_examples, visualize_example, generate_with_motif_switching, test_gate_forcing
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

# first motif found
first_pattern, positions = next(iter(motifs.items()))
first_occurrence_start = positions[0]
motif_notes = melody[first_occurrence_start : first_occurrence_start + 7]  # 6 transitions = 7 notes

vector = motif_notes_to_vector(motif_notes, tokenizer, model)
print("motif vector shape:", vector.shape)

hidden_dim = model.config.hidden_size
print("hidden_dim:", hidden_dim)
model.model.encode_layers[14] = BlockWithMotifAttention(model.model.encode_layers[14], hidden_dim)
print("swapped in successfully")

r"""
old single forward pass tests
prompt = tokenizer.encode_from_file(path, return_tensors="pt")

with torch.no_grad():
    output = model(prompt.input_ids[..., :100])

logits = output[0]
print("forward pass succeeded")
print(logits.shape)

motif_vecs = vector.unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 1536)

# the wrapped block
patched_block = model.model.encode_layers[14]
patched_block.motif_vectors = motif_vecs

with torch.no_grad():
    output = model(prompt.input_ids[..., :100])

print("forward pass with real motif attention succeeded")
print(output[0].shape)
"""

r"""
training run, finished overnight. checkpoint.pt is the result
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-5
)

patched_block = model.model.encode_layers[14]
train_one_epoch(model, patched_block, tokenizer, r"C:\Users\kira\data\maestro-precomputed", optimizer, max_examples=None, checkpoint_every=200)
"""

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

# before/after generation with the trained checkpoint
patched_block = model.model.encode_layers[14]

model.load_state_dict(torch.load("checkpoint.pt"))

start = time.time()
with_motifs = generate_continuation(model, tokenizer, patched_block, path, use_motifs=True, motif_vector=vector)
print(f"with-motifs generation took {time.time() - start:.1f}s")

start = time.time()
without_motifs = generate_continuation(model, tokenizer, patched_block, path, use_motifs=False)
print(f"without-motifs generation took {time.time() - start:.1f}s")

# back to midi
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

# quick loss check on the trained checkpoint
model.eval()
total_loss = 0.0
count = 0

with open(r"C:\Users\kira\data\maestro-precomputed\index.json") as f:
    piece_index = json.load(f)

for entry in piece_index[:20]:  # first 20 pieces only
    with open(os.path.join(r"C:\Users\kira\data\maestro-precomputed", entry["file"])) as f:
        examples = json.load(f)

    for example in examples[:2]:  # first 2 examples per piece
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

# gate after training
gate_value = torch.sigmoid(patched_block.motif_attn.gate).item()
print(f"trained gate value: {gate_value:.4f}")

# loss with motifs on vs off
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
wandb test run, worked
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=1e-5
)

train_one_epoch(model, patched_block, tokenizer, r"C:\Users\kira\data\maestro-precomputed", optimizer, max_examples=30, use_wandb=True, wandb_run_name="wandb-test")
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

no dropout in the config. looked at the model source and TransformerBlock sets
self.drop_p = 0.0 but never uses it in forward, so aria basically has no dropout
"""

r"""
print(inspect.getsourcefile(type(model)))

output:
C:\Users\kira\.cache\huggingface\modules\transformers_modules\loubb\aria_hyphen_medium_hyphen_base\5a1ef36c5007ee10fbe57d61d96725a7244bd2b2\modeling_aria.py
"""

r"""
ablation sweep, all 7 runs done (logged to wandb)
run_ablation_sweep("loubb/aria-medium-base", tokenizer, r"C:\Users\kira\data\maestro-precomputed", examples_per_run=800)

output:
=== run: baseline (lr=1e-05, dropout=0.0, accum=1) ===
...
=== run: accum_16 (lr=1e-05, dropout=0.0, accum=16) ===
...
done: 800 examples, avg loss = 2.4318
final checkpoint saved to checkpoint_accum_16.pt
all 7 ablation runs complete: ['baseline', 'lr_1e-6', 'lr_1e-4', 'dropout_0.1', 'dropout_0.2', 'accum_4', 'accum_16']
"""

r"""
earlier attempt at a held out val check. never ran it, replaced by the epoch based train/val/test setup

splits = load_maestro_splits(r"C:\Users\kira\data\maestro-v3.0.0\maestro-v3.0.0.csv")

fresh_model = AutoModelForCausalLM.from_pretrained("loubb/aria-medium-base", trust_remote_code=True)
for param in fresh_model.model.tok_embeddings.parameters():
    param.requires_grad = False
fresh_model.model.encode_layers[14] = BlockWithMotifAttention(fresh_model.model.encode_layers[14], fresh_model.config.hidden_size)
fresh_patched_block = fresh_model.model.encode_layers[14]
fresh_optimizer = torch.optim.AdamW([p for p in fresh_model.parameters() if p.requires_grad], lr=1e-5)

train_on_split_only(fresh_model, fresh_patched_block, tokenizer, fresh_optimizer, r"C:\Users\kira\data\maestro-precomputed", splits, target_split="train", max_examples=300)

count_v, loss_v, ppl_v = eval_on_split_only(fresh_model, fresh_patched_block, tokenizer, r"C:\Users\kira\data\maestro-precomputed", splits, target_split="validation", use_motifs=True)
print(f"held-out validation: {count_v} examples, loss={loss_v:.4f}, perplexity={ppl_v:.2f}")
"""

r"""
checking the maestro split lookup
splits = load_maestro_splits(r"C:\Users\kira\data\maestro-v3.0.0\maestro-v3.0.0.csv")
grouped = group_index_by_split(r"C:\Users\kira\data\maestro-precomputed", splits)

output:
grouped pieces: train=961, validation=137, test=177, unmatched=0
"""

r"""
small test of train_with_epochs with batching (batch_size=4): 5 pieces, 2 epochs, worked

splits = load_maestro_splits(r"C:\Users\kira\data\maestro-v3.0.0\maestro-v3.0.0.csv")
grouped = group_index_by_split(r"C:\Users\kira\data\maestro-precomputed", splits)

test_grouped = {"train": grouped["train"][:5]}
original_group_fn = motifattention.group_index_by_split
motifattention.group_index_by_split = lambda *args, **kwargs: test_grouped

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)

train_with_epochs(model, patched_block, tokenizer, r"C:\Users\kira\data\maestro-precomputed", optimizer, splits, num_epochs=2, use_wandb=True, wandb_run_name="epoch-test")

motifattention.group_index_by_split = original_group_fn
"""

r"""
the real run (monophonic targets): full 961 piece train split, 3 epochs, ran overnight. results:

epoch 1 done: 961 batches (3844 examples, batch_size=4), avg loss = 1.9600
epoch 1 validation: 30 examples, loss = 2.4462, perplexity = 11.54
epoch 2 done: 961 batches (3844 examples, batch_size=4), avg loss = 1.3879
epoch 2 validation: 30 examples, loss = 2.5518, perplexity = 12.83
epoch 3 done: 961 batches (3844 examples, batch_size=4), avg loss = 0.9879
epoch 3 validation: 30 examples, loss = 3.0843, perplexity = 21.85

finding: train loss keeps dropping but val loss gets worse every epoch so it's overfitting.
epoch 1 was probably the best one, but this was before i fixed the checkpoint saving so
checkpoint_epochs.pt is epoch 3 (the worst). fixed in train_with_epochs, reruns save every epoch.
(this used the old monophonic target dataset)

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)

train_with_epochs(
    model, patched_block, tokenizer, r"C:\Users\kira\data\maestro-precomputed", optimizer, splits,
    num_epochs=3, batch_size=4, use_wandb=True, wandb_run_name="real-epoch-run",
    checkpoint_path="checkpoint_epochs.pt",
)
"""

r"""
30 example (10 train/val/test) piano roll + audio set, untrained (visualizations/) and with the
epoch 3 checkpoint (visualizations_trained/). context and target were both still skyline melody
here so every piano roll came out monophonic. that's why i switched to polyphonic targets below.

splits = load_maestro_splits(r"C:\Users\kira\data\maestro-v3.0.0\maestro-v3.0.0.csv")
fixed = select_fixed_examples(r"C:\Users\kira\data\maestro-precomputed", splits, n=10)

for split_name in ["train", "validation", "test"]:
    for example_entry in fixed[split_name]:
        visualize_example(example_entry, model, tokenizer, patched_block, r"C:\Users\kira\motif\visualizations", label=split_name)

trained_model = AutoModelForCausalLM.from_pretrained("loubb/aria-medium-base", trust_remote_code=True)
for param in trained_model.model.tok_embeddings.parameters():
    param.requires_grad = False
trained_model.model.encode_layers[14] = BlockWithMotifAttention(trained_model.model.encode_layers[14], trained_model.config.hidden_size)
trained_patched_block = trained_model.model.encode_layers[14]

trained_model.load_state_dict(torch.load("checkpoint_epochs.pt"))

for split_name in ["train", "validation", "test"]:
    for example_entry in fixed[split_name]:
        visualize_example(example_entry, trained_model, tokenizer, trained_patched_block, r"C:\Users\kira\motif\visualizations_trained", label=split_name)
"""

r"""
redesign: motifs and context stay as skyline melody but the target is now the full polyphonic
performance after the cut. added build_training_examples_polyphonic to doingstuff.py,
computingstuff.py calls it now, and note_fields() in motifattention.py handles 3 and 4 element notes.
"""

r"""
quick test of the polyphonic precompute_piece on one file, works. targets are bigger than
the context now (756 melody notes -> 1276 target notes). one example had a smaller target
(context 1039, target 633) because the cut landed late in the piece (275s), which is expected.

from computingstuff import precompute_piece

test_path = r"C:\Users\kira\data\maestro-v3.0.0\2018\MIDI-Unprocessed_Recital1-3_MID--AUDIO_01_R1_2018_wav--1.midi"
examples = precompute_piece(test_path)

print(f"{len(examples)} examples produced")
for ex in examples[:2]:
    print(f"context: {len(ex['context'])} melody notes, target: {len(ex['target'])} polyphonic notes, {len(ex['motifs'])} motifs, cut_time={ex['cut_time']:.2f}")
"""

r"""
regenerated the whole dataset with polyphonic targets (overwrote the old maestro-precomputed folder):

finished year 2014: 913 pieces so far, 0 errors so far
finished year 2015: 1042 pieces so far, 0 errors so far
finished year 2017: 1182 pieces so far, 0 errors so far
finished year 2018: 1275 pieces so far, 0 errors so far
done in 328.4s
1275 pieces succeeded, 0 failed, 5100 total training examples

same counts as the monophonic run (5100) so only what's inside the examples changed

from computingstuff import precompute_all

precompute_all(
    maestro_root=r"C:\Users\kira\data\maestro-v3.0.0",
    output_dir=r"C:\Users\kira\data\maestro-precomputed",
)
"""

r"""
retrained on the polyphonic dataset overnight as a background process. the checkpoint fix
worked this time (checkpoint_epochs_v2_epoch1/2/3.pt). results:

epoch 1 done: 961 batches (3844 examples, batch_size=4), avg loss = 2.0031
epoch 1 validation: 30 examples, loss = 2.4608, perplexity = 11.71
epoch 2 done: 961 batches (3844 examples, batch_size=4), avg loss = 1.4392
epoch 2 validation: 30 examples, loss = 2.5567, perplexity = 12.89
epoch 3 done: 961 batches (3844 examples, batch_size=4), avg loss = 1.0445
epoch 3 validation: 30 examples, loss = 3.0378, perplexity = 20.86

best epoch by validation loss: epoch 1 (val_loss=2.4608)

finding: same overfitting as the monophonic run. gate is still 0.5001, basically hasn't moved, 4th run in a row

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)

train_with_epochs(
    model, patched_block, tokenizer, r"C:\Users\kira\data\maestro-precomputed", optimizer, splits,
    num_epochs=3, batch_size=4, use_wandb=True, wandb_run_name="polyphonic-target-run",
    checkpoint_path="checkpoint_epochs_v2.pt",
)
"""

r"""
visualized the same 30 examples with checkpoint_epochs_v2_epoch1.pt (best epoch, polyphonic data),
saved to visualizations_v2_best/. first polyphonic before/after

splits = load_maestro_splits(r"C:\Users\kira\data\maestro-v3.0.0\maestro-v3.0.0.csv")
fixed = select_fixed_examples(r"C:\Users\kira\data\maestro-precomputed", splits, n=10)

best_model = AutoModelForCausalLM.from_pretrained("loubb/aria-medium-base", trust_remote_code=True)
for param in best_model.model.tok_embeddings.parameters():
    param.requires_grad = False
best_model.model.encode_layers[14] = BlockWithMotifAttention(best_model.model.encode_layers[14], best_model.config.hidden_size)
best_patched_block = best_model.model.encode_layers[14]

best_model.load_state_dict(torch.load("checkpoint_epochs_v2_epoch1.pt"))
print("loaded checkpoint_epochs_v2_epoch1.pt (best epoch, polyphonic-target dataset)")

for split_name in ["train", "validation", "test"]:
    for example_entry in fixed[split_name]:
        visualize_example(example_entry, best_model, tokenizer, best_patched_block, r"C:\Users\kira\motif\visualizations_v2_best", label=split_name)
"""

# reload the best checkpoint (earlier blocks might have been run out of order)
best_model = AutoModelForCausalLM.from_pretrained("loubb/aria-medium-base", trust_remote_code=True)
for param in best_model.model.tok_embeddings.parameters():
    param.requires_grad = False
best_model.model.encode_layers[14] = BlockWithMotifAttention(best_model.model.encode_layers[14], best_model.config.hidden_size)
best_patched_block = best_model.model.encode_layers[14]
best_model.load_state_dict(torch.load("checkpoint_epochs_v2_epoch1.pt"))
print("loaded checkpoint_epochs_v2_epoch1.pt for motif-switching + gate-forcing tests")

# seed context and two hand written motifs: A (C-E-G-C arpeggio), B (descending, higher)
seed_path = r"C:\Users\kira\data\maestro-v3.0.0\2018\MIDI-Unprocessed_Recital1-3_MID--AUDIO_01_R1_2018_wav--1.midi"
seed_notes_full = loadnotes(seed_path)
seed_melody = skyline_melody(seed_notes_full)
seed_notes = seed_melody[:100]  # first 100 melody notes as the seed context

motif_a = [(0.0, 0.3, 60), (0.3, 0.6, 64), (0.6, 0.9, 67), (0.9, 1.2, 60)]  # C-E-G-C
motif_b = [(0.0, 0.2, 72), (0.2, 0.4, 69), (0.4, 0.6, 65), (0.6, 0.8, 60)]  # descending, higher register

# switch motifs mid generation: A, B, A
result_ids = generate_with_motif_switching(
    best_model, tokenizer, best_patched_block, seed_notes,
    motif_sequence=[motif_a, motif_b, motif_a],
    tokens_per_segment=50,
)

midi_dict = tokenizer.decode(result_ids[0].tolist())
mido_midi = midi_dict.to_midi()
mido_midi.save(r"C:\Users\kira\motif\motif_switching_demo.mid")
print("saved motif_switching_demo.mid")

# does forcing the gate to either extreme change the output, whatever value it learned?
logits_off = test_gate_forcing(best_model, best_patched_block, tokenizer, seed_notes, motif_a, gate_override=-10.0)
logits_on = test_gate_forcing(best_model, best_patched_block, tokenizer, seed_notes, motif_a, gate_override=10.0)

diff = (logits_on - logits_off).abs()
print(f"max logit difference: {diff.max().item():.4f}")
print(f"mean logit difference: {diff.mean().item():.4f}")

top_off = logits_off[0, -1].argmax().item()
top_on = logits_on[0, -1].argmax().item()
print(f"top predicted token, gate forced off: {top_off}")
print(f"top predicted token, gate forced on: {top_on}")
print(f"prediction changed: {top_off != top_on}")