import csv
import torch
from transformers import AutoModelForCausalLM
import torch.nn as nn
import math
from torch.optim.lr_scheduler import CosineAnnealingLR


def note_fields(note):
    """works on 3 element melody notes and 4 element raw notes, velocity defaults to 80"""
    if len(note) > 3:
        return note[0], note[1], note[2], note[3]
    return note[0], note[1], note[2], 80


class MotifCrossAttention(nn.Module):

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states, motif_vectors, motif_padding_mask=None):

        attn_out, _ = self.cross_attn(
            query=hidden_states,
            key=motif_vectors,
            value=motif_vectors,
            key_padding_mask=motif_padding_mask,
        )

        gate_value = torch.sigmoid(self.gate)
        return hidden_states + gate_value * attn_out


import tempfile
import os
import torch
import pretty_midi

def motif_notes_to_vector(motif_notes, tokenizer, model):

    # write the motif to a tiny midi file so the tokenizer can read it
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)  # piano

    offset = motif_notes[0][0]
    for start, end, pitch in motif_notes:
        note = pretty_midi.Note(
            velocity=80, 
            pitch=int(pitch),
            start=start - offset,
            end=end - offset,
        )
        instrument.notes.append(note)

    midi.instruments.append(instrument)


    with tempfile.NamedTemporaryFile(suffix=".midi", delete=False) as tmp:
        tmp_path = tmp.name
    midi.write(tmp_path)

    try:
        token_ids = tokenizer.encode_from_file(tmp_path, return_tensors="pt")
    finally:
        os.remove(tmp_path)

    # mean pool the token embeddings (table is frozen)
    with torch.no_grad():
        token_embeds = model.model.tok_embeddings(token_ids.input_ids)  # shape: (1, num_tokens, 1536)
        pooled = token_embeds.mean(dim=1)  # shape: (1, 1536)

    return pooled.squeeze(0)  # shape: (1536,)

class BlockWithMotifAttention(nn.Module):
  
    def __init__(self, original_block, hidden_dim, dropout=0.1):
        super().__init__()
        self.original_block = original_block
        self.motif_attn = MotifCrossAttention(hidden_dim, dropout=dropout)
        self.motif_vectors = None
        self.motif_padding_mask = None

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        block_output = self.original_block(hidden_states, attention_mask, **kwargs)

        if isinstance(block_output, tuple):
            hidden_states, *extras = block_output
        else:
            hidden_states = block_output
            extras = []

        if self.motif_vectors is not None:
            hidden_states = self.motif_attn(hidden_states, self.motif_vectors, self.motif_padding_mask)

        if extras:
            return (hidden_states, *extras)
        return hidden_states


import json
import torch
import torch.nn.functional as F
import pretty_midi
import tempfile
import os

def notes_to_token_ids(notes, tokenizer):
    """notes -> aria token ids"""
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    offset = notes[0][0]
    for note in notes:
        start, end, pitch, velocity = note_fields(note)
        instrument.notes.append(pretty_midi.Note(
            velocity=velocity, pitch=int(pitch), start=start - offset, end=end - offset
        ))
    midi.instruments.append(instrument)

    with tempfile.NamedTemporaryFile(suffix=".midi", delete=False) as tmp:
        tmp_path = tmp.name
    midi.write(tmp_path)
    try:
        result = tokenizer.encode_from_file(tmp_path, return_tensors="pt")
    finally:
        os.remove(tmp_path)
    return result.input_ids  # shape: (1, num_tokens)

import time
import wandb

def train_one_epoch(model, patched_block, tokenizer, precomputed_dir, optimizer, max_examples=None, checkpoint_every=200, checkpoint_path="checkpoint.pt", use_wandb=False, wandb_project="motif-memory", wandb_run_name=None, accumulation_steps=1):

    if use_wandb:
        wandb.init(project=wandb_project, name=wandb_run_name)

    with open(os.path.join(precomputed_dir, "index.json")) as f:
        piece_index = json.load(f)

    model.train()
    total_loss = 0.0
    count = 0
    start_time = time.time()
    optimizer.zero_grad()

    for entry in piece_index:
        if max_examples is not None and count >= max_examples:
            break

        with open(os.path.join(precomputed_dir, entry["file"])) as f:
            examples = json.load(f)

        for example in examples:
            if max_examples is not None and count >= max_examples:
                break

            motif_vecs = []
            for pattern_str, positions in list(example["motifs"].items())[:3]:
                pos = positions[0]
                motif_notes = example["context"][pos:pos + 7]
                if len(motif_notes) < 2:
                    continue
                vec = motif_notes_to_vector(motif_notes, tokenizer, model)
                motif_vecs.append(vec)

            if not motif_vecs:
                continue

            motif_tensor = torch.stack(motif_vecs).unsqueeze(0)
            patched_block.motif_vectors = motif_tensor
            patched_block.motif_padding_mask = None

            full_notes = example["context"] + example["target"]
            if len(full_notes) < 2:
                continue

            try:
                input_ids = notes_to_token_ids(full_notes, tokenizer)
            except Exception:
                continue

            input_ids = input_ids[:, :512]
            if input_ids.shape[1] < 2:
                continue

            output = model(input_ids)
            logits = output[0]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            (loss / accumulation_steps).backward()

            count += 1

            if count % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

            gate_val = torch.sigmoid(patched_block.motif_attn.gate).item()

            if use_wandb:
                wandb.log({
                    "loss": loss.item(),
                    "gate": gate_val,
                    "avg_loss_so_far": total_loss / count,
                    "example": count,
                })

            if count % 10 == 0:
                elapsed = time.time() - start_time
                rate = elapsed / count
                print(f"example {count}: loss = {loss.item():.4f}, gate = {gate_val:.4f}, "
                      f"{rate:.2f}s/example, elapsed {elapsed/60:.1f}min")

            if count % checkpoint_every == 0:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"  -> checkpoint saved at example {count}")

    if count % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    torch.save(model.state_dict(), checkpoint_path)
    print(f"\ndone: {count} examples, avg loss = {total_loss / max(count, 1):.4f}")
    print(f"final checkpoint saved to {checkpoint_path}")

    if use_wandb:
        wandb.finish()

def generate_continuation(model, tokenizer, patched_block, midi_path, num_tokens=100, use_motifs=True, motif_vector=None):
    """generate from a real piece with motif attention on or off"""
    prompt = tokenizer.encode_from_file(midi_path, return_tensors="pt")
    input_ids = prompt.input_ids[..., :100]  # use first 100 tokens as the seed

    if use_motifs and motif_vector is not None:
        patched_block.motif_vectors = motif_vector.unsqueeze(0).unsqueeze(0)
    else:
        patched_block.motif_vectors = None

    model.eval()
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            max_new_tokens=num_tokens,
            do_sample=True,
            temperature=0.9,
        )

    return generated


def run_ablation_sweep(base_model_name, tokenizer, precomputed_dir, layer_idx=14, examples_per_run=800):

    baseline = {"lr": 1e-5, "dropout": 0.0, "accum": 1}

    runs = [
        {**baseline, "name": "baseline"},
        {**baseline, "lr": 1e-6, "name": "lr_1e-6"},
        {**baseline, "lr": 1e-4, "name": "lr_1e-4"},
        {**baseline, "dropout": 0.1, "name": "dropout_0.1"},
        {**baseline, "dropout": 0.2, "name": "dropout_0.2"},
        {**baseline, "accum": 4, "name": "accum_4"},
        {**baseline, "accum": 16, "name": "accum_16"},
    ]

    results = []

    for cfg in runs:
        print(f"\n=== run: {cfg['name']} (lr={cfg['lr']}, dropout={cfg['dropout']}, accum={cfg['accum']}) ===")

        run_model = AutoModelForCausalLM.from_pretrained(base_model_name, trust_remote_code=True)
        for param in run_model.model.tok_embeddings.parameters():
            param.requires_grad = False

        hidden_dim = run_model.config.hidden_size
        run_model.model.encode_layers[layer_idx] = BlockWithMotifAttention(
            run_model.model.encode_layers[layer_idx], hidden_dim, dropout=cfg["dropout"]
        )
        run_patched_block = run_model.model.encode_layers[layer_idx]

        run_optimizer = torch.optim.AdamW(
            [p for p in run_model.parameters() if p.requires_grad], lr=cfg["lr"]
        )

        train_one_epoch(
            run_model, run_patched_block, tokenizer, precomputed_dir, run_optimizer,
            max_examples=examples_per_run, use_wandb=True, wandb_run_name=cfg["name"],
            accumulation_steps=cfg["accum"], checkpoint_path=f"checkpoint_{cfg['name']}.pt",
        )

        results.append(cfg["name"])

    print(f"\nall {len(results)} ablation runs complete: {results}")
    return results

import csv

def load_maestro_splits(csv_path):
  
    splits = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # csv has "2018/MIDI-....midi", the piece ids are "2018_MIDI-....midi"
            piece_id = row["midi_filename"].replace("/", "_")
            splits[piece_id] = row["split"]
    return splits


def train_on_split_only(model, patched_block, tokenizer, optimizer, precomputed_dir, splits, target_split="train", max_examples=300):
    
    with open(os.path.join(precomputed_dir, "index.json")) as f:
        piece_index = json.load(f)

    model.train()
    count = 0
    total_loss = 0.0

    for entry in piece_index:
        if count >= max_examples:
            break

        piece_id = entry["piece_id"]
        if splits.get(piece_id) != target_split:
            continue

        with open(os.path.join(precomputed_dir, entry["file"])) as f:
            examples = json.load(f)

        for example in examples:
            if count >= max_examples:
                break

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

            output = model(input_ids)
            logits = output[0]
            loss = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                input_ids[:, 1:].reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            count += 1

            if count % 20 == 0:
                print(f"train (split-filtered) example {count}: loss={loss.item():.4f}")

    print(f"trained on {count} examples from '{target_split}' split, avg loss={total_loss/max(count,1):.4f}")


def eval_on_split_only(model, patched_block, tokenizer, precomputed_dir, splits, target_split="validation", max_examples=100, use_motifs=True):
    """eval only (no training) on one maestro split"""
    patched_block.motif_padding_mask = None  # can be left over from batched training

    with open(os.path.join(precomputed_dir, "index.json")) as f:
        piece_index = json.load(f)

    model.eval()
    total_loss = 0.0
    count = 0

    for entry in piece_index:
        if count >= max_examples:
            break

        piece_id = entry["piece_id"]
        if splits.get(piece_id) != target_split:
            continue

        with open(os.path.join(precomputed_dir, entry["file"])) as f:
            examples = json.load(f)

        for example in examples:
            if count >= max_examples:
                break

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
    return count, avg_loss, math.exp(avg_loss)



import csv

def load_maestro_splits(csv_path):
    """maps piece_id -> split (train / validation / test) using the maestro csv"""
    splits = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # csv has "2018/MIDI-....midi", the piece ids are "2018_MIDI-....midi"
            piece_id = row["midi_filename"].replace("/", "_")
            splits[piece_id] = row["split"]
    return splits

def group_index_by_split(precomputed_dir, splits):

    with open(os.path.join(precomputed_dir, "index.json")) as f:
        piece_index = json.load(f)

    grouped = {"train": [], "validation": [], "test": []}
    unmatched = 0

    for entry in piece_index:
        split = splits.get(entry["piece_id"])
        if split in grouped:
            grouped[split].append(entry)
        else:
            unmatched += 1

    print(f"grouped pieces: train={len(grouped['train'])}, "
          f"validation={len(grouped['validation'])}, test={len(grouped['test'])}, "
          f"unmatched={unmatched}")

    return grouped

def train_with_epochs(model, patched_block, tokenizer, precomputed_dir, optimizer, splits, num_epochs=3, checkpoint_path="checkpoint.pt", use_wandb=False, wandb_project="motif-memory", wandb_run_name=None, batch_size=4):
    grouped = group_index_by_split(precomputed_dir, splits)
    train_pieces = grouped["train"]

    # load every train example up front so batches can mix pieces
    all_train_examples = []
    for entry in train_pieces:
        with open(os.path.join(precomputed_dir, entry["file"])) as f:
            all_train_examples.extend(json.load(f))

    total_steps = (len(all_train_examples) // batch_size) * num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    if use_wandb:
        wandb.init(project=wandb_project, name=wandb_run_name)

    global_step = 0

    # best val loss so far, to know which epoch to actually use
    best_val_loss = float("inf")
    best_epoch = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_count = 0

        for i in range(0, len(all_train_examples), batch_size):
            examples_batch = all_train_examples[i:i + batch_size]

            batch = build_batch(examples_batch, tokenizer, model)
            if batch is None:
                continue
            input_ids, attention_mask, motif_vectors, motif_padding_mask = batch

            patched_block.motif_vectors = motif_vectors
            patched_block.motif_padding_mask = motif_padding_mask

            output = model(input_ids, attention_mask=attention_mask)
            logits = output[0]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=2,  # pad token
            )

            optimizer.zero_grad()
            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=float("inf")
            )
            optimizer.step()
            scheduler.step()

            global_step += 1
            epoch_loss += loss.item()
            epoch_count += 1

            gate_val = torch.sigmoid(patched_block.motif_attn.gate).item()

            if use_wandb:
                wandb.log({
                    "loss": loss.item(),
                    "gate": gate_val,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "global_grad_norm": total_norm.item(),
                    "epoch": epoch,
                    "step": global_step,
                })

        train_avg_loss = epoch_loss / max(epoch_count, 1)
        print(f"epoch {epoch} done: {epoch_count} batches ({epoch_count * batch_size} examples, batch_size={batch_size}), avg loss = {train_avg_loss:.4f}")

        val_count, val_loss, val_ppl = eval_on_split_only(
            model, patched_block, tokenizer, precomputed_dir, splits,
            target_split="validation", max_examples=30, use_motifs=True
        )
        print(f"epoch {epoch} validation: {val_count} examples, loss = {val_loss:.4f}, perplexity = {val_ppl:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "step": global_step,
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
            })

        # one file per epoch. this used to overwrite the same file so only the last epoch survived
        epoch_checkpoint_path = checkpoint_path.replace(".pt", f"_epoch{epoch}.pt")
        torch.save(model.state_dict(), epoch_checkpoint_path)
        print(f"  -> checkpoint saved to {epoch_checkpoint_path}")

    print(f"\nbest epoch by validation loss: epoch {best_epoch} (val_loss={best_val_loss:.4f})")

    if use_wandb:
        wandb.finish()

def build_batch(examples_batch, tokenizer, model, pad_token_id=2, max_motifs=3, max_seq_len=512):
    """pad a list of examples into one batch: token ids, attention mask, motif vectors"""
    all_token_ids = []
    all_motif_vecs = []

    for example in examples_batch:
        motif_vecs = []
        for pattern_str, positions in list(example["motifs"].items())[:max_motifs]:
            pos = positions[0]
            motif_notes = example["context"][pos:pos + 7]
            if len(motif_notes) < 2:
                continue
            motif_vecs.append(motif_notes_to_vector(motif_notes, tokenizer, model))

        full_notes = example["context"] + example["target"]
        try:
            token_ids = notes_to_token_ids(full_notes, tokenizer)[0, :max_seq_len]
        except Exception:
            continue

        if token_ids.shape[0] < 2 or not motif_vecs:
            continue

        all_token_ids.append(token_ids)
        all_motif_vecs.append(torch.stack(motif_vecs))

    if not all_token_ids:
        return None

    # pad to the longest sequence in the batch
    max_len = max(t.shape[0] for t in all_token_ids)
    padded_ids = torch.full((len(all_token_ids), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(all_token_ids), max_len), dtype=torch.long)
    for i, t in enumerate(all_token_ids):
        padded_ids[i, :t.shape[0]] = t
        attention_mask[i, :t.shape[0]] = 1

    # same for the motifs
    max_motif_count = max(m.shape[0] for m in all_motif_vecs)
    hidden_dim = all_motif_vecs[0].shape[1]
    padded_motifs = torch.zeros((len(all_motif_vecs), max_motif_count, hidden_dim))
    motif_padding_mask = torch.ones((len(all_motif_vecs), max_motif_count), dtype=torch.bool)
    for i, m in enumerate(all_motif_vecs):
        padded_motifs[i, :m.shape[0]] = m
        motif_padding_mask[i, :m.shape[0]] = False  # True = padding

    return padded_ids, attention_mask, padded_motifs, motif_padding_mask


def select_fixed_examples(precomputed_dir, splits, n=10, seed=42):
    """same n examples from each split every time (seeded), so runs are comparable"""
    import random
    rng = random.Random(seed)

    grouped = group_index_by_split(precomputed_dir, splits)
    fixed = {}

    for split_name in ["train", "validation", "test"]:
        pieces = grouped[split_name]
        chosen_pieces = rng.sample(pieces, min(n, len(pieces)))

        examples_for_split = []
        for entry in chosen_pieces:
            with open(os.path.join(precomputed_dir, entry["file"])) as f:
                piece_examples = json.load(f)
            if piece_examples:
                # first example of each piece
                examples_for_split.append({
                    "piece_id": entry["piece_id"],
                    "example": piece_examples[0],
                })

        fixed[split_name] = examples_for_split
        print(f"{split_name}: selected {len(examples_for_split)} fixed examples")

    return fixed


def plot_piano_roll(notes, title, save_path):
    """piano roll plot saved to save_path. only start/end/pitch are used so 3 or 4 element notes both work"""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))
    for note in notes:
        start, end, pitch = note[0], note[1], note[2]
        ax.plot([start, end], [pitch, pitch], linewidth=4, color="steelblue")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("pitch (MIDI note number)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)
    print(f"saved piano roll to {save_path}")


def visualize_example(example_entry, model, tokenizer, patched_block, output_dir, label):
    """piano roll + wav of the ground truth and of a generated continuation, for one example"""
    import pretty_midi
    import tempfile

    os.makedirs(output_dir, exist_ok=True)
    example = example_entry["example"]
    piece_id = example_entry["piece_id"]

    # ground truth = melody context + polyphonic target
    ground_truth_notes = example["context"] + example["target"]
    plot_piano_roll(ground_truth_notes, f"{label} ground truth: {piece_id}",
                     os.path.join(output_dir, f"{label}_{piece_id}_ground_truth.png"))

    # motifs from the context
    motif_vecs = []
    for pattern_str, positions in list(example["motifs"].items())[:3]:
        pos = positions[0]
        motif_notes = example["context"][pos:pos + 7]
        if len(motif_notes) < 2:
            continue
        motif_vecs.append(motif_notes_to_vector(motif_notes, tokenizer, model))

    # prompt is just the context, capped at 512 like in training (some blow past aria's 8192 limit)
    try:
        context_ids = notes_to_token_ids(example["context"], tokenizer)[:, :512]
    except Exception as e:
        print(f"skipping {piece_id}: tokenization failed ({e})")
        return

    if context_ids.shape[1] < 2:
        print(f"skipping {piece_id}: context too short after truncation")
        return

    if motif_vecs:
        patched_block.motif_vectors = torch.stack(motif_vecs).unsqueeze(0)
    else:
        patched_block.motif_vectors = None
    patched_block.motif_padding_mask = None

    model.eval()
    with torch.no_grad():
        generated = model.generate(context_ids, max_new_tokens=100, do_sample=True, temperature=0.9)

    # decode gives a mido file, so go through a temp file to get pretty_midi
    midi_dict = tokenizer.decode(generated[0].tolist())
    mido_midi = midi_dict.to_midi()

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = tmp.name
    mido_midi.save(tmp_path)

    try:
        generated_midi = pretty_midi.PrettyMIDI(tmp_path)
    finally:
        os.remove(tmp_path)

    generated_notes = [
        [note.start, note.end, note.pitch]
        for instrument in generated_midi.instruments
        for note in instrument.notes
    ]
    plot_piano_roll(generated_notes, f"{label} generated: {piece_id}",
                     os.path.join(output_dir, f"{label}_{piece_id}_generated.png"))

    # synthesize() is just sine waves but doesn't need a soundfont
    import soundfile as sf

    def notes_to_midi_obj(notes):
        midi = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(program=0)
        for note in notes:
            start, end, pitch, velocity = note_fields(note)
            inst.notes.append(pretty_midi.Note(velocity=velocity, pitch=int(pitch), start=start, end=end))
        midi.instruments.append(inst)
        return midi

    gt_audio = notes_to_midi_obj(ground_truth_notes).synthesize()
    gen_audio = generated_midi.synthesize()

    sf.write(os.path.join(output_dir, f"{label}_{piece_id}_ground_truth.wav"), gt_audio, 44100)
    sf.write(os.path.join(output_dir, f"{label}_{piece_id}_generated.wav"), gen_audio, 44100)
    print(f"saved audio for {piece_id}")


def generate_with_motif_switching(model, tokenizer, patched_block, seed_notes, motif_sequence, tokens_per_segment=50):
    """
    generate in segments and swap the motif between them, to hear if the output reacts.
    motif_sequence is a list of (start, end, pitch) note lists, segment N uses motif N.
    """
    current_ids = notes_to_token_ids(seed_notes, tokenizer)[:, :512]

    model.eval()
    for i, motif_notes in enumerate(motif_sequence):
        motif_vector = motif_notes_to_vector(motif_notes, tokenizer, model)
        patched_block.motif_vectors = motif_vector.unsqueeze(0).unsqueeze(0)
        patched_block.motif_padding_mask = None

        with torch.no_grad():
            current_ids = model.generate(
                current_ids,
                max_new_tokens=tokens_per_segment,
                do_sample=True,
                temperature=0.9,
            )
        print(f"segment {i+1}: generated {tokens_per_segment} tokens using motif {i+1}")

    return current_ids


def test_gate_forcing(model, patched_block, tokenizer, seed_notes, motif_notes, gate_override):
    """
    force the gate to a value and return the logits, to check the cross attention
    changes the output no matter what gate value got learned.
    gate_override is pre-sigmoid, so -10 is basically off and +10 basically on.
    compare the logits from two overrides.
    """
    with torch.no_grad():
        patched_block.motif_attn.gate.fill_(gate_override)

    motif_vector = motif_notes_to_vector(motif_notes, tokenizer, model)
    patched_block.motif_vectors = motif_vector.unsqueeze(0).unsqueeze(0)
    patched_block.motif_padding_mask = None

    context_ids = notes_to_token_ids(seed_notes, tokenizer)[:, :512]

    model.eval()
    with torch.no_grad():
        output = model(context_ids)
        logits = output[0]

    return logits


def build_batch_token_concatenated(examples, pad_token_id=2, max_seq_len=1024):
    """
    pads build_token_concatenated_example outputs (doingstuff.py) into a batch.
    labels are -100 before each example's target_start_index so the loss only
    counts the target tokens.
    """
    IGNORE_INDEX = -100

    sequences = []
    target_starts = []
    for ex in examples:
        if ex is None:
            continue
        seq = ex["input_ids"][:max_seq_len]
        sequences.append(seq)
        target_starts.append(min(ex["target_start_index"], seq.shape[0]))

    if not sequences:
        return None

    max_len = max(s.shape[0] for s in sequences)
    input_ids = torch.full((len(sequences), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
    labels = torch.full((len(sequences), max_len), IGNORE_INDEX, dtype=torch.long)

    for i, seq in enumerate(sequences):
        input_ids[i, :seq.shape[0]] = seq
        attention_mask[i, :seq.shape[0]] = 1
        labels[i, target_starts[i]:seq.shape[0]] = seq[target_starts[i]:]

    return input_ids, attention_mask, labels


def train_token_concatenated(model, tokenizer, precomputed_dir_or_pieces, optimizer, num_epochs=1, batch_size=4, use_wandb=False, wandb_project="motif-memory", wandb_run_name=None, checkpoint_path="checkpoint_tokens.pt"):
    """
    training loop for the token concatenated version. no patched block or motif
    vectors here, model is just a plain AutoModelForCausalLM.

    for now precomputed_dir_or_pieces is a list of (raw_notes, melody, motifs)
    tuples. no precompute pipeline for this format yet.
    """
    from doingstuff import build_token_concatenated_example
    import time

    if use_wandb:
        wandb.init(project=wandb_project, name=wandb_run_name)

    IGNORE_INDEX = -100
    global_step = 0
    model.train()

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        epoch_count = 0

        examples_batch = []
        for raw_notes, melody, motifs in precomputed_dir_or_pieces:
            ex = build_token_concatenated_example(raw_notes, melody, motifs, tokenizer)
            if ex is None:
                continue
            examples_batch.append(ex)

            if len(examples_batch) < batch_size:
                continue

            batch = build_batch_token_concatenated(examples_batch)
            examples_batch = []
            if batch is None:
                continue
            input_ids, attention_mask, labels = batch

            output = model(input_ids, attention_mask=attention_mask)
            logits = output[0]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
                ignore_index=IGNORE_INDEX,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            global_step += 1
            epoch_loss += loss.item()
            epoch_count += 1

            if use_wandb:
                wandb.log({"loss": loss.item(), "epoch": epoch, "step": global_step})

            if global_step % 10 == 0:
                print(f"step {global_step}: loss = {loss.item():.4f}")

        print(f"epoch {epoch} done: {epoch_count} batches, avg loss = {epoch_loss / max(epoch_count, 1):.4f}")

    torch.save(model.state_dict(), checkpoint_path)
    print(f"saved checkpoint to {checkpoint_path}")

    if use_wandb:
        wandb.finish()