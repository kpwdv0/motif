import csv
import torch
from transformers import AutoModelForCausalLM
import torch.nn as nn
import math
from torch.optim.lr_scheduler import CosineAnnealingLR
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

    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)

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


    with torch.no_grad():
        token_embeds = model.model.tok_embeddings(token_ids.input_ids)
        pooled = token_embeds.mean(dim=1)

    return pooled.squeeze(0)

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
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    offset = notes[0][0]
    for start, end, pitch in notes:
        instrument.notes.append(pretty_midi.Note(
            velocity=80, pitch=int(pitch), start=start - offset, end=end - offset
        ))
    midi.instruments.append(instrument)

    with tempfile.NamedTemporaryFile(suffix=".midi", delete=False) as tmp:
        tmp_path = tmp.name
    midi.write(tmp_path)
    try:
        result = tokenizer.encode_from_file(tmp_path, return_tensors="pt")
    finally:
        os.remove(tmp_path)
    return result.input_ids

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
    prompt = tokenizer.encode_from_file(midi_path, return_tensors="pt")
    input_ids = prompt.input_ids[..., :100]

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
    patched_block.motif_padding_mask = None

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
    splits = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
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

    all_train_examples = []
    for entry in train_pieces:
        with open(os.path.join(precomputed_dir, entry["file"])) as f:
            all_train_examples.extend(json.load(f))

    total_steps = (len(all_train_examples) // batch_size) * num_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1))

    if use_wandb:
        wandb.init(project=wandb_project, name=wandb_run_name)

    global_step = 0

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
                ignore_index=2,
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

        if use_wandb:
            wandb.log({
                "epoch": epoch,
                "step": global_step,
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
            })

        torch.save(model.state_dict(), checkpoint_path)
        print(f"  -> checkpoint saved after epoch {epoch}")

    if use_wandb:
        wandb.finish()

def build_batch(examples_batch, tokenizer, model, pad_token_id=2, max_motifs=3, max_seq_len=512):
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

    max_len = max(t.shape[0] for t in all_token_ids)
    padded_ids = torch.full((len(all_token_ids), max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(all_token_ids), max_len), dtype=torch.long)
    for i, t in enumerate(all_token_ids):
        padded_ids[i, :t.shape[0]] = t
        attention_mask[i, :t.shape[0]] = 1

    max_motif_count = max(m.shape[0] for m in all_motif_vecs)
    hidden_dim = all_motif_vecs[0].shape[1]
    padded_motifs = torch.zeros((len(all_motif_vecs), max_motif_count, hidden_dim))
    motif_padding_mask = torch.ones((len(all_motif_vecs), max_motif_count), dtype=torch.bool)
    for i, m in enumerate(all_motif_vecs):
        padded_motifs[i, :m.shape[0]] = m
        motif_padding_mask[i, :m.shape[0]] = False

    return padded_ids, attention_mask, padded_motifs, motif_padding_mask
