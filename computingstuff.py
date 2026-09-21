import os
import json
import time
from doingstuff import loadnotes, skyline_melody, find_motifs, build_training_examples_polyphonic

def clean_for_json(notes):
    """json.dump chokes on numpy floats, so turn every note into a plain list of python numbers"""
    cleaned = []
    for note in notes:
        cleaned.append([float(note[0]), float(note[1]), int(note[2])] +
                        ([int(note[3])] if len(note) > 3 else []))
    return cleaned


def precompute_piece(path):
    """context = melody up to the cut, target = all the raw notes after it"""
    notes = loadnotes(path)
    melody = skyline_melody(notes)
    motifs = find_motifs(melody, window_size=6)
    examples = build_training_examples_polyphonic(notes, melody, motifs, num_cuts=4)

    if not examples:
        return None

    # tuple keys don't work in json so they get stringified
    cleaned_examples = []
    for ex in examples:
        cleaned_examples.append({
            "motifs": {str(pattern): positions for pattern, positions in ex["motifs"].items()},
            "context": clean_for_json(ex["context"]),
            "target": clean_for_json(ex["target"]),
            "cut_time": float(ex["cut_time"]),
        })

    return cleaned_examples


def precompute_all(maestro_root, output_dir, years=None):
    os.makedirs(output_dir, exist_ok=True)

    if years is None:
        years = [d for d in os.listdir(maestro_root) if os.path.isdir(os.path.join(maestro_root, d))]

    index = []
    errors = []
    start_time = time.time()

    for year in years:
        year_path = os.path.join(maestro_root, year)
        if not os.path.isdir(year_path):
            continue

        midi_files = [f for f in os.listdir(year_path) if f.endswith(".midi")]

        for filename in midi_files:
            path = os.path.join(year_path, filename)
            piece_id = f"{year}_{filename}"  # filenames repeat across years

            try:
                examples = precompute_piece(path)
                if examples is None:
                    continue  # too short

                out_path = os.path.join(output_dir, f"{piece_id}.json")
                with open(out_path, "w") as f:
                    json.dump(examples, f)

                index.append({
                    "piece_id": piece_id,
                    "num_examples": len(examples),
                    "file": f"{piece_id}.json",
                })

            except Exception as e:
                errors.append({"piece_id": piece_id, "error": str(e)})

        print(f"finished year {year}: {len(index)} pieces so far, {len(errors)} errors so far")

    # index of what got written
    with open(os.path.join(output_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    elapsed = time.time() - start_time
    total_examples = sum(entry["num_examples"] for entry in index)

    print(f"\ndone in {elapsed:.1f}s")
    print(f"{len(index)} pieces succeeded, {len(errors)} failed, {total_examples} total training examples")

    if errors:
        with open(os.path.join(output_dir, "errors.json"), "w") as f:
            json.dump(errors, f, indent=2)
        print(f"error details saved to errors.json")

    return index, errors


if __name__ == "__main__":
    precompute_all(
        maestro_root=r"C:\Users\kira\data\maestro-v3.0.0",
        output_dir=r"C:\Users\kira\data\maestro-precomputed",
    )