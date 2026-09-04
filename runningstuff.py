from doingstuff import loadnotes, skyline_melody, find_motifs, build_training_examples

path = r"C:\Users\kira\data\maestro-v3.0.0\2018\MIDI-Unprocessed_Recital1-3_MID--AUDIO_01_R1_2018_wav--1.midi"
notes = loadnotes(path)
melody = skyline_melody(notes)
motifs = find_motifs(melody, window_size=6)

examples = build_training_examples(melody, motifs, num_cuts=4)
print(len(examples), "examples generated")

for ex in examples:
    print(f"cut at note {ex['cut_index']}: {len(ex['context'])} context notes, {len(ex['target'])} target notes, {len(ex['motifs'])} motifs in context")
