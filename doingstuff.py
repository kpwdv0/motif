import pretty_midi

def loadnotes(midi_path):
    midi = pretty_midi.PrettyMIDI(midi_path)

    notes = []
    for instrument in midi.instruments:
        for note in instrument.notes:
            notes.append((note.start,note.end,note.pitch,note.velocity))

    notes.sort(key=lambda n: (n[0],-n[2])) #(start,pitch)
    return notes

def skyline_melody(notes, maxgap = 0.5):
    events = sorted(notes, key = lambda n: n[0]) #sort js in case idk

    melody = []
    active = []

    for start,end, pitch, velocity in events:
        active = [n for n in active if n[1] > start] 
        active.append((start,end,pitch))
        highest = max(active, key=lambda n: n[2])

        if highest[2] == pitch:
            if melody and melody[-1][1] > start:
                prev_start,_, prev_pitch = melody[-1]
                melody[-1] = (prev_start, start, prev_pitch)

            melody.append((start,end,pitch))

    filtered =[]
    for i, note in enumerate(melody):
        start = note[0]

        gapbef = start - melody[i-1][1] if i>0 else None
        gapaft = melody[i+1][0] - note[1] if i < len(melody) - 1 else None

        closebef = gapbef is not None and gapbef <= maxgap
        closeaft = gapaft is not None and gapaft <= maxgap 

        if closeaft or closebef:
            filtered.append(note)
    return filtered


r"""

tested it and got some notesss:

code:
from doingstuff import loadnotes, skyline_melody
path = r"C:\Users\kira\data\maestro-v3.0.0\2018\MIDI-Unprocessed_Recital1-3_MID--AUDIO_01_R1_2018_wav--1.midi"
notes = loadnotes(path)
melody = skyline_melody(notes)
print(len(notes), "raw notes ->", len(melody), "melody notes")
for n in melody[:20]:
    print(n)


basically takes the first stuff of the melody yippee

(np.float64(0.9713541666666666), np.float64(1.0911458333333333), 74)
(np.float64(1.7044270833333333), np.float64(1.7578125), 43)
(np.float64(1.7578125), np.float64(1.8138020833333333), 47)
(np.float64(1.8138020833333333), np.float64(1.875), 50)
(np.float64(1.875), np.float64(1.91015625), 55)
(np.float64(1.91015625), np.float64(2.0026041666666665), 76)
(np.float64(2.020833333333333), np.float64(2.091145833333333), 74)
(np.float64(2.102864583333333), np.float64(2.15625), 76)
(np.float64(2.1783854166666665), np.float64(2.231770833333333), 74)
(np.float64(2.231770833333333), np.float64(2.302083333333333), 76)
(np.float64(2.325520833333333), np.float64(2.778645833333333), 74)
(np.float64(2.9518229166666665), np.float64(3.059895833333333), 74)
(np.float64(3.063802083333333), np.float64(3.161458333333333), 76)
(np.float64(3.1783854166666665), np.float64(3.2252604166666665), 78)
(np.float64(3.24609375), np.float64(3.305989583333333), 76)
(np.float64(3.333333333333333), np.float64(3.3697916666666665), 78)
(np.float64(3.40234375), np.float64(3.44140625), 76)
(np.float64(3.453125), np.float64(3.505208333333333), 78)
(np.float64(3.5234375), np.float64(3.5885416666666665), 76)
(np.float64(3.60546875), np.float64(3.677083333333333), 74)

notes: looks pretty, good, they sit within the same range

but some of the notes are really really low (43) which isnt a melody sooo

maybe filter y pitch range or minimum active notes or smth? gonna return to this later (star star)
"""


def note_transitions(melody):
    signatures = []

    for prev_note, curr_note in zip(melody, melody[1:]):
        prev_start, prev_end, prev_pitch = prev_note
        curr_start, curr_end, curr_pitch = curr_note

        if curr_pitch > prev_pitch:
            contour = 1
        elif curr_pitch < prev_pitch:
            contour = -1
        else:
            contour = 0

        prev_dur = prev_end - prev_start
        curr_dur = curr_end - curr_start

        ratio = curr_dur / prev_dur if prev_dur > 0 else 1.0
        if ratio < 0.75:
            rhythm = -1
        elif ratio > 1.33:
            rhythm = 1
        else:
            rhythm = 0

        signatures.append((contour, rhythm))

    return signatures


from collections import defaultdict

def find_motifs(melody, window_size=4):

    signatures = note_transitions(melody)

    occurrences = defaultdict(list)

    for i in range(len(signatures) - window_size + 1):
        window = tuple(signatures[i:i + window_size])
        occurrences[window].append(i)

    motifs = {
        pattern: positions
    for pattern, positions in occurrences.items()
    if len(positions) >= 2 and not is_periodic(pattern)}

    return motifs
    
import random

def build_training_examples(melody, motifs, num_cuts=4, min_context=20, min_target=20):

    examples = []

    earliest_cut = min_context
    latest_cut = len(melody) - min_target

    if earliest_cut >= latest_cut:
        return examples

    for _ in range(num_cuts):
        cut = random.randint(earliest_cut, latest_cut)

        context = melody[:cut]
        target = melody[cut:]
        motifs_in_context = {}
        for pattern, positions in motifs.items():
            before_cut = [p for p in positions if p < cut]
            if before_cut:
                motifs_in_context[pattern] = before_cut

        examples.append({
            "motifs": motifs_in_context,
            "context": context,
            "target": target,
            "cut_index": cut,
        })

    return examples


def is_periodic(pattern):

    n = len(pattern)
    for period in range(1, n // 2 + 1):
        if n % period != 0:
            continue
        if all(pattern[i] == pattern[i - period] for i in range(period, n)):
            return True
    return False
r"""
ran testing stuff
tingstuff.py
230 repeated patterns found
((1, 0), (1, 0), (1, -1), (1, 1)) -> occurs at note indices [0, 369, 425, 1381]
((1, 0), (1, -1), (1, 1), (-1, 0)) -> occurs at note indices [1, 370]
((1, -1), (1, 1), (-1, 0), (1, 0)) -> occurs at note indices [2, 739]
((1, 1), (-1, 0), (1, 0), (-1, 0)) -> occurs at note indices [3, 451, 463, 494, 740, 904, 1038, 1044, 1057]
((-1, 0), (1, 0), (-1, 0), (1, 0)) -> occurs at note indices [4, 452, 454, 464, 466, 495, 601, 603, 605, 607, 651, 669, 759, 761, 793, 795, 797, 799, 905, 1026, 1031, 1033, 1039, 1058, 1091, 1093]
(motif) PS C:\Users\kira\motif> 

okay the dow up down up pattern looks mostly like trills instead of actual melodies??? maybe keep patterrs that only occur x times

increased winow size:

(motif) PS C:\Users\kira\motif> python testingstuff.py
132 repeated patterns found
2 occurrences -> ((1, -1), (-1, 0), (1, 1), (-1, 0), (-1, 0), (1, 0)) at [13, 1087]
2 occurrences -> ((1, 0), (1, 0), (1, 0), (1, 0), (1, 0), (1, 1)) at [30, 304]
2 occurrences -> ((1, 0), (1, 0), (1, 0), (1, 0), (1, 1), (-1, -1)) at [31, 305]
2 occurrences -> ((1, 1), (-1, -1), (1, 1), (-1, -1), (1, 0), (-1, 0)) at [35, 813]
2 occurrences -> ((-1, 0), (1, -1), (-1, 1), (1, 0), (-1, 0), (1, 0)) at [47, 1376]
2 occurrences -> ((1, -1), (-1, 1), (1, 0), (-1, 0), (1, 0), (1, 0)) at [48, 1377]
2 occurrences -> ((1, 0), (-1, 0), (1, 0), (1, 0), (-1, -1), (1, 1)) at [50, 894]
2 occurrences -> ((1, 1), (-1, -1), (1, 1), (-1, -1), (-1, -1), (1, 1)) at [78, 718]
2 occurrences -> ((-1, -1), (1, 1), (0, -1), (-1, 1), (-1, -1), (0, 1)) at [88, 545]
2 occurrences -> ((1, 0), (-1, 1), (-1, -1), (1, 0), (1, 1), (1, 1)) at [100, 780]
(motif) PS C:\Users\kira\motif> 

better! pattern count dropped so hopefully there arent a lot o trill patterns
132 repeated patterns found
11 occurrences -> ((-1, 0), (1, 0), (-1, 0), (1, 0), (-1, 0), (1, 0)) at [452, 464, 601, 603, 605, 759, 793, 795, 797, 1031, 1091]
10 occurrences -> ((1, 0), (-1, 0), (1, 0), (-1, 0), (1, 0), (-1, 0)) at [453, 602, 604, 758, 760, 792, 794, 796, 798, 1030]
6 occurrences -> ((1, 0), (1, 0), (1, 0), (-1, 0), (1, 0), (1, 0)) at [311, 1224, 1246, 1250, 1300, 1318]
5 occurrences -> ((-1, -1), (1, 0), (1, 1), (-1, -1), (1, 0), (1, 0)) at [265, 929, 1231, 1260, 1365]
5 occurrences -> ((1, 0), (1, 1), (-1, -1), (1, 0), (1, 1), (-1, -1)) at [409, 1229, 1255, 1258, 1363]
5 occurrences -> ((1, 1), (-1, -1), (1, 0), (1, 1), (-1, -1), (1, 0)) at [410, 1230, 1256, 1259, 1364]
5 occurrences -> ((-1, -1), (1, 1), (-1, 0), (1, 0), (-1, 0), (1, 0)) at [450, 462, 493, 903, 1037]
5 occurrences -> ((1, 0), (-1, 0), (1, 0), (-1, -1), (1, 1), (-1, 0)) at [909, 1034, 1040, 1059, 1148]
5 occurrences -> ((-1, 0), (1, 0), (-1, -1), (1, 1), (-1, 0), (1, 0)) at [910, 1035, 1041, 1060, 1149]
4 occurrences -> ((1, 1), (-1, -1), (1, 0), (1, 0), (-1, -1), (1, 1)) at [208, 267, 1240, 1262]

this time sorted by occurence count this is the trill thing again :((

ok so i added a check to check whether it was like periodic

ran it agian with this and it was more filtered out yppeeeeee
"""
