"""Parse the SplendidGrandPiano SFZ region files into a reference-sample map.

The master .sfz maps velocity ranges to dynamic-layer include files:
PP=41-67, MP=68-84, MF=85-100, FF=101-127. (Velocities 1-40 reuse the PP
samples behind a 1 kHz lowpass — a synthetic layer, so we don't score
against it.) Each layer file lists regions with an exact pitch_keycenter,
which is the only place we trust for note numbers.
"""

import re
from dataclasses import dataclass
from pathlib import Path

# layer name -> (lovel, hivel, representative velocity used when rendering)
LAYERS = {
    "PP": (41, 67, 54),
    "MP": (68, 84, 76),
    "MF": (85, 100, 92),
    "FF": (101, 127, 114),
}


@dataclass(frozen=True)
class ReferenceSample:
    note: int          # MIDI note (pitch_keycenter)
    layer: str         # PP / MP / MF / FF
    velocity: int      # representative MIDI velocity for this layer
    path: Path         # FLAC file


def load_reference_map(repo: Path) -> list[ReferenceSample]:
    """Return all (note, layer) reference samples, sorted by note then layer."""
    samples = []
    for layer in LAYERS:
        text = (repo / "Data" / f"{layer}.txt").read_text()
        for line in text.splitlines():
            if "<region>" not in line:
                continue
            keycenter = re.search(r"pitch_keycenter=(\d+)", line)
            sample = re.search(r"sample=(.+?\.)\$EXT", line)
            if not keycenter or not sample:
                continue
            path = repo / "Samples" / (sample.group(1) + "flac")
            if not path.exists():
                raise FileNotFoundError(path)
            samples.append(
                ReferenceSample(
                    note=int(keycenter.group(1)),
                    layer=layer,
                    velocity=LAYERS[layer][2],
                    path=path,
                )
            )
    samples.sort(key=lambda s: (s.note, s.layer))
    return samples
