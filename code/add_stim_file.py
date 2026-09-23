"""Add a BIDS ``stim_file`` column to every *_events.tsv and describe the columns.

``value`` (the N of ``trial_type = image/N``) is the 1-based position of the
image in NSD's shared1000 list. The image is resolved as code/align_stimuli.py
documents: value -> ``sharedix[value - 1]`` (1-based NSD id, from
code/0_data_collection/nsd_expdesign.mat) -> cocoId and cocoSplit (from
code/1_preprocessing/data/nsd_stim_info_merged.csv) ->
``stimuli/<cocoSplit>/<cocoId:012d>.jpg``. For sub-02 ses-01 this reproduces the
NSD and COCO ids of all 3036 trials in the original release (osf.io/kqgs8,
05_125/subj02_session1.h5).

Usage: python code/add_stim_file.py <dataset root> [nsd_expdesign.mat] [nsd_stim_info_merged.csv]
"""

import json
import sys
from pathlib import Path

import pandas as pd
from scipy.io import loadmat

EVENTS_JSON = {
    "onset": {"Description": "Onset of the image, from the first data point.", "Units": "s"},
    "duration": {"Description": "0: the event marks the image onset.", "Units": "s"},
    "trial_type": {"Description": "Image presentation, as 'image/<value>'."},
    "value": {
        "Description": (
            "1-based position of the image in NSD's shared1000 list (sharedix in "
            "code/0_data_collection/nsd_expdesign.mat); stim_file names the image."
        )
    },
    "sample": {"Description": "Onset in samples, from the first data point (0)."},
    "stim_file": {"Description": "Image shown at this onset, relative to stimuli/."},
}


def image_paths(root: Path, mat: Path, csv: Path) -> dict[int, str]:
    sharedix = loadmat(mat)["sharedix"][0]
    info = pd.read_csv(csv, index_col=0).set_index("nsdId")
    paths = {}
    for value, nsd_id in enumerate(sharedix, start=1):
        coco_id, split = info.loc[int(nsd_id) - 1, ["cocoId", "cocoSplit"]]
        paths[value] = f"{split}/{int(coco_id):012d}.jpg"
    return paths


def add_stim_file(events: Path, paths: dict[int, str], stimuli: Path) -> None:
    header, *lines = events.read_text(encoding="utf-8").splitlines()
    assert header == "onset\tduration\ttrial_type\tvalue\tsample", events
    out = [header + "\tstim_file"]
    for line in lines:
        _, _, trial_type, value, _ = line.split("\t")
        assert trial_type == f"image/{value}", (events.name, line)
        path = paths[int(value)]
        assert (stimuli / path).exists(), (events.name, path)
        out.append(f"{line}\t{path}")
    events.write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    mat = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "code/0_data_collection/nsd_expdesign.mat"
    csv = Path(sys.argv[3]) if len(sys.argv) > 3 else root / "code/1_preprocessing/data/nsd_stim_info_merged.csv"
    paths = image_paths(root, mat, csv)
    files = sorted(root.glob("sub-*/ses-*/eeg/*_events.tsv"))
    for events in files:
        add_stim_file(events, paths, root / "stimuli")
    (root / "task-images_events.json").write_text(json.dumps(EVENTS_JSON, indent=2) + "\n")
    print(f"added stim_file to {len(files)} events files")
