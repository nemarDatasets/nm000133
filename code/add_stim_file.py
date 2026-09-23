"""Write the BIDS ``stim_file`` column of every *_events.tsv and describe the columns.

``value`` (the N of ``trial_type = image/N``) is the 1-based position of the
image in NSD's shared1000 list, so the NSD image shown is
``sharedix[value - 1]`` (1-based NSD id, from
code/0_data_collection/nsd_expdesign.mat) and ``stim_file`` is
``nsd/<nsdId>.png`` with NSD's 0-based id: the crop code/crop_stimuli.py
writes. For sub-02 ses-01 this reproduces the NSD ids of all 3036 trials in
the original release (osf.io/kqgs8, 05_125/subj02_session1.h5).

Usage: python code/add_stim_file.py <dataset root> [nsd_expdesign.mat]
"""

import json
import sys
from pathlib import Path

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
    "stim_file": {
        "Description": (
            "NSD image shown at this onset, relative to stimuli/: the COCO image "
            "cropped by NSD's cropBox and resized to 425x425 (code/crop_stimuli.py)."
        )
    },
}


def add_stim_file(events: Path, sharedix, stimuli: Path) -> None:
    header, *lines = events.read_text(encoding="utf-8").splitlines()
    columns = "onset\tduration\ttrial_type\tvalue\tsample"
    assert header in (columns, columns + "\tstim_file"), events
    out = [columns + "\tstim_file"]
    for line in lines:
        onset, duration, trial_type, value, sample = line.split("\t")[:5]
        assert trial_type == f"image/{value}", (events.name, line)
        path = f"nsd/{int(sharedix[int(value) - 1]) - 1}.png"
        assert (stimuli / path).exists(), (events.name, path)
        out.append("\t".join([onset, duration, trial_type, value, sample, path]))
    events.write_text("\n".join(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    root = Path(sys.argv[1])
    mat = Path(sys.argv[2]) if len(sys.argv) > 2 else root / "code/0_data_collection/nsd_expdesign.mat"
    sharedix = loadmat(mat)["sharedix"][0]
    files = sorted(root.glob("sub-*/ses-*/eeg/*_events.tsv"))
    for events in files:
        add_stim_file(events, sharedix, root / "stimuli")
    (root / "task-images_events.json").write_text(json.dumps(EVENTS_JSON, indent=2) + "\n")
    print(f"wrote stim_file for {len(files)} events files")
