"""Crop the COCO originals in stimuli/train2017/ to the NSD images participants saw.

NSD shows each COCO image cropped to a square by its ``cropBox`` (top, bottom,
left, right fractions, in code/1_preprocessing/data/nsd_stim_info_merged.csv)
and resized to 425x425. The crops are written as ``stimuli/nsd/<nsdId>.png``,
NSD's 0-based 73k index, and match NSD's ``nsd_stimuli.hdf5`` to within a few
intensity levels (JPEG decoding and resampling).

Usage: python code/crop_stimuli.py <dataset root>
"""

import ast
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

root = Path(sys.argv[1])
info = pd.read_csv(
    root / "code/1_preprocessing/data/nsd_stim_info_merged.csv", index_col=0
).set_index("cocoId")
out = root / "stimuli" / "nsd"
out.mkdir(exist_ok=True)
for src in sorted((root / "stimuli" / "train2017").glob("*.jpg")):
    row = info.loc[int(src.stem)]
    top, bottom, left, right = ast.literal_eval(row["cropBox"])
    img = Image.open(src).convert("RGB")
    w, h = img.size
    box = (round(left * w), round(top * h), round(w - right * w), round(h - bottom * h))
    img.crop(box).resize((425, 425), Image.BICUBIC).save(out / f"{row['nsdId']}.png")
