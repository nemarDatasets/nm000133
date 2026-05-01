"""
Align nm000133 (Alljoined1) events.tsv rows with their stimulus images.

The dataset's events.tsv uses ``trial_type = "image/N"`` where N is a
1-indexed image position (1..960) within the *shared* NSD subset that
all 8 subjects view (the first 960 rows of NSD's ``shared1000``).

The mapping chain is:

    events.tsv `value` (or N in `image/N`)            (1..960, 1-indexed)
        ↓
    `sharedix[0]` from code/0_data_collection/nsd_expdesign.mat
        (1-indexed NSD ids)
        ↓
    `nsdId = sharedix[N-1] - 1`                       (0-indexed NSD id)
        ↓
    code/1_preprocessing/data/nsd_stim_info_merged.csv
        ↓
    `cocoId`, `cocoSplit` (val2017 / train2017)
        ↓
    stimulus path: stimuli/<cocoSplit>/000000<cocoId:012d>.jpg
        (preserves original COCO 2017 directory layout)

Empirically, all 960 sharedix images for this dataset map to train2017
COCO images (none in val2017).

Usage
-----
    aligner = StimulusAligner(root='/path/to/nm000133')
    paths = aligner.paths_for_events(events_df)         # list[Path | None]
    img   = aligner.image_for_event(row, mode='PIL')    # single row
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Iterable

import pandas as pd

try:
    from PIL import Image as _PIL_Image
except ImportError:
    _PIL_Image = None


def _coco_filename(coco_id: int, coco_split: str) -> str:
    return f'{coco_split}/{int(coco_id):012d}.jpg'


class StimulusAligner:
    def __init__(
        self,
        root: str | Path,
        nsd_expdesign_mat: str | Path | None = None,
        nsd_stim_info_csv: str | Path | None = None,
    ) -> None:
        self.root = Path(root)
        self.stim_root = self.root / 'stimuli'

        mat_p = Path(nsd_expdesign_mat) if nsd_expdesign_mat else \
            self.root / 'code' / '0_data_collection' / 'nsd_expdesign.mat'
        csv_p = Path(nsd_stim_info_csv) if nsd_stim_info_csv else \
            self.root / 'code' / '1_preprocessing' / 'data' / 'nsd_stim_info_merged.csv'

        from scipy.io import loadmat
        mat = loadmat(str(mat_p))
        self._sharedix = mat['sharedix'][0]   # shape (1000,), 1-indexed NSD ids
        self._subjectim = mat['subjectim']    # shape (8, 10000), 1-indexed NSD ids

        df = pd.read_csv(csv_p, index_col=0)
        # NSD id (0-indexed) → (cocoId, cocoSplit)
        self._nsd_to_coco = {
            int(r['nsdId']): (int(r['cocoId']), str(r['cocoSplit']))
            for _, r in df[['nsdId', 'cocoId', 'cocoSplit']].iterrows()
        }

    # ---- per-row resolution ----

    def path_for_event(self, row, subject: int = 1, session: int = 1) -> Optional[Path]:
        """Return the stimulus image path for a single events.tsv row.

        Parameters
        ----------
        row
            A pandas Series-like with a ``value`` (or ``trial_type='image/N'``) field.
        subject, session
            1-indexed subject / session numbers. Used only when the dataset
            ever ships subject-specific (non-shared) images. For the current
            shared-960 release this argument is irrelevant: every odd-session
            slot in `subjectim[s][:1000]` is identical to `sharedix[0]`.
        """
        n = self._extract_position(row)
        if n is None:
            return None
        if n < 1 or n > 1000:
            return None  # out of shared range; would need subjectim slice
        nsd_1based = int(self._sharedix[n - 1])
        nsd_0based = nsd_1based - 1
        ent = self._nsd_to_coco.get(nsd_0based)
        if ent is None:
            return None
        coco_id, coco_split = ent
        return self.stim_root / _coco_filename(coco_id, coco_split)

    def image_for_event(self, row, mode: str = 'PIL', **kw):
        p = self.path_for_event(row, **kw)
        if p is None:
            return None
        if mode == 'path':
            return p
        if mode == 'bytes':
            return p.read_bytes()
        if _PIL_Image is None:
            raise RuntimeError("Pillow is not installed; use mode='path' or 'bytes'.")
        return _PIL_Image.open(p)

    @staticmethod
    def _extract_position(row) -> Optional[int]:
        """Pull the 1-indexed position N out of an events.tsv row.

        Prefers the ``value`` column; falls back to parsing
        ``trial_type='image/N'``. Returns None if neither yields an int.
        """
        v = row.get('value') if hasattr(row, 'get') else getattr(row, 'value', None)
        if v is not None and not (isinstance(v, float) and pd.isna(v)):
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
        t = row.get('trial_type') if hasattr(row, 'get') else getattr(row, 'trial_type', None)
        if t is None or (isinstance(t, float) and pd.isna(t)):
            return None
        s = str(t)
        if '/' in s:
            tail = s.split('/', 1)[1]
            try:
                return int(tail)
            except ValueError:
                return None
        return None

    # ---- vectorised ----

    def paths_for_events(
        self, events: pd.DataFrame, subject: int = 1, session: int = 1
    ) -> list[Optional[Path]]:
        out: list[Optional[Path]] = []
        for _, row in events.iterrows():
            out.append(self.path_for_event(row, subject=subject, session=session))
        return out

    # ---- helpers ----

    def needed_coco_files(self) -> Iterable[tuple[int, str, str]]:
        """Yield (coco_id, coco_split, target_filename) triples for the 960
        shared-subset stimuli this dataset references. Useful for a targeted
        downloader (only fetch what the BIDS data uses)."""
        for n in range(1, 961):
            nsd_0 = int(self._sharedix[n - 1]) - 1
            ent = self._nsd_to_coco.get(nsd_0)
            if ent is not None:
                cid, csp = ent
                yield cid, csp, _coco_filename(cid, csp)


def demo(
    root: str = '/data/tau/iceberg_1/titanic_1/datasets/bids/nm000133',
    subject: str = '01',
    session: str = '01',
) -> None:
    root_p = Path(root)
    aligner = StimulusAligner(root_p)
    ev = root_p / f'sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-images_events.tsv'
    df = pd.read_csv(ev, sep='\t')
    paths = aligner.paths_for_events(df, subject=int(subject), session=int(session))
    n_total = len(df)
    n_resolved = sum(1 for p in paths if p is not None)
    n_exists = sum(1 for p in paths if p is not None and p.exists())
    print(f'== sub-{subject} ses-{session} ==')
    print(f'   rows           : {n_total}')
    print(f'   resolved paths : {n_resolved}')
    print(f'   resolved+exists: {n_exists}')


if __name__ == '__main__':
    demo()
