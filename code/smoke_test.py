"""
Smoke test for nm000133 (Alljoined1) stimulus alignment.

Walks every ``sub-*/ses-*/eeg/*_events.tsv``, resolves each row via
``StimulusAligner``, and reports per-subject/per-session totals plus
unresolved references.

Run AFTER ``code/download_stimuli.py`` has populated ``stimuli/``.
"""
from __future__ import annotations
import re, sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_stimuli import StimulusAligner

ROOT_DEFAULT = Path(__file__).resolve().parent.parent
PATH_RE = re.compile(r'sub-(\d+)/ses-(\d+)/')


def run(root: Path = ROOT_DEFAULT) -> int:
    aligner = StimulusAligner(root)
    tsvs = sorted(root.glob('sub-*/ses-*/eeg/*_events.tsv'))
    if not tsvs:
        print(f'No events.tsv files under {root}')
        return 1
    print(f'== smoke test on {len(tsvs)} events.tsv files ==')

    total_rows = total_resolved = total_missing = 0
    by_session: dict[tuple[int, int], tuple[int, int, int]] = {}

    for p in tsvs:
        m = PATH_RE.search(str(p))
        if not m:
            continue
        sub, ses = int(m.group(1)), int(m.group(2))
        df = pd.read_csv(p, sep='\t')
        paths = aligner.paths_for_events(df, subject=sub, session=ses)
        n_rows = len(df)
        n_ok = sum(1 for x in paths if x is not None and x.exists())
        n_miss = sum(1 for x in paths if x is not None and not x.exists())
        flag = '✓' if n_miss == 0 else '✗'
        print(f'  {flag} sub-{sub:02d} ses-{ses:02d}: rows={n_rows:4d}  resolved+exists={n_ok:4d}  missing={n_miss:3d}')
        by_session[(sub, ses)] = (n_rows, n_ok, n_miss)
        total_rows += n_rows
        total_resolved += n_ok
        total_missing += n_miss

    print()
    print('== summary ==')
    print(f'   sessions   : {len(by_session)}')
    print(f'   total rows : {total_rows}')
    print(f'   resolved+exists : {total_resolved}')
    print(f'   resolved+missing: {total_missing}')
    return 1 if total_missing else 0


if __name__ == '__main__':
    sys.exit(run())
