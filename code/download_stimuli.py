"""
Targeted downloader for nm000133 (Alljoined1) stimulus images.

The BIDS dataset's events.tsv files reference 960 unique COCO 2017
images (a subset of NSD's `shared1000`). Rather than fetch the entire
20 GB COCO 2017 distribution, we download only those 960 images
(~140 MB) directly from COCO's public mirror.

After this, every ``stim_file`` reference in every events.tsv resolves
to ``stimuli/<cocoSplit>/<filename>.jpg``.

Idempotent: skips files already on disk with the expected size.

License notes
-------------
COCO 2017 images are downloadable under the COCO license / Flickr terms.
This dataset bundles them under the upstream Alljoined1 license
(CC-BY-NC-ND-4.0) for research / non-commercial use.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parent))
from align_stimuli import StimulusAligner

ROOT_DEFAULT = Path(__file__).resolve().parent.parent
COCO_BASE = 'http://images.cocodataset.org'
USER_AGENT = 'Mozilla/5.0 nm000133-stimulus-fetcher'


def fetch_one(url: str, dest: Path, retries: int = 3, timeout: int = 30) -> None:
    if dest.exists() and dest.stat().st_size > 1024:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={'User-Agent': USER_AGENT})
            with urlopen(req, timeout=timeout) as r:
                data = r.read()
            dest.write_bytes(data)
            return
        except URLError as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise SystemExit(f'failed to download {url}: {last_err}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=str(ROOT_DEFAULT),
                    help='dataset root (default: parent of code/)')
    ap.add_argument('-j', '--jobs', type=int, default=4,
                    help='parallel downloads (default: 4)')
    args = ap.parse_args()

    root = Path(args.root)
    stim_root = root / 'stimuli'
    stim_root.mkdir(exist_ok=True)

    aligner = StimulusAligner(root)
    needed = list(aligner.needed_coco_files())
    print(f'[download_stimuli] {len(needed)} unique COCO images to fetch '
          f'into {stim_root}')

    from concurrent.futures import ThreadPoolExecutor, as_completed
    done = 0
    errors = 0

    def task(coco_id: int, coco_split: str, fname: str) -> None:
        url = f'{COCO_BASE}/{coco_split}/{int(coco_id):012d}.jpg'
        dest = stim_root / fname
        fetch_one(url, dest)

    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(task, *t): t for t in needed}
        for fut in as_completed(futures):
            try:
                fut.result()
                done += 1
            except Exception as e:
                errors += 1
                print(f'  ERROR: {e}')
            if done % 50 == 0:
                print(f'  progress: {done}/{len(needed)}')

    print(f'\n[download_stimuli] done: {done} ok, {errors} errors')


if __name__ == '__main__':
    main()
