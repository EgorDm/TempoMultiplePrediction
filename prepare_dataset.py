import os
from typing import List

import msgpack
import click
import random

import settings
import osu


@click.command()
@click.option('--n_samples', default=3000, help='Number of sampels to pick')
def main(n_samples):
    db = osu.database.OsuDB(os.getenv('OSU_ROOT') + '\\osu!.db')

    def criteria(e: osu.models.BeatmapEntry):
        if e.ranked < 4: return False
        if e.mode != 0: return False
        if e.time_drain < 60 and e.time_drain < 420: return False
        return True

    samples = pick_samples(db.beatmaps, n_samples, criteria, True)
    save_entries(samples, 'samples')


def serialize_entry(entry: osu.models.BeatmapEntry):
    return {
        'data': entry.get_path(os.getenv("OSU_ROOT"), entry.osu_file),
        'music': entry.get_path(os.getenv("OSU_ROOT"), entry.audio_file),
    }


def save_entries(samples, name, serialize=serialize_entry):
    with open(f'data/{name}.dat', 'wb') as f:
        msgpack.pack([serialize(s) for s in samples], f, encoding='utf-8')


def pick_samples(candidates: List[osu.models.BeatmapEntry], pick_count, predicate, pick_random=True):
    if len(candidates) < pick_count: raise Exception('Not enough candidates')
    ret = []
    ret_set_ids = []

    if pick_random:
        def gen():
            while True: yield random.randint(0, len(candidates) - 1)

        generator = gen()
    else: generator = iter(range(len(candidates)))

    while len(ret) < pick_count:
        candidate = candidates[next(generator)]
        if not predicate(candidate):
            ret_set_ids.append(candidate.set_id)
            continue
        if candidate.set_id in ret_set_ids: continue
        ret.append(candidate)
        ret_set_ids.append(candidate.set_id)

    return ret


if __name__ == "__main__":
    main()
