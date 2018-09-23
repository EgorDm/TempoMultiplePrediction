import os

import click
import pandas
import settings
from helpers import sample_utils, osu


@click.command()
@click.option('--n_samples', default=3000, help='Number of sampels to pick')
def main(n_samples):
    db = osu.database.OsuDB(os.getenv('OSU_ROOT') + '\\osu!.db')

    def criteria(e: osu.models.BeatmapEntry):
        if e.ranked < 4: return False
        if e.mode != 0: return False
        if e.time_drain < 60 and e.time_drain < 620: return False
        return True

    candidates = list(filter(criteria, db.beatmaps))
    dataframe = sample_utils.entries_dataframe(candidates)
    dataframe.drop_duplicates(subset='set_id', inplace=True)
    dataframe.to_csv('data/entries.csv')


if __name__ == "__main__":
    main()
