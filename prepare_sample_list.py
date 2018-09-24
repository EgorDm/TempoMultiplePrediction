import os
import numpy as np

import click
import settings
import pandas as pd
from helpers import sample_utils, osu


@click.command()
def main():
    db = osu.database.OsuDB(os.getenv('OSU_ROOT') + '\\osu!.db')

    def criteria(e: osu.models.BeatmapEntry):
        if e.ranked < 4: return False
        if e.mode != 0: return False
        if e.time_drain < 60 and e.time_drain < 620: return False
        return True

    # Load all entries
    candidates = list(filter(criteria, db.beatmaps))
    entries = sample_utils.entries_dataframe(candidates).drop_duplicates(subset='set_id')
    entries.to_csv('data/entries.csv')

    # Create a balanced dataset
    n_buckets = 30
    buckets = np.linspace(30, 300, n_buckets)
    bucket_labels = np.arange(n_buckets - 1)
    entries['bpm_class'] = pd.cut(entries['avg_bpm'], buckets, labels=bucket_labels)

    start_bucket = 4
    end_bucket = 18
    filtered_entries = entries.loc[(entries['bpm_class'] >= start_bucket) & (entries['bpm_class'] <= end_bucket)]
    grouped_entries = filtered_entries.groupby('bpm_class')

    group_select_count = grouped_entries.size()[grouped_entries.size() > 0].min()
    if group_select_count == 0: raise Exception('')

    balanced_entries = grouped_entries.apply(lambda x: x.sample(group_select_count) if len(x) > 0 else x)
    balanced_entries.reset_index(drop=True, inplace=True)
    # balanced_entries.drop(columns=['Unnamed: 0'], inplace=True)
    balanced_entries.to_csv('data/dataset_entries.csv')
    print(f'Created a dataset with size of {group_select_count * (end_bucket - start_bucket)}')


if __name__ == "__main__":
    main()
