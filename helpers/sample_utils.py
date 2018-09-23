import random
from typing import List

import msgpack
import os
import pandas as pd
from helpers import osu


def entries_dataframe(entries: [osu.models.BeatmapEntry], columns=None) -> pd.DataFrame:
    if columns is None: columns = ['title', 'artist', 'creator', 'folder_name', 'audio_file', 'osu_file', 'beatmap_id', 'set_id', 'time_total', 'avg_bpm']
    column_values = {key: [] for key in columns}

    for e in entries:
        for key in columns:
            column_values[key].append(getattr(e, key))

    return pd.DataFrame(data=column_values)
