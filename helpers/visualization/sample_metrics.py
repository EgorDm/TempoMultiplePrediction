import numpy as np
import matplotlib.pyplot as plt
import helpers


def bpm_histogram(candidates: [helpers.osu.models.BeatmapEntry]):
    data = []
    for c in candidates:
        timingpoints = list(filter(lambda x: isinstance(x, helpers.osu.models.KeyTimingPoint) and x.mpb > 0, c.timingpoints))
        if len(timingpoints) == 0: continue
        avg_mpb = sum([t.mpb for t in timingpoints]) / len(timingpoints)
        data.append(60000/avg_mpb)

    plt.hist(np.array(data), 50, range=(50, 250))
    plt.plot()


# import os
# db = helpers.osu.database.OsuDB(os.getenv('OSU_ROOT') + '\\osu!.db')
# ranked_entries = list(filter(lambda e: e.ranked >= 4, db.beatmaps))
# bpm_histogram(ranked_entries)
