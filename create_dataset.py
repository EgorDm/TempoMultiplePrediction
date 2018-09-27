from typing import List, Tuple

import msgpack
import os

import click
import numpy as np
import libtempo_py as lt
import pandas as pd
import h5py
from tqdm import tqdm

import settings
from helpers import sample_utils, osu

OSU_ROOT = os.getenv('OSU_ROOT')
ref_tempo = 60
bpm_range = (30, 630)
merge_truth_threshold = 10  # s
merge_labels_threshold = 4  # s


def process_entries(entry):
    sample_path = f'{OSU_ROOT}\\Songs\\{entry.folder_name}'
    beatmap = osu.beatmap_reader.read(f'{sample_path}\\{entry.osu_file}', ['timing'])

    audio = lt.audio.AudioWrapper(f'{sample_path}\\{entry.audio_file}')
    audio_mono = lt.wrap_arrayF(np.mean(audio.get_data().to_array(), axis=0))

    # Novelty curve
    novelty_curve, novelty_curve_sr = lt.audio_to_novelty_curve(audio_mono, audio.get_sr())

    # Tempogram
    tempogram_y_axis = lt.wrap_arrayD(np.arange(*bpm_range))
    tempogram, tempogram_y_axis, t = lt.novelty_curve_to_tempogram(novelty_curve, tempogram_y_axis, novelty_curve_sr, 8)
    cyclic_tempogram, cyclic_tempogram_y_axis = lt.tempogram_to_cyclic_tempogram(tempogram, tempogram_y_axis, ref_tempo=ref_tempo)
    smoothed_tempogram = lt.smoothen_tempogram(cyclic_tempogram, cyclic_tempogram_y_axis, t, 20)

    # Tempo sections
    tempo_curve = lt.tempogram_to_tempo_curve(smoothed_tempogram, cyclic_tempogram_y_axis)
    tempo_curve = lt.correct_tempo_curve(tempo_curve, t, 5)
    tempo_sections = lt.curve_to_sections(tempo_curve, t, ref_tempo, 100000, 1.5)

    # Truth sections
    key_timingpoints = list(filter(lambda x: isinstance(x, osu.models.KeyTimingPoint), beatmap.timingpoints))
    truth_sections = timingpoints_to_section(key_timingpoints, tempo_sections[-1].end)
    truth_merge_predicate = lambda x, y: abs(x.bpm - y.bpm) < merge_truth_threshold
    truth_sections = merge_if(truth_sections, truth_merge_predicate, merge_sections)

    # Label sections & label multiples
    label_section_multiple = []
    candidate_sections = list(reversed(tempo_sections))
    while len(candidate_sections) > 0:
        s = candidate_sections.pop()
        s_truth = section_from_t(s.start, truth_sections)
        if s.end > s_truth.end:
            s, sn = slice_section(s, s_truth.end)
            candidate_sections.append(sn)

        multiple = int(np.round(s_truth.bpm / s.bpm))
        label_section_multiple.append((s, multiple))
    label_merge_predicate = lambda x, y: x[1] == y[1] and abs(x[0].bpm - y[0].bpm) < merge_labels_threshold
    label_merge_fn = lambda x, y: (merge_sections(x[0], y[0]), x[1])
    label_section_multiple = merge_if(label_section_multiple, label_merge_predicate, label_merge_fn)

    tempogram_np = np.abs(tempogram.to_array())
    t_np = t.to_array()[:, 0]
    samples = [(section_to_sample(tempogram_np, t_np, s), s.bpm, m) for s, m in label_section_multiple]

    return samples


def section_to_sample(tempogram: np.ndarray, t: np.ndarray, s: lt.Section):
    section_start_idx = np.searchsorted(t, s.start)
    section_end_idx = np.searchsorted(t, s.end)

    tempo_distribution = np.sum(tempogram[:, section_start_idx:section_end_idx], axis=1)
    tempo_distribution -= np.mean(tempo_distribution)
    tempo_distribution[tempo_distribution < 0] = 0
    tempo_distribution /= max(np.max(tempo_distribution), 0.00001)

    tempo_hints = np.zeros_like(tempo_distribution)
    for i in np.exp2([0, 1, 2, 3]):
        pos = int(np.round(i * s.bpm - bpm_range[0]))
        if pos >= tempogram.shape[0]: continue
        tempo_hints[pos] = 1 if i == 1 else 0.5

    return np.stack([tempo_distribution, tempo_hints], axis=1)


def timingpoints_to_section(tps: List[osu.models.KeyTimingPoint], end_t) -> List[lt.Section]:
    ret = []
    for i in range(len(tps)):
        end = tps[i + 1].offset if i + 1 < len(tps) else end_t
        tp = tps[i]
        ret.append(lt.Section(abs(tp.offset) / 1000, end, 60000 / tp.mpb, tp.offset / 1000))
    return ret


def merge_sections(s1: lt.Section, s2: lt.Section) -> lt.Section:
    return lt.Section(min(s1.start, s2.start), max(s1.end, s2.end), np.mean([s1.bpm, s2.bpm]), np.mean([s1.offset, s2.offset]))


def section_from_t(t, sections: List[lt.Section]) -> lt.Section:
    for s in sections:
        if s.start <= t < s.end: return s
    return sections[-1]


def slice_section(s: lt.Section, t) -> Tuple[lt.Section, lt.Section]:
    s1 = lt.Section(s.start, t, s.bpm, s.offset)
    s2 = lt.Section(t, s.end, s.bpm, s.offset)
    return s1, s2


def merge_if(l, predicate, merge_f):
    if len(l) == 0: return l
    ret = [l[0]]
    for x in l:
        if predicate(ret[-1], x): ret[-1] = merge_f(ret[-1], x)
        else: ret.append(x)
    return ret


def save_dataset(samples: np.ndarray, i: int):
    with open(f'data/dataset_part_{i}.npz', 'wb') as file:
        header = {'part': i, 'samples': len(samples)}
        np.save(file, header)
        np.save(file, samples)


@click.command()
def main():
    dataset = pd.read_csv("data/dataset_entries.csv", dtype=dict(avg_bpm=np.float64, time_total=np.int32))

    samples = []
    dataset_part = 0
    samples_per_dataset = 500
    error_entries = []
    bar = tqdm(total=len(dataset.index))
    for i, entry in tqdm(dataset.iterrows()):
        try:
            samples += process_entries(entry)
        except Exception as e:
            error_entries.append(entry)
            print(f'Failed processing {entry.title}. {str(e)}')

        bar.update(1)

        if len(samples) > samples_per_dataset:
            save_dataset(samples, dataset_part)
            dataset_part += 1

    save_dataset(np.array(samples), dataset_part)


if __name__ == "__main__":
    main()
