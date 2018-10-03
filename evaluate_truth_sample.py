import click

import settings
import models
import libtempo_py as lt
import numpy as np
import os

ref_tempo = 60
bpm_range = (30, 630)
merge_truth_threshold = 10  # s
merge_labels_threshold = 4  # s
section_length_thresh = 60


@click.command()
@click.option('--audio', default='data/riders.mp3', help='Audio file to test on')
@click.option('--model', default='dnn_bpm_classify_real_comp_model', help='Model save file')
@click.option('--weights', default='dnn_bpm_classify_real_comp_weights-07-0.854', help='Weigth save file')
def main(audio, model, weights):
    audio_file = lt.audio.AudioFile.open(audio)
    audio_mono = np.mean(audio_file.data, axis=0)

    # Novelty curve
    novelty_curve, novelty_curve_sr = lt.audio_to_novelty_curve(audio_mono, audio_file.sr)

    # Tempogram
    tempogram_y_axis = np.arange(*bpm_range)
    tempogram, t = lt.novelty_curve_to_tempogram(novelty_curve, tempogram_y_axis, novelty_curve_sr, 8)
    cyclic_tempogram, cyclic_tempogram_y_axis = lt.tempogram_to_cyclic_tempogram(tempogram, tempogram_y_axis, ref_tempo=ref_tempo)
    smoothed_tempogram = lt.smoothen_tempogram(cyclic_tempogram, cyclic_tempogram_y_axis, t, 20)

    # Tempo sections
    tempo_curve = lt.tempogram_to_tempo_curve(smoothed_tempogram, cyclic_tempogram_y_axis)
    tempo_curve = lt.correct_tempo_curve(tempo_curve, t, 5)
    tempo_sections = lt.curve_to_sections(tempo_curve, t, ref_tempo, 100000, 1.5)
    if tempo_sections[-1].end > len(novelty_curve) / novelty_curve_sr: tempo_sections[-1].end -= 0.1
    tempo_sections = lt.sections_extract_offset(novelty_curve, tempo_sections, [1, 2, 4], novelty_curve_sr, bpm_doubt_window=5)

    # Correct sections by multiple
    model = models.load_model(model, weights)

    # Crea
    for s in tempo_sections:
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

        x = np.stack([tempo_distribution, tempo_hints], axis=1)
        y = model.predict(np.array([x]), 1)[0]
        m = np.exp2(np.argmax(y))

        print(f'Section: START({s.start:.2f}) BPM_FROM({s.bpm:.2f}) BPM_TO({s.bpm * m:.2f}) MULTIPLE({m}) CONFIDENCE({(y[np.argmax(y)]):.2f})')
        s.bpm = s.bpm * m

    # Add clicks to audio
    click_track = np.zeros_like(audio_mono)
    for s in tempo_sections:
        length = int((s.end - s.start) * audio_file.sr)
        start = int(s.start * audio_file.sr)
        clicks = lt.audio.annotation.click_track_from_tempo(s.bpm, s.offset - s.start, length, 8, audio_file.sr)

        if len(clicks) + start >= len(click_track): clicks = clicks[0: len(click_track) - start]

        click_track[start: start + len(clicks)] = clicks

    audio_data = np.copy(audio_file.data)
    for c in range(audio_data.shape[0]):
        audio_data[c, :] = np.clip(audio_data[c, :] + click_track, -1, 1)

    # save audio
    audio_file.data = audio_data
    audio_file.save(os.path.splitext(audio)[0] + '_clicks.wav')


if __name__ == "__main__":
    main()
