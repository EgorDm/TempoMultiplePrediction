import msgpack
import os

import click

from helpers import sample_utils, osu


@click.command()
@click.option('--sample_list', default='samples', help='List with samples to use')
def main(sample_list):
    samples = sample_utils.read_samples(sample_list)
    process_sample(samples[0])


def process_sample(sample):
    audio = sample['music']
    beatmap = osu.beatmap_reader.read(sample['data'])
    timingpoints = list(filter(lambda x: isinstance(x, osu.models.KeyTimingPoint), beatmap.timingpoints))
    print('hi')


if __name__ == "__main__":
    main()
