import os

import click
import numpy as np


@click.command()
@click.option('--dataset_path', default='data/dataset', help='Path of your dataset.')
def main(dataset_path):
    data_files = os.listdir(dataset_path)

    samples = []
    for file in data_files:
        with open(f'{dataset_path}/{file}', 'rb') as f:
            header = np.load(f).item()
            data = np.load(f)
            samples.append(data)

    samples = np.concatenate(samples, axis=0)

    with open(f'data/dataset/dataset.npz', 'wb') as file:
        header = {'samples': len(samples)}
        np.save(file, header)
        np.save(file, samples)


if __name__ == "__main__":
    main()
