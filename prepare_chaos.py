from datasets import load_dataset, DatasetDict, load_from_disk
import numpy as np
from math import log, e
from collections import Counter


def process(sample):
    label = np.argmax(sample['label_dist'])
    diff = 1 - max(sample['label_dist'])

    ent_class = 0
    for i in range(2, -1, -1):
        if sample['entropy'] >= thresholds[i]:
            ent_class = i
            break

    return {'label': label,
            'sentence1': sample['example']['premise'],
            'sentence2': sample['example']['hypothesis'],
            'diff': diff,
            'entropy_class': ent_class}


if __name__ == '__main__':
    train = load_dataset('json',
            data_files= {'train': ['./chaosNLI_v1.0/chaosNLI_snli.jsonl',
                    './chaosNLI_v1.0/chaosNLI_mnli_m.jsonl']})
    snli = load_from_disk('/data/mohamed/data/snli_balanced')
    snli['dev'] = snli['dev'].filter(lambda x: x['pairID'] not in train['train']['uid'])


    thresholds = [np.percentile(train['train']['entropy'], i/3*100)
            for i in range(3)]

    train = train.map(process,
            remove_columns = ['uid', 'label_counter', 'majority_label', 'label_dist', 'label_count', 'example', 'old_label', 'old_labels'])


    data = DatasetDict({'train': train['train'],
        'dev': snli['dev'],
        'test': snli['test']})

    data.save_to_disk('./chaosnli')
