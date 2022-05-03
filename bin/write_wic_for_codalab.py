
from allennlp.models.archival import load_archive
from allennlp.data import DatasetReader, Vocabulary, DataIterator
from allennlp.nn.util import move_to_device
from allennlp.common import Params

import numpy as np
import pandas as pd

from kb.include_all import *
from sklearn.linear_model import logistic_regression_path


def write_for_official_eval(model_archive_file, test_file, output_file):
    archive = load_archive(model_archive_file)
    model = archive.model

    reader = DatasetReader.from_params(archive.config['dataset_reader'])

    iterator = DataIterator.from_params(Params({"type": "basic", "batch_size": 32}))
    vocab = Vocabulary.from_params(archive.config['vocabulary'])
    iterator.index_with(vocab)

    model.cuda()
    model.eval()

    label_ids_to_label = {0: '0', 1: '1'}

    instances = reader.read(test_file)
    predictions = []
    logits = []
    for batch in iterator(instances, num_epochs=1, shuffle=False):
        batch = move_to_device(batch, cuda_device=0)
        output = model(**batch)

        batch_labels = [
            label_ids_to_label[i]
            for i in output['predictions'].cpu().numpy().tolist()
        ]
        batch_logits = [
            i
            for i in output['logits'].cpu().numpy().tolist()
        ]

        predictions.extend(batch_labels)
        logits.extend(batch_logits)
    # assert len(predictions) == 1400

    # with open(output_file, 'w') as fout:
    #     for p in predictions:
    #         fout.write("{}\n".format(p))
    # with open(output_file, 'w') as fout:
    #     for p in logits:
    #         fout.write("{}\n".format(p))
    logits = pd.DataFrame(logits)
    logits.to_csv(output_file, sep = '\t', header=None, index = None)
write_for_official_eval('/nas/home/gujiashe/kb/yago_output/model.tar.gz', '/nas/home/gujiashe/critic/lp','/nas/home/gujiashe/critic/lp.logits.txt')

