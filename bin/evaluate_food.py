
from allennlp.commands.evaluate import *
from kb.include_all import *

import glob

def go(archive_file, cuda_device, fname, all_only=True):
    archive = load_archive(archive_file, cuda_device)
    
    config = archive.config
    prepare_environment(config)

    reader_params = config.pop('dataset_reader')
    if reader_params['type'] == 'multitask_reader':
        reader_params = reader_params['dataset_readers']['language_modeling']
    # reader_params['num_workers'] = 0
    validation_reader_params = {
        "type": "food",
        "tokenizer_and_candidate_generator": reader_params['base_reader']['tokenizer_and_candidate_generator'].as_dict()
    }
    dataset_reader = DatasetReader.from_params(Params(validation_reader_params))
    instances = dataset_reader.read(fname)

    model = archive.model
    model.eval()


    iterator = DataIterator.from_params(Params(
        {"type": "basic", "batch_size": 1}
    ))
    iterator.index_with(model.vocab)

    # if all_only:
    #     fnames = [datadir + '/all.text']
    # else:
    #     fnames = glob.glob(datadir + '/*.text')

    # for fname in sorted(fnames):
    
    print("start")
    metrics = evaluate(model, instances, iterator, cuda_device, "")
    print("================")
    print(fname)
    print(metrics)
    print("================")
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_archive', type=str)
    parser.add_argument('--datafile', type=str)
    parser.add_argument('--cuda_device', type=int, default=-1)

    args = parser.parse_args()

    go(args.model_archive, args.cuda_device, args.datafile)





