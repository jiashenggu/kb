from allennlp.commands.evaluate import *
from kb.include_all import *
from allennlp.nn import util as nn_util
import torch

def generate(archive_file, cuda_device, line, all_only=True):
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
    vocab = dataset_reader._tokenizer_and_candidate_generator.bert_tokenizer.vocab
    instance = dataset_reader.read_food(line)

    model = archive.model
    model.eval()
    

    print("start")
    # metrics = evaluate(model, instances, iterator, cuda_device, "")
    batch = nn_util.move_to_device(instance, cuda_device)
    output_dict = model(**batch)
    pooled_output = output_dict.get("pooled_output")
    contextual_embeddings = output_dict.get("contextual_embeddings")
    prediction_scores, seq_relationship_score = model.pretraining_heads(
    contextual_embeddings, pooled_output
    )
    
    print("================")
    print(prediction_scores)
    print("================")
    idx = torch.argmin(prediction_scores, dim = 1)
    print(line, vocab[idx])

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_archive', type=str)
    # parser.add_argument('--line', type=str)
    parser.add_argument('--cuda_device', type=int, default=-1)

    args = parser.parse_args()
    line = "banana\tcolor\tyellow"
    if not line:
        line = input()
    generate(args.model_archive, args.cuda_device, line)





