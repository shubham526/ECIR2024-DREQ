import argparse
import torch
import os
from transformers import AutoTokenizer
import utils
from model import BaselineModel
from dataset import BaselineDataset
from dataloader import BaselineDataLoader
import evaluate


def test(model, data_loader, device):
    return evaluate.evaluate(model=model, data_loader=data_loader, device=device)


def main():
    parser = argparse.ArgumentParser("Script to run inference on a fine-tuned model.")
    parser.add_argument('--model', help='Name of model (bert|distilbert|roberta|deberta|ernie|electra|conv-bert|t5). '
                                        'Default: bert.', type=str, default='bert')
    parser.add_argument('--test', help='Test data.', required=True, type=str)
    parser.add_argument('--max-len', help='Maximum length for truncation/padding. Default: 512', default=512, type=int)
    parser.add_argument('--save-dir', help='Directory where output run is saved.', required=True, type=str)
    parser.add_argument('--run', help='Output run file.', required=True, type=str)
    parser.add_argument('--checkpoint', help='Name of checkpoint to load.', type=str, default=None)
    parser.add_argument('--batch-size', help='Size of each batch. Default: 8.', type=int, default=8)
    parser.add_argument('--num-workers', help='Number of workers for DataLoader. Default: 8', type=int, default=8)
    parser.add_argument('--use-cuda', help='Whether to use CUDA. Default: False', action='store_true')
    parser.add_argument('--cuda', help='CUDA device number. Default: 0.', type=int, default=0)

    args = parser.parse_args()

    cuda_device = 'cuda:' + str(args.cuda)
    print('CUDA Device: {} '.format(cuda_device))

    device = torch.device(
        cuda_device if torch.cuda.is_available() and args.use_cuda else 'cpu'
    )

    model_map = {
        'bert': 'bert-base-uncased',
        'distilbert': 'distilbert-base-uncased',
        'roberta': 'roberta-base',
        'deberta': 'microsoft/deberta-base',
        'ernie': 'nghuyong/ernie-2.0-base-en',
        'electra': 'google/electra-small-discriminator',
        'conv-bert': 'YituTech/conv-bert-base',
        't5': 't5-base'
    }

    pretrain = vocab = model_map[args.model]
    tokenizer = AutoTokenizer.from_pretrained(vocab)

    print('Reading test data....')
    test_set = BaselineDataset(
        dataset=args.test,
        tokenizer=tokenizer,
        train=False,
        max_len=args.max_len
    )
    print('[Done].')

    print('Creating data loader...')
    test_loader = BaselineDataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    print('[Done].')
    model = BaselineModel(pretrained=pretrain)
    if args.checkpoint is not None:
        print('Loading checkpoint...')
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        print('[Done].')

    model.to(device)

    print('Running inference...')

    res_dict = test(
        model=model,
        data_loader=test_loader,
        device=device
    )
    print('Writing run file...')
    utils.save_trec(os.path.join(args.save_dir, args.run), res_dict)
    print('[Done].')

    print('[Done].')
    print('Run file saved to ==> {}'.format(args.run))


if __name__ == "__main__":
    main()
