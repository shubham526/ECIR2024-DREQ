import torch
import utils
import argparse
from dataset import DocRankingDataset
from model import DocRankingModel
from transformers import AutoTokenizer
from dataloader import DocRankingDataLoader
import evaluate


def test(model, data_loader, run_file, device):
    res_dict = evaluate.evaluate(
        model=model,
        data_loader=data_loader,
        device=device
    )

    print('Writing run file...')
    utils.save_trec(run_file, res_dict)
    print('[Done].')


def main():
    parser = argparse.ArgumentParser("Script to test a model.")
    parser.add_argument('--test', help='Training data.', required=True, type=str)
    parser.add_argument('--max-len', help='Maximum length for truncation/padding. Default: 512', default=512, type=int)
    parser.add_argument('--checkpoint', help='Name of checkpoint to load.', required=True, type=str)
    parser.add_argument('--text-enc', help='Name of model (bert|distilbert|roberta|deberta|ernie|electra|conv-bert|t5).'
                                           'Default: bert.', type=str, default='bert')
    parser.add_argument('--save', help='Output run file in TREC format.', required=True,
                        type=str)
    parser.add_argument('--batch-size', help='Size of each batch. Default: 8.', type=int, default=8)
    parser.add_argument('--num-workers', help='Number of workers to use for DataLoader. Default: 0', type=int,
                        default=0)
    parser.add_argument('--cuda', help='CUDA device number. Default: 0.', type=int, default=0)
    parser.add_argument('--use-cuda', help='Whether or not to use CUDA. Default: False.', action='store_true')
    args = parser.parse_args()

    cuda_device = 'cuda:' + str(args.cuda)
    print('CUDA Device: {} '.format(cuda_device))

    device = torch.device(
        cuda_device if torch.cuda.is_available() and args.use_cuda else 'cpu'
    )

    cuda_device = 'cuda:' + str(args.cuda)
    print('CUDA Device: {} '.format(cuda_device))

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

    pretrain = vocab = model_map[args.text_enc]
    tokenizer = AutoTokenizer.from_pretrained(vocab, model_max_length=args.max_len)
    print('Text Encoder: {}'.format(args.text_enc))

    print('Creating test set...')
    test_set = DocRankingDataset(
        dataset=args.test,
        tokenizer=tokenizer,
        train=False,
        max_len=args.max_len,
    )
    print('[Done].')

    print('Creating data loaders...')
    print('Number of workers = ' + str(args.num_workers))
    print('Batch Size = ' + str(args.batch_size))

    test_loader = DocRankingDataLoader(
        dataset=test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    print('[Done].')

    model = DocRankingModel(pretrained=pretrain)

    print('Loading checkpoint...')
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    print('[Done].')

    print('Using device: {}'.format(device))
    model.to(device)

    print("Starting to test...")
    test(
        model=model,
        data_loader=test_loader,
        run_file=args.save,
        device=device
    )

    print('Test complete.')


if __name__ == '__main__':
    main()
