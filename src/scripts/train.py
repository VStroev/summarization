import argparse
import os

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.dataset import FieldUnstackingDataset, PreprocessingDataset, NewsJsonDataset
from data.processors import TagCleaningProcessor, PavlovSentenceDetectionProcessor, PyMyStemLemmatizingProcessor, \
    SentenceRankingProcessor, Word2VecProcessor, SentenceEmbeddingSumProcessor, FieldFilteringProcessor, \
    RankLabelingProcessor, ComplexProcessor
from models.extractive import Perceptron


def train_loop(model, loader, optimizer, criterion, ckpt_file, epoches, steps = 0):
    writer = SummaryWriter()
    cumloss = 0
    loss_counter = 0
    for epoch in range(1, epoches + 1):
        print('Epoch', epoch)
        for inputs, labels in loader:
            inputs = inputs.float()
            labels = labels.float()
            steps += 1
            optimizer.zero_grad()

            res = model(inputs)
            res = res.squeeze(-1)
            loss = criterion(res, labels)

            loss.backward()
            optimizer.step()
            cumloss += loss.item()
            loss_counter += 1
            if steps % 1000 == 0:
                print(steps, cumloss / loss_counter)
                writer.add_scalar('Loss/train', cumloss / loss_counter, steps)

                cumloss = 0
                loss_counter = 0
            if steps % 5000 == 0:
                torch.save({'cls': model.state_dict(),
                            'optim': optimizer.state_dict(),
                            'steps': steps},
                           ckpt_file)


def train_model(data_path, embedding_model_path, batch_size, load_ckpt, ckpt_path, epoches):
    processors = [TagCleaningProcessor(),
                  PavlovSentenceDetectionProcessor(),
                  PyMyStemLemmatizingProcessor(),
                  SentenceRankingProcessor(),
                  Word2VecProcessor(embedding_model_path),
                  SentenceEmbeddingSumProcessor(),
                  FieldFilteringProcessor(keep_fields=['sentence_ranks', 'sentence_embeddings'], ),
                  RankLabelingProcessor()
                  ]

    dataset = FieldUnstackingDataset(PreprocessingDataset(NewsJsonDataset(data_path),
                                                          processor=ComplexProcessor(processors)),
                                     ['sentence_embeddings', 'rank_labels'])
    loader = DataLoader(dataset, batch_size)

    model = Perceptron(300, 100)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    criterion = torch.nn.BCEWithLogitsLoss()

    steps = 0

    if load_ckpt and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['cls'])
        optimizer.load_state_dict(ckpt['optim'])
        steps = ckpt['steps']

    train_loop(model, loader, optimizer, criterion, ckpt_path, epoches, steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--embedding_model', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--test_size', type=int, required=False, default=1000)
    parser.add_argument('--batch_size', type=int, required=False, default=5)
    parser.add_argument('--epoch', type=int, required=False, default=1)
    parser.add_argument('--load', required=False, default=False, action='store_true')

    args = parser.parse_args()
    train_model(args.data_path, args.embedding_model, args.batch_size, args.load, args.out_path, args.epoch)


if __name__ == '__main__':
    main()
