import argparse
from typing import Dict, List

from rouge import Rouge
from tqdm import tqdm
import numpy as np

from data.dataset import PreprocessingDataset, NewsJsonDataset
from data.processors import ComplexProcessor
from models.extractive import extractive_predict_max
from scripts.utils import load_model, get_processors


def print_rouge(scores: List[Dict[str, Dict[str, float]]]):
    rouge_1_f = np.mean([x['rouge-1']['f'] for x in scores])
    rouge_2_f = np.mean([x['rouge-2']['f'] for x in scores])
    rouge_l_f = np.mean([x['rouge-l']['f'] for x in scores])

    rouge_1_p = np.mean([x['rouge-1']['p'] for x in scores])
    rouge_2_p = np.mean([x['rouge-2']['p'] for x in scores])
    rouge_l_p = np.mean([x['rouge-l']['p'] for x in scores])

    rouge_1_r = np.mean([x['rouge-1']['r'] for x in scores])
    rouge_2_r = np.mean([x['rouge-2']['r'] for x in scores])
    rouge_l_r = np.mean([x['rouge-l']['r'] for x in scores])

    print('====================================')
    print('| metric | precision | recall | f1 |')
    print('====================================')
    print(f'| rouge-1 | {rouge_1_p} | {rouge_1_r} | {rouge_1_f} |')
    print(f'| rouge-2 | {rouge_2_p} | {rouge_2_r} | {rouge_2_f} |')
    print(f'| rouge-l | {rouge_l_p} | {rouge_l_r} | {rouge_l_f} |')
    print('====================================')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--embedding_model', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)

    args = parser.parse_args()

    processors = get_processors(args.embedding_model)
    model = load_model(args.model_path)

    dataset = PreprocessingDataset(NewsJsonDataset(args.data_path, gziped=True), processor=ComplexProcessor(processors))
    rouge = Rouge()
    expected = []
    predicted = []
    for sample in tqdm(dataset):
        title = ' '.join(sample['title_lemmas'])

        idx = extractive_predict_max(model, sample['sentence_embeddings'])

        predicted_sentence = ' '.join(sample['text_lemmas'][idx])
        expected.append(title)
        predicted.append(predicted_sentence)
    scores = rouge.get_scores(predicted, expected)

    print_rouge(scores)


if __name__ == '__main__':
    main()
