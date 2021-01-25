import torch

from data.processors import TagCleaningProcessor, PavlovSentenceDetectionProcessor, PyMyStemLemmatizingProcessor, \
    SentenceRankingProcessor, Word2VecProcessor, SentenceEmbeddingSumProcessor
from models.extractive import extractive_predict_max, Perceptron


def load_model(model_path):
    ckpt = torch.load(model_path)
    model = Perceptron(300, 100)
    model.load_state_dict(ckpt['cls'])
    model.eval()
    return model


def get_processors(embedding_path):
    return [TagCleaningProcessor(),
            PavlovSentenceDetectionProcessor(),
            PyMyStemLemmatizingProcessor(),
            SentenceRankingProcessor(),
            Word2VecProcessor(embedding_path),
            SentenceEmbeddingSumProcessor(),
            ]


class TextSummariser:
    def __init__(self, processors, model):
        self.model = model
        self.processors = processors

    def __call__(self, text: str):
        sample = {'text': text, 'title': ''}
        sample = self._apply_processors(sample)
        idx = extractive_predict_max(self.model, sample['sentence_embeddings'])

        return sample['text_sentences'][idx]

    def _apply_processors(self, sample):
        for processor in self.processors:
            sample = processor(sample)
        return sample

