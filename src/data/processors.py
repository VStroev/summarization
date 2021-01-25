import abc
import re
import numpy as np
from typing import List, Optional

import gensim

import nltk
from deeppavlov.models.tokenizers.ru_sent_tokenizer import RuSentTokenizer
from pymystem3 import Mystem

# TODO: Need tests


class EmptySampleException(Exception):
    def __init__(self, message="This sample is empty!"):
        super().__init__(message)


class RequirementNotSatisfiedException(Exception):
    def __init__(self, field: str, message="Requirement is not satisfied, field: "):
        super().__init__(message + field)


class BaseProcessor(metaclass=abc.ABCMeta):
    """
    Abstract class for sample pre processing (examples: tokenization, lemmatization)
    """
    @abc.abstractmethod
    def __call__(self, sample: dict) -> dict:
        raise Exception("Abstract class method")


class ComplexProcessor(BaseProcessor):
    """
       Apply processors for sample sequentially
   """
    def __init__(self, processors: List[BaseProcessor]):
        self.processors = processors

    def __call__(self, sample: dict) -> dict:
        for processor in self.processors:
            sample = processor(sample)
        return sample


class NLTKSentenceDetectionProcessor(BaseProcessor):
    def __call__(self, sample: dict) -> dict:
        """
        :param sample: requires fields: text
        :return: sample with text_sentences
        """
        if 'text' not in sample:
            raise RequirementNotSatisfiedException('text')
        ret = {**sample}

        sentences = nltk.sent_tokenize(ret['text'], language='russian')
        ret['text_sentences'] = sentences
        return ret


class PavlovSentenceDetectionProcessor(BaseProcessor):
    def __call__(self, sample: dict) -> dict:
        """
        :param sample: requires fields: text
        :return: sample with text_sentences
        """
        if 'text' not in sample:
            raise RequirementNotSatisfiedException('text')
        ret = {**sample}
        sent_tokenizer = RuSentTokenizer()
        sentences = sent_tokenizer([ret['text']])[0]
        sentences = [x for x in sentences if x.strip()]
        ret['text_sentences'] = sentences
        # print(sentences)
        return ret


class NLTKTokenizingProcessor(BaseProcessor):
    def __call__(self, sample: dict) -> dict:
        """
        :param sample: requires fields: text_sentences, title
        :return: sample with title_tokens, text_tokens
        """
        if 'text_sentences' not in sample:
            raise RequirementNotSatisfiedException('text_sentences')
        ret = {**sample}
        if 'title' in ret:
            ret['title_tokens'] = nltk.word_tokenize(ret['title'], language='russian')
        tokens = [nltk.word_tokenize(sentence, language='russian') for sentence in ret['text_sentences']]
        ret['text_tokens'] = tokens
        return ret


class TagCleaningProcessor(BaseProcessor):
    def __call__(self, sample: dict) -> dict:
        """
        :param sample: requires fields: text
        :return: sample with text without <..> tags
        """
        if 'text' not in sample:
            raise RequirementNotSatisfiedException('text')
        ret = {**sample}

        clean_text = self._clean_tags(ret['text'])
        ret['text'] = clean_text.lstrip()  # There is always leading \n

        return ret

    def _clean_tags(self, raw_html: str) -> str:
        clean_re = re.compile('<.*?>')
        clean_text = re.sub(clean_re, '', raw_html)
        return clean_text


class PyMyStemLemmatizingProcessor(BaseProcessor):
    def __init__(self):
        self.mystem = Mystem()

    def __call__(self, sample: dict) -> dict:
        """
        :param sample: requires fields: title, text_sentences
        :return: sample with title_lemmas, text_lemmas
        """
        if 'text_sentences' not in sample:
            raise RequirementNotSatisfiedException('text_sentences')
        ret = {**sample}
        if 'title' in ret:
            title_lemmas = self.mystem.lemmatize(ret['title'])
            title_lemmas = self._remove_empty_lemmas([title_lemmas])[0]
            ret['title_lemmas'] = title_lemmas
        lemmas = [self.mystem.lemmatize(sentence) for sentence in ret['text_sentences']]
        lemmas = self._remove_empty_lemmas(lemmas)  # for some reason there are lemmas containing whitespaces only
        ret['text_lemmas'] = lemmas
        return ret

    def _remove_empty_lemmas(self, lemmas: List[List[str]]) -> List[List[str]]:
        ret = []
        for sentence in lemmas:
            ret.append([lemma for lemma in sentence if len(lemma.strip()) != 0])
        return ret


class SentenceRankingProcessor(BaseProcessor):

    def __call__(self, sample: dict) -> dict:
        """
        :param sample: requires fields: title_lemmas, text_lemmas
        :return: sample with sentence_ranks
        """
        if 'title_lemmas' not in sample:
            raise RequirementNotSatisfiedException('title_lemmas')
        elif 'text_lemmas' not in sample:
            raise RequirementNotSatisfiedException('text_lemmas')
        ret = {**sample}
        ranks = self._rank_sentences(ret['text_lemmas'], ret['title_lemmas'])
        ret['sentence_ranks'] = ranks
        return ret

    def _rank_sentences(self, text_lemmas: List[List[str]], title_lemmas: List[str]) -> List[float]:
        ranks = [self._count_collisions(sentence_lemmas, title_lemmas) for  sentence_lemmas in text_lemmas]
        return ranks

    def _count_collisions(self, sentence_lemmas: List[str], title_lemmas: List[str]) -> int:
        ret = 0
        for sent_lemma in sentence_lemmas:
            if sent_lemma in title_lemmas:
                ret += 1
        return ret


class FieldFilteringProcessor(BaseProcessor):
    def __init__(self, *, keep_fields: Optional[List[str]] = None, remove_fields: Optional[List[str]] = None):
        if keep_fields is None and remove_fields is None:
            raise Exception('Both arguments are None')

        self.keep_fields = keep_fields
        self.remove_fields = remove_fields

    def __call__(self, sample: dict) -> dict:
        """
        :param sample: any kind of sample
        :return: sample containing only field mentioned in keep_fields but not in remove fields
        """
        ret = {key: val for key, val in sample.items() if self._check_field(key)}
        return ret

    def _check_field(self, key):

        if self.remove_fields is not None and key in self.remove_fields:
            return False

        if self.keep_fields is None or key in self.keep_fields:
            return True
        return False


class Word2VecProcessor(BaseProcessor):
    def __init__(self, path):
        self.model = gensim.models.KeyedVectors.load(path)
        self.vec_size = self.model.vector_size

    def __call__(self, sample: dict) -> dict:
        """
        :param sample: requires fields: text_lemmas
        :return: sample with word_embeddings
        """
        if 'text_lemmas' not in sample:
            raise RequirementNotSatisfiedException('text_lemmas')
        ret = {**sample}
        vectors = self._get_word_vectors(ret['text_lemmas'])
        ret['word_embeddings'] = vectors
        return ret

    def _get_word_vectors(self, text_lemmas: List[List[str]]):
        vectors = []
        for sentence in text_lemmas:
            sent_vectors = []
            for lemma in sentence:
                if lemma in self.model.vocab:
                    sent_vectors.append(self.model[lemma])
                else:
                    sent_vectors.append(np.zeros(self.vec_size))
            vectors.append(sent_vectors)
        return vectors


class SentenceEmbeddingSumProcessor(BaseProcessor):

    def __call__(self, sample: dict) -> dict:
        """
        :param sample: requires fields: word_embeddings
        :return: sample with word_embeddings
        """
        if 'word_embeddings' not in sample:
            raise RequirementNotSatisfiedException('word_embeddings')
        ret = {**sample}
        vectors = [np.sum(x, axis=0) for x in ret['word_embeddings']]
        ret['sentence_embeddings'] = vectors
        return ret


class RankLabelingProcessor(BaseProcessor):

    def __call__(self, sample: dict) -> dict:
        """
        :param sample: requires fields: sentence_ranks
        :return: sample with rank_labels
        """
        if 'sentence_ranks' not in sample:
            raise RequirementNotSatisfiedException('sentence_ranks')
        ret = {**sample}
        rank_labels = ret['sentence_ranks']
        if not rank_labels:
            raise EmptySampleException()
        rank_labels = np.array(rank_labels)
        rank_labels = rank_labels == np.max(rank_labels)
        ret['rank_labels'] = rank_labels
        return ret