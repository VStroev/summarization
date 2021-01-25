import gzip
import json
from typing import List, Generator

from torch.utils.data import IterableDataset
from data.processors import EmptySampleException, BaseProcessor

# TODO: Need tests


class NewsJsonDataset(IterableDataset):
    def __init__(self, data_path: str, clean_html: bool = True, gziped: bool = True):
        self.data_path = data_path
        self.clean_html = clean_html
        self.open_fn = gzip.open if gziped else open
        self.gziped = gziped

    def __iter__(self):
        open_fn = gzip.open if self.gziped else open
        with open_fn(self.data_path, 'r') as f:
            for line in f:
                line = line.decode() if self.gziped else line
                sample = json.loads(line)
                yield sample


class PreprocessingDataset(IterableDataset):
    def __init__(self, base_dataset: IterableDataset, processor: BaseProcessor):
        """
        :param base_dataset: wrapped dataset instance
        :param processor: list of processors. NOTE: order is necessary for dependencies
        """
        self.base_dataset = base_dataset
        self.processor = processor

    def __iter__(self):
        for sample in self.base_dataset:
            try:
                processed_sample = self.processor(sample)
            except EmptySampleException:
                continue
            yield processed_sample


class FieldUnstackingDataset(IterableDataset):
    """
    Unwraps values foe mentioned sample's field for batch collation
    """
    def __init__(self, base_dataset: IterableDataset, fields: List[str]):
        """
        :param base_dataset: wrapped dataset instance
        :param fields: list of fields to unwrap
        """
        self.base_dataset = base_dataset
        self.fields = fields

    def __iter__(self):
        for sample in self.base_dataset:
            ret_fields = [list(self._unwrap_field(sample, field)) for field in self.fields]

            yield from zip(*ret_fields)

    def _unwrap_field(self, sample: dict, field: str) -> Generator:
        for val in sample[field]:
            yield val
