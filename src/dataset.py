# Dataset class with several functions for merging corpora
# Copyright (c) 2021 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import sys
from typing import List, Union


class Dataset:
    def __init__(self, tokens: List[List[str]], labels: List[List[str]],
                 name: str = 'dataset', label_list: Union[List[str], None] = None):
        assert len(tokens) == len(labels), 'The number of words and labels has to be equal'
        self.tokens = tokens
        self.labels = labels
        self.name = name

        self.label_list, self.label2idx, self.idx2label = [], [], []
        self.set_tag_map(label_list)
        
    def get_flat_tokens(self, add_labels=False):
        tokens = [t for sent in self.tokens for t in sent]
        if add_labels:
            labels = [l for sent in self.labels for l in sent]
            zipped = [(t, l) for t, l in zip(tokens, labels)]
            return zipped
        else:
            return tokens
        
    def num_tokens(self):
        return sum([len(t) for t in self.tokens])

    def sentences(self):
        return [' '.join(words) for words in self.tokens]

    def cut_sentences(self, maxlen):
        new_tokens, new_labels = [], []
        for sent, tags in zip(self.tokens, self.labels):
            while len(sent) > maxlen:
                new_tokens.append(sent[:maxlen])
                new_labels.append(tags[:maxlen])
                sent, tags = sent[maxlen:], tags[maxlen:]
            new_tokens.append(sent)
            new_labels.append(tags)
        self.tokens = new_tokens
        self.labels = new_labels

    def set_tag_map(self, label_list: Union[List[str], None] = None):
        self.label_list, self.label2idx, self.idx2label = self.get_tag_map(label_list)

    def get_tag_map(self, label_list: Union[List[str], None] = None):
        if label_list is None:
            label_list = list(set([label for sent in self.labels for label in sent]))
        label_list.extend(["[SEP]", "[PAD]"])  # for BERT tokens
        # label_list.extend(["<START>", "<STOP>"])  # for CRF tokens TODO implement CRF
        for label in sorted(label_list):
            if label.startswith('B-'):
                label_list.append('I-' + label[2:])
        label_list = sorted(list(set(label_list)))
        label2idx = {l: i for i, l in enumerate(label_list)}
        idx2label = {v: k for k, v in label2idx.items()}
        label2idx['[PAD]'] = -100
        idx2label[-100] = ['[PAD]']
        return label_list, label2idx, idx2label

    def split(self, n, inplace=False):
        """Split the dataset into two
        n :argument - if n < 1: n is treated as a percentage value
                      if 1 > n > 0: n is treated as a limit
        inplace :argument - changes the original dataset
        """
        if n > 1:
            limit = n
        elif 0 < n < 1:
            limit = int(len(self.tokens) * n)
        else:
            raise ValueError('Invalid n: 0 < n < 1 or n > 1. n=' + str(n))
        
        second_dataset = self.__copy__()
        second_dataset.tokens = second_dataset.tokens[limit:]
        second_dataset.labels = second_dataset.labels[limit:]
        
        first_dataset = self if inplace else self.__copy__()
        first_dataset.tokens = first_dataset.tokens[:limit]
        first_dataset.labels = first_dataset.labels[:limit]
        
        if inplace:
            return second_dataset
        return first_dataset, second_dataset
    
    def limit(self, n):
        if n > 0:
            _ = self.split(n, inplace=True)
        return self

    def __copy__(self):
        return Dataset(
            [x for x in self.tokens],
            [y for y in self.labels],
            self.name,
            [l for l in self.label_list]
        )

    def __add__(self, other):
        new = self.__copy__()
        assert self.label_list == other.label_list, 'Can only add similar datasets'
        new.tokens.extend(other.tokens)
        new.labels.extend(other.labels)
        new.name += '-' + other.name
        return new

    def __str__(self):
        if len(self.tokens) > 5:
            s = f'Dataset({self.name},'
            s += f'sents={len(self.tokens)},'
            s += f'words={sum([len(x) for x in self.tokens])},'
            s += f'{self.tokens[0][0:10]})'
        else:
            s = f'Dataset({self.name},sentences={self.tokens})'
        return s

    def __repr__(self):
        return self.__str__()

    @classmethod
    def read_conll_file(cls, filename: str, encoding: str = 'utf-8', label_list=None):
        # words and labels are lists of lists, outer for sentences and
        # inner for the words/labels of each sentence.
        words, labels = [], []
        curr_words, curr_labels = [], []
        with open(filename, 'r', encoding=encoding) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    fields = line.split('\t')
                    if len(fields) > 1:
                        curr_words.append(fields[0])
                        curr_labels.append(fields[-1])
                    else:
                        print('ignoring line: {}'.format(line), file=sys.stderr)
                        print((filename, i), file=sys.stderr)
                        pass
                elif curr_words:
                    words.append(curr_words)
                    labels.append(curr_labels)
                    curr_words, curr_labels = [], []
        if curr_words:
            words.append(curr_words)
            labels.append(curr_labels)
        name = filename.split('/')[-3:]
        name = '/'.join(name)
        return Dataset(words, labels, name, label_list)
    
    def write_to_conll_file(self, filename, encoding: str = 'utf-8'):
        with open(filename, 'w', encoding=encoding) as f:
            for sid, sent in enumerate(self.tokens):
                for token, label in zip(sent, self.labels[sid]):
                    f.write(f'{token}\t{label}\n')
                f.write('\n')
                
        
        
        
