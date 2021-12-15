# BertModel class and the abstract Sequence Tagger. 
# The BertModel is a wrapper for the huggingface models.
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

import os
from collections import namedtuple
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertConfig, BertTokenizerFast, BertForTokenClassification

from src.dataset import Dataset
from src.embeddings import Embeddings

from abc import ABC, abstractmethod
from typing import Union, Dict, List
import json
 
class SequenceTagger(ABC, torch.nn.Module):
    DEFAULT_LR = 2e-2
 
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    @classmethod
    @abstractmethod
    def load_model(cls, model_path, dataset=None):
        pass
    
    @abstractmethod
    def save_model(self, model_path):
        pass
    
    @abstractmethod
    def swap_classification_head(self, dataset):
        pass
    
    @abstractmethod
    def convert_dataset(self, dataset):
        pass
    
    @abstractmethod
    def get_predictions(self, dataloader):
        pass

    def _eval_batch(self, batch, logits):
        prop = logits.cpu().detach().numpy()
        pred = np.argmax(prop, axis=-1)
        
        convert_word_ids = 'input_ids' in batch

        output = []
        for sid, sent in enumerate(batch['attention_mask']):
            cur_word, cur_gold, cur_pred = '', '', ''
            cur_sent = []

            for tid, mask in enumerate(sent):
                if mask == 1:
                    if convert_word_ids:
                        word_id = batch['input_ids'][sid][tid]
                        word = self.tokenizer.decode([word_id])
                        is_start = batch['offset_mapping'][sid][tid][0] == 0
                    else:
                        word = batch['tokens'][sid][tid]
                        is_start = True # only used for word-pieces

                    if is_start:
                        if cur_word not in ['', '[CLS]', '[SEP]']:
                            cur_sent.append((cur_word, cur_gold, cur_pred))
                        gold_label = self.config.id2label[batch['labels'][sid][tid].item()]
                        pred_label = self.config.id2label[pred[sid][tid]]
                        cur_word, cur_gold, cur_pred = word, gold_label, pred_label
                    else:
                        cur_word += word[2:] if word.startswith('##') else word
            if cur_word not in ['', '[CLS]', '[SEP]']:
                cur_sent.append((cur_word, cur_gold, cur_pred))
            output.append(cur_sent)
        #for sent in output:
        #    for word, gold, pred in sent:
        #        print(f'{word}\t{gold}\t{pred}')
        #    print()
        return output


class BertModel(BertForTokenClassification, SequenceTagger):
    DEFAULT_LR = 2e-5
    
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.init_weights()
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(device)
        
    def set_tokenizer(self, bert_path):
        self.bert_path = bert_path
        self.tokenizer = BertTokenizerFast.from_pretrained(bert_path, do_lower_case='uncased' in bert_path)
        
    def set_labels(self, dataset):
        self.config.label2id = dataset.label2idx
        self.config.id2label = dataset.idx2label
        self.num_labels = len(dataset.label2idx)
        
    def swap_classification_head(self, dataset):
        self.set_labels(dataset)
        linear = torch.nn.Linear(self.config.hidden_size, len(dataset.label2idx)).to(self.device)
        self.classifier = linear
            
    def _encode_words(self, dataset):
        encoded_words = self.tokenizer(
            dataset.tokens, 
            is_split_into_words=True, 
            return_offsets_mapping=True, 
            max_length=512,
            padding=True, 
            truncation=True,
        )
        return encoded_words
        
    def _encode_tags(self, dataset, encodings):
        labels = [[dataset.label2idx[tag] for tag in doc] for doc in dataset.labels]
        encoded_labels = []
        
        i = 0
        for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
            #print(doc_labels)
            #print(doc_offset)
            #print(encodings)
            # create an empty array of -100
            doc_enc_labels = np.ones(len(doc_offset),dtype=int) * -100
            arr_offset = np.array(doc_offset)
            
            # Fix error when tokenizer does truncations 
            # happens for strange sentences with many wordpieces: >= 512 for 128 tokens
            num_wordpieces = 0
            for start, end in doc_offset:
                if start == 0 and end != 0:
                    num_wordpieces += 1
            if num_wordpieces != len(doc_labels):
                print(f'Len labels: {len(doc_labels)}')
                doc_labels = doc_labels[:num_wordpieces]
                print(f'num_wordpieces: {num_wordpieces}')
                print(f'New labels: {len(doc_labels)}')

            # set labels whose first offset position is 0 and the second is not 0
            try:
                doc_enc_labels[(arr_offset[:,0] == 0) & (arr_offset[:,1] != 0)] = doc_labels
                i += 1
            except:
                print(encodings.input_ids[i])
            encoded_labels.append(doc_enc_labels.tolist())
        return encoded_labels
    
    def convert_dataset(self, dataset, batch_size, shuffle=True):
        word_encodings = self._encode_words(dataset)
        label_encodings = self._encode_tags(dataset, word_encodings)
        data = self.BertDataset(word_encodings, label_encodings)
        loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
        return loader
    
    def get_predictions(self, dataloader):
        conll_outputs = []
        avg_loss = 0
        if not isinstance(dataloader, torch.utils.data.DataLoader):
            dataloader = [dataloader]
        for batch in dataloader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            outputs = self(input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs
            avg_loss += loss.item()
            conll_outputs.extend(self._eval_batch(batch, logits))
        return conll_outputs, loss
    
    @staticmethod
    def _read_label_list(bert_path):
        config_path = bert_path + 'config.json'
        with open(config_path, 'r', encoding='utf-8') as fin:
            content = json.loads(fin.read())
        return content['label2id']
    
    @classmethod
    def load_model(cls, bert_path, dataset=None):
        try:
            if dataset is None:
                num_labels = len(cls._read_label_list(bert_path))
            else:
                num_labels = len(dataset.label_list)
            model = cls.from_pretrained(bert_path, num_labels=num_labels)
        except:
            print('Retrieve old number of labels')
            f = open(bert_path + 'config.json',)
            data = json.load(f)
            f.close()
            num_labels = len(data['label2id'])
            model = cls.from_pretrained(bert_path, num_labels=num_labels)
        model.set_tokenizer(bert_path)
        if dataset is not None:
            model.set_labels(dataset)
        return model
    
    def save_model(self, model_path):
        self.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
    
    class BertDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, labels):
            self.encodings = encodings
            self.labels = labels

        def __getitem__(self, idx):
            item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
            item['labels'] = torch.tensor(self.labels[idx])
            return item

        def __len__(self):
            return len(self.labels)
        