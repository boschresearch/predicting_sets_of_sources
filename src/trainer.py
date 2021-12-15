# Trainer class for BertModel.
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

import torch
from transformers import AdamW

import numpy as np

from seqeval.metrics import classification_report
from seqeval.metrics import f1_score, recall_score, precision_score, accuracy_score

from tqdm import tqdm


def get_scores(output, verbose=0):
    is_pos = True  # POS labels have to be converted for seqeval functions
    y_true, y_pred = [], []
    for sent in output:
        y_true.append([])
        y_pred.append([])
        for word, gold, pred in sent:
            if pred[0] not in ['B', 'I', 'O']:
                is_pos = True
            y_true[-1].append(gold)
            y_pred[-1].append(pred)
    if is_pos:
        y_true = [['B-' + y if y != 'O' else 'O' for y in sent] for sent in y_true]
        y_pred = [['B-' + y if y != 'O' else 'O' for y in sent] for sent in y_pred]
    acc = accuracy_score(y_true, y_pred) * 100
    pre = precision_score(y_true, y_pred) * 100
    rec = recall_score(y_true, y_pred) * 100
    f1 = f1_score(y_true, y_pred) * 100
    cr = classification_report(y_true, y_pred)
    if verbose > 0:
        print(cr)
    return acc, pre, rec, f1, cr

class Trainer:
    def __init__(self, train=None, dev=None, test=None):
        self.train_data, self.test_data, self.dev_data = None, None, None
        self.set_corpus(train, dev, test)
        
    def set_corpus(self, train=None, dev=None, test=None):
        self.train_data = train
        self.test_data = test
        self.dev_data = dev
        
    class TrainingResults:
        def __init__(self, main_score='f1', batch_size=32):
            assert main_score in ['loss', 'accuracy', 'precision', 'recall', 'f1']
            self.metrics = {
                'train.loss': [],
                'train.accuracy': [],
                'train.precision': [],
                'train.recall': [],
                'train.f1': [],
                'dev.loss': [],
                'dev.accuracy': [],
                'dev.precision': [],
                'dev.recall': [],
                'dev.f1': [],
                'test.loss': [],
                'test.accuracy': [],
                'test.precision': [],
                'test.recall': [],
                'test.f1': [],
            }
            self.main_score = main_score
            
        def add_from_predictions(self, mode, loss, conll_output):
            assert mode in ['train', 'dev', 'test']
            self.metrics[mode + '.loss'].append(loss)
            acc, pre, rec, f1, _ = get_scores(conll_output)
            self.metrics[mode + '.accuracy'].append(acc)
            self.metrics[mode + '.precision'].append(pre)
            self.metrics[mode + '.recall'].append(rec)
            self.metrics[mode + '.f1'].append(f1)
            return self.metrics[mode + '.' + self.main_score][-1]
        
        def __str__(self):
            return str(self.metrics)
        
        def __repr__(self):
            return str(self.metrics)
            
            
    def evaluate(self, model, test_data, batch_size=32):
        tmp_train, tmp_dev, tmp_test = self.train_data, self.dev_data, self.test_data
        self.train_data, self.dev_data, self.test_data = None, None, test_data
        results, cr, _ = self.train(model, None, 0, batch_size=batch_size)  # skip training
        self.train_data, self.dev_data, self.test_data = tmp_train, tmp_dev, tmp_test
        return results
        
    def train(self, model, model_path=None, num_epochs=100, lr=-1,
              main_score='f1', batch_size=32, 
              early_stopping_epochs=5, revert_to_best_model=True, 
              shuffle=True, store_final_model=False, use_tqdm=False):
        if model_path is not None and not model_path.endswith('/'):
            model_path += '/'
            
        if self.train_data is not None:
            train_loader = model.convert_dataset(self.train_data, batch_size=batch_size, shuffle=shuffle)
        if self.test_data is not None:
            test_loader = model.convert_dataset(self.test_data, batch_size=batch_size, shuffle=shuffle)
        if self.dev_data is not None:
            dev_loader = model.convert_dataset(self.dev_data, batch_size=batch_size, shuffle=shuffle)
        
        if lr <= 0:
            lr = model.DEFAULT_LR
            print('Set learning rate to: ' + str(lr))
        optim = AdamW(model.parameters(), lr=lr)


        cr = None
        best_score = -1
        early_stopping = 0
        early_stopping_limit = num_epochs if early_stopping_epochs <= 0 else early_stopping_epochs
        training_results = self.TrainingResults(main_score, batch_size)

        # perform training
        if self.train_data is not None:
            for epoch in range(num_epochs):
                model.train()
                
                if use_tqdm:
                    t = tqdm(total=len(train_loader), desc=('Epoch: ' + str(epoch+1)))

                collected_loss, processed_batches = 0, 0
                conll_outputs = []
                for batch in train_loader:
                    optim.zero_grad()
                    predictions, loss = model.get_predictions(batch)
                    conll_outputs.extend(predictions)
                    collected_loss += loss.item()
                    processed_batches += len(batch["labels"])
                    avg_loss = collected_loss / processed_batches
                    loss.backward()
                    optim.step()
                    if use_tqdm:
                        t.set_postfix_str(f'Loss {round(avg_loss, 5)}')
                        t.update(1)
                training_results.add_from_predictions('train', avg_loss, conll_outputs)
                _, _, _, _, cr = get_scores(conll_outputs)
                if use_tqdm:
                    t.close()

                model.eval()
                # get validation F1
                if self.dev_data is not None:
                    dev_pred, dev_loss = model.get_predictions(dev_loader)
                    dev_loss = dev_loss.item() / len(self.dev_data.tokens)
                    dev_score = training_results.add_from_predictions('dev', dev_loss, dev_pred)
                    _, _, _, _, cr = get_scores(dev_pred)
                    if dev_score > best_score:
                        print(f'Dev score ({main_score}): {round(dev_score,2)} [new best]')
                        best_score = dev_score
                        early_stopping = 0
                        if model_path is not None:
                            model.save_model(model_path + 'best')
                    else:
                        early_stopping += 1
                        print(f'Dev score ({main_score}): {round(dev_score,2)} [{early_stopping} epoch(s) w/o improvement]')
                        if early_stopping >= early_stopping_limit:
                            print('Early stopping')
                            if model_path is not None:
                                model.save_model(model_path + 'best')
                            break
                if model_path is not None and store_final_model:
                    model.save_model(model_path + 'final')

            if revert_to_best_model and model_path is not None:
                print('Load best model for testing')
                model = model.__class__.load_model(model_path + 'best', self.train_data)

        # get test F1
        if self.test_data is not None:
            test_pred, test_loss = model.get_predictions(test_loader)
            test_loss = test_loss.item() / len(self.test_data.tokens)
            test_score = training_results.add_from_predictions('test', test_loss, test_pred)
            _, _, _, _, cr = get_scores(test_pred)
            print(f'Test score ({main_score}): {round(test_score,2)} [final result]')
            
        print()
        print(cr)
            
        if model_path is not None:
            with open(model_path + 'results.txt', 'w') as fout:
                for key, values in training_results.metrics.items():
                    fout.write(f'{key}\t{values}\n')
        
        return training_results, cr, model
