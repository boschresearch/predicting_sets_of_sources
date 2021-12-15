# Main class for training a sequence tagger
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
import numpy as np

from src.dataset import Dataset
from src.dataloader import load_and_split_file
from src.trainer import Trainer
from src.model import BertModel

from options import opt

def set_seeds(seed):
    print('Set seeds to ' + str(seed))
    np.random.seed(seed)
    #python_random.seed(seed)
    #tensorflow.random.set_random_seed(seed)
    torch.manual_seed(seed)

def prepare_data(opt):
    print('Load data')
    if ',' in opt.dataset:
        datasets = opt.dataset.split(',')
        print(f'Load multiple datasets: {datasets}')
        train, dev, test = load_and_split_file(opt.task, datasets[0], opt.limit, opt.max_seq_len)
        for d in datasets[1:]:
            t_train, t_dev, t_test = load_and_split_file(opt.task, d, opt.limit, opt.max_seq_len)
            train += t_train
            dev += t_dev
            test += t_test
        train.name = 'train/' + opt.dataset
        dev.name = 'dev/' + opt.dataset
        test.name = 'test/' + opt.dataset
    else:
        train, dev, test = load_and_split_file(opt.task, opt.dataset, opt.limit, opt.max_seq_len)
    return train, dev, test

def prepare_model(opt, dataset) -> BertModel:
    print('Load model')
    if opt.pretrained_model is not None:
        model = BertModel.load_model(opt.pretrained_model, dataset)
    else:
        model = BertModel.load_model(opt.bert_path, dataset)
    return model.to(opt.device)

def main():
    print('Options')
    print(opt)
    set_seeds(opt.seed)
    
    train, dev, test = prepare_data(opt)
    print(train)
    print(dev)
    print(test)
    model = prepare_model(opt, train)
    if opt.swap_heads:
        print('Swap classification head to new dataset')
        model.swap_classification_head(train)
    print(model)
    trainer = Trainer(train, dev, test)
    
    main_score = 'accuracy' if opt.task.lower() in ['p', 'pos'] else 'f1'
    results, _, model = trainer.train(
        model, opt.output_dir, num_epochs=opt.max_epochs, batch_size=int(opt.batch_size),
        lr=opt.learning_rate, early_stopping_epochs=opt.early_stopping, 
        main_score=main_score, use_tqdm=opt.tqdm
    )

if __name__ == '__main__':
    main()
