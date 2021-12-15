# Command-line arguments for main.py
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

import argparse

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--max_epochs', '-e', type=int, default=20, help='Number of maximum training epochs')
parser.add_argument('--early_stopping', type=int, default=20, help='Early stopping after given number without improvements')
parser.add_argument('--seed', '-s', type=int, default=123, help='Seed for random generator initialization')
parser.add_argument('--task', '-t', required=True, help='Task identifier as used in src/Dataloader.py')
parser.add_argument('--dataset', '-d', required=True, help='Dataset identifier as used in src/Dataloader.py')
parser.add_argument('--limit', type=int, default=-1, help='Limit train/dev/test sentences to n if n > 1')
parser.add_argument('--max_seq_len', type=int, default=128)

# storage parameters
parser.add_argument('--output_dir', '-o', default=None, 
                    help='Persist trained model to this directory. Model is not stored if not given')
parser.add_argument('--pretrained_model', '-p', default=None, 
                    help='Load trained model from this directory. Use new model if not given')

# model parameters
parser.add_argument('--batch_size', type=int, default=32, help='Batch size used during training')
parser.add_argument('--embed_base_path', type=str, default='path/to/local/bert/file/')
parser.add_argument('--bert_path', type=str, default='bert-base-cased/')
parser.add_argument('--tqdm', action="store_true", 
                    help='Uses tqdm bars to show progress during training')
parser.add_argument('--swap_heads', action="store_true", 
                    help='Swap model heads after loading. Useful when a different dataset is used.')

# training parameters
parser.add_argument('--learning_rate', '-lr', type=float, default=-1, 
                    help='Learning rate during training. If < 0: default learning rate of each model is used')
parser.add_argument('--use_cpu', action="store_true", help='Use only CPU for computations')

opt = parser.parse_args()

# automatically prepared options
if not torch.cuda.is_available() or opt.use_cpu:
    opt.device = 'cpu'
else:
    opt.device = 'cuda:0'

# Concatenate embedding paths
if not opt.embed_base_path.endswith('/'):
    opt.embed_base_path += '/'
opt.bert_path = opt.embed_base_path + opt.bert_path
