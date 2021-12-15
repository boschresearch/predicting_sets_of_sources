# DataLoader methods for combining and loading corpora
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

from typing import Dict, List, Union

from src.dataset import Dataset

DATA_PATH = 'data/'


def load_file(data_type: str, file_name: str) -> List[Dataset]:
    data_type, file_name = data_type.lower(), file_name.lower()
    
    if data_type == 'ner' or data_type == 'n':
        dataloader = NerDataloader()
    elif data_type == 'gum' or data_type == 'g':
        dataloader = GumDataloader()
    elif data_type == 'pos' or data_type == 'p':
        dataloader = PosDataloader()
    elif data_type == 'time' or data_type == 't':
        dataloader = TimeDataloader()
    else:
        raise ValueError('Unknown data type: ' + data_type)
    return dataloader.load_file(file_name)


COMBINED_CORPORA = ['gum', 'ace']
def load_and_split_file(task: str, dataset: str, limit=-1, max_seq_len=-1, dev_pct=0.2, test_pct=0.3) -> List[Dataset]:
    assert dev_pct + test_pct < 1.0
    
    if dataset.lower() in COMBINED_CORPORA:
        train, dev, test = load_combined_corpus(
            task, dataset, limit, max_seq_len
        )
        return [train, dev, test]
        
    data = load_file(task, dataset)
    if len(data) == 3:  # train, dev and test split given
        train, dev, test = data
    elif len(data) == 2:  # only train and test split given
        train, dev = data[0].split(1-dev_pct)
        train.name += '/train'
        dev.name += '/dev'
        train, dev, test = [train, dev, data[1]]
    elif len(data) == 1:  # only train split given
        train, test = data[0].split(1-test_pct)
        dev = train.split(1-dev_pct, inplace=True)
        train.name += '/train'
        test.name += '/test'
        dev.name += '/dev'
    else:
        raise ValueError('Wrong file format for: ' + file_name)
       
    if max_seq_len > 0:
        train.cut_sentences(max_seq_len)
        dev.cut_sentences(max_seq_len)
        test.cut_sentences(max_seq_len)
    train.limit(limit)
    dev.limit(limit)
    return [train, dev, test]
        
        
def load_combined_corpus(task, name, limit=-1, max_seq_len=-1):
    if name.lower() == 'ace':
        dataloader = TimeDataloader()
        domains = [dom for dom, paths in dataloader.FILES.items() if 'ACE' in paths[0]]
        task = 'TIME'
    elif name.lower() == 'gum' and task.lower() == 'pos':
        dataloader = PosDataloader()
        domains = list(GumDataloader.FILES.keys())
        task = 'POS'
    elif name.lower() == 'gum':
        dataloader = GumDataloader()
        domains = list(GumDataloader.FILES.keys())
        task = 'GUM'
    # Prevent infinity-loops to load_and_split_file call 
    assert len([c for c in COMBINED_CORPORA if c in domains]) == 0
    
    if limit > 0:
        prev_limit = limit
        limit = int(limit / len(domains)) # use equal amounts of files from different domains
        print(f'Use {limit} sentences per domain. ({limit*len(domains)}/{prev_limit})')
        
    # Load first domain
    train, dev, test = load_and_split_file(task, domains[0])
    if max_seq_len > 0:
        train.cut_sentences(max_seq_len)
        dev.cut_sentences(max_seq_len)
        test.cut_sentences(max_seq_len)
    train.limit(limit)
    dev.limit(limit)
    
    # Add remaining domains
    for domain in domains[1:]:
        p_train, p_dev, p_test = load_and_split_file(task, domain)
        if max_seq_len > 0:
            p_train.cut_sentences(max_seq_len)
            p_dev.cut_sentences(max_seq_len)
            p_test.cut_sentences(max_seq_len)
        p_train.limit(limit)
        p_dev.limit(limit)
        train += p_train
        dev += p_dev
        test += p_test
        
    ds_name = task.upper() + '/' + name.upper() + '/'
    train.name = ds_name + 'train'
    dev.name = ds_name + 'dev'
    test.name = ds_name + 'test'
    return [train, dev, test]


class Dataloader:
    # def __init__(self, file_dict: Dict[str:List[str]], label_list: Union[List[str], None] = None):
    def __init__(self, file_dict, label_list=None):
        self.file_dict = file_dict
        self.label_list = label_list

    def load_file(self, name: str) -> List[Dataset]:
        file_names = []
        if name in self.file_dict:
            file_names = self.file_dict[name]
        else:
            for corpus_name in self.file_dict:
                if corpus_name[0].lower() == name[0].lower():
                    file_names = self.file_dict[corpus_name]
        if len(file_names) == 0:
            raise ValueError('Unknown filename: ' + name)
        datasets = [Dataset.read_conll_file(x, label_list=self.label_list) for x in file_names]
        all_labels = []
        for d in datasets:
            all_labels.extend(d.label_list)
        for d in datasets:
            d.set_tag_map(all_labels)
        return datasets


class NerDataloader(Dataloader):
    FILES = {  # <train_file, dev_file, test_file>
        "news": [DATA_PATH + 'NER/CoNLL/train.bio',
                 DATA_PATH + 'NER/CoNLL/dev.bio',
                 DATA_PATH + 'NER/CoNLL/test.bio'],
        "wetlab": [DATA_PATH + 'NER/WNUT-20/train.bio',
                   DATA_PATH + 'NER/WNUT-20/dev.bio',
                   DATA_PATH + 'NER/WNUT-20/test.bio'],
        "social": [DATA_PATH + 'NER/WNUT-17/train.bio',
                   DATA_PATH + 'NER/WNUT-17/dev.bio',
                   DATA_PATH + 'NER/WNUT-17/test.bio'],
        "twitter": [DATA_PATH + 'NER/WNUT-16/train.bio',
                    DATA_PATH + 'NER/WNUT-16/dev.bio',
                    DATA_PATH + 'NER/WNUT-16/test.bio'],
        "privacy": [DATA_PATH + 'NER/i2b2-2014/train.bio',
                    DATA_PATH + 'NER/i2b2-2014/dev.bio',
                    DATA_PATH + 'NER/i2b2-2014/test.bio'],
        "clinical": [DATA_PATH + 'NER/i2b2-2010/train.bio',
                     DATA_PATH + 'NER/i2b2-2010/test.bio'],
        "financial": [DATA_PATH + 'NER/SEC/SEC.bio'],
        "literature": [DATA_PATH + 'NER/LitBank/LitBank.bio'],
        "materials": [DATA_PATH + 'NER/MatKB/MatKB.bio'],
        #"zmodal": ['../../data/corpus_sentences_with_modals.txt'],
        #"zcysec": [DATA_PATH + 'NER/Cysec/Cysec.bio'],
        #"zpatents": [DATA_PATH + 'NER/CEMP/cemp-train.bio', 
        #             DATA_PATH + 'NER/CEMP/cemp-dev-3.bio', 
        #             DATA_PATH + 'NER/CEMP/cemp-dev-3.bio'],
    }

    def __init__(self):
        super(NerDataloader, self).__init__(self.FILES)


class GumDataloader(Dataloader):
    FILES = {  # <train_file, dev_file, test_file>
        "biography": [DATA_PATH + "NER/GUM/biography.bio"],
        "interview": [DATA_PATH + "NER/GUM/interview.bio"],
        "academic": [DATA_PATH + "NER/GUM/academic.bio"],
        "fiction": [DATA_PATH + "NER/GUM/fiction.bio"],
        "reddit": [DATA_PATH + "NER/GUM/reddit.bio"],
        "voyage": [DATA_PATH + "NER/GUM/voyage.bio"],
        "wikihow": [DATA_PATH + "NER/GUM/whow.bio"],
        "news": [DATA_PATH + "NER/GUM/news.bio"],
    }

    LABELS = [
        "O",
        "B-person", "I-person",
        "B-place", "I-place",
        "B-organization", "I-organization",
        "B-object", "I-object",
        "B-event", "I-event",
        "B-time", "I-time",
        "B-substance", "I-substance",
        "B-animal", "I-animal",
        "B-plant", "I-plant",
        "B-abstract", "I-abstract",
        "B-quantity", "I-quantity"
    ]

    def __init__(self):
        super(GumDataloader, self).__init__(self.FILES, self.LABELS)


class PosDataloader(Dataloader):
    FILES = {  # <train_file, dev_file, test_file>
        "academic": [DATA_PATH + "POS/GUM/academic.bio"],
        "biography": [DATA_PATH + "POS/GUM/biography.bio"],
        "fiction": [DATA_PATH + "POS/GUM/fiction.bio"],
        "interview": [DATA_PATH + "POS/GUM/interview.bio"],
        "news": [DATA_PATH + "POS/GUM/news.bio"],
        "reddit": [DATA_PATH + "POS/GUM/reddit.bio"],
        "voyage": [DATA_PATH + "POS/GUM/voyage.bio"],
        "wikihow": [DATA_PATH + "POS/GUM/whow.bio"],
#        "gum": [DATA_PATH + "POS/UD_English-GUM/train.bio",
#                DATA_PATH + "POS/UD_English-GUM/dev.bio",
#                DATA_PATH + "POS/UD_English-GUM/test.bio"],
#        "essays": [DATA_PATH + "POS/UD_English-ESL/train.bio",
#                   DATA_PATH + "POS/UD_English-ESL/dev.bio",
#                   DATA_PATH + "POS/UD_English-ESL/test.bio"],
        "general": [DATA_PATH + "POS/UD_English-ParTUT/train.bio",
                    DATA_PATH + "POS/UD_English-ParTUT/dev.bio",
                    DATA_PATH + "POS/UD_English-ParTUT/test.bio"],
        "online": [DATA_PATH + "POS/UD_English-EWT/train.bio",
                     DATA_PATH + "POS/UD_English-EWT/dev.bio",
                     DATA_PATH + "POS/UD_English-EWT/test.bio"],
        "literature": [DATA_PATH + "POS/UD_English-LinES/train.bio",
                       DATA_PATH + "POS/UD_English-LinES/dev.bio",
                       DATA_PATH + "POS/UD_English-LinES/test.bio"],
    }

    LABELS = [
        "ADJ", "ADP", "ADV", "AUX", "CCONJ",
        "DET", "INTJ", "NOUN", "NUM", "PART",
        "PRON", "PROPN", "PUNCT", "SCONJ",
        "SYM", "VERB", "X"
    ]

    def __init__(self):
        super(PosDataloader, self).__init__(self.FILES, self.LABELS)


class TimeDataloader(Dataloader):
    FILES = {  # <train_file, dev_file, test_file>
        "discussion": [DATA_PATH + "TIME/ACE05/broadcast_conversations.bio"],
        "telephony": [DATA_PATH + "TIME/ACE05/conversational_telephony.bio"],
        "broadcast": [DATA_PATH + "TIME/ACE05/broadcast_news.bio"],
        "newswire": [DATA_PATH + "TIME/ACE05/newswire.bio"],
        "usenet": [DATA_PATH + "TIME/ACE05/usenet.bio"],
        "online": [DATA_PATH + "TIME/ACE05/webblog.bio"],
        "historical": [DATA_PATH + "TIME/AncientTimes/AncientTimes.bio"],
        "clinical": [DATA_PATH + "TIME/i2b2-2012/train.bio",
                     DATA_PATH + "TIME/i2b2-2012/test.bio"],
        "general": [DATA_PATH + "TIME/TempEval3/TempEval3-train.bio",
                    DATA_PATH + "TIME/TempEval3/TempEval3-test.bio"],
        "aquaint": [DATA_PATH + "TIME/TempEval3/Aquaint.bio"],
        "wiki": [DATA_PATH + "TIME/WikiWars/WikiWars.bio"],
        "pubmed": [DATA_PATH + "TIME/Time4X/Time4SCI.bio"],
        "sms": [DATA_PATH + "TIME/Time4X/Time4SMS.bio"],
    }

    LABELS = [
        "O",
        "B-TIME", "I-TIME",
        "B-DATE", "I-DATE",
        "B-SET", "I-SET",
        "B-DURATION", "I-DURATION",
    ]

    def __init__(self):
        super(TimeDataloader, self).__init__(self.FILES, self.LABELS)
