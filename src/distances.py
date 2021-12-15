# Domain distances/Similarity measures used in the paper
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

from collections import Counter
from typing import Dict, List

import numpy as np

from nltk.lm import KneserNeyInterpolated
from nltk.lm.preprocessing import padded_everygram_pipeline

import scipy
from scipy.spatial import procrustes
from scipy.stats import spearmanr

from sklearn.metrics import ndcg_score

from src.model import BertModel
from src.dataset import Dataset

##
## Corpus-based similarity measures
##

def distance_SIZE(src: Dataset, tgt: Dataset, count_tokens=False):
    """Returns the sizes of the source datasets"""
    if count_tokens:
        src_size = len([t for sent in src.tokens for t in sent])
    else:
        src_size = len(src.tokens)
    return src_size

def distance_TVC(src: Dataset, tgt: Dataset, min_occurences=0, max_occurences=float('inf')):
    """Computes the Target Vocabulary Overlap (TVC)"""
    if (src.name, len(src.tokens)) in cache_vocab:
        vocab_src = cache_vocab[(src.name, len(src.tokens))]
    else:
        count_src = Counter(src.get_flat_tokens())
        vocab_src = set([x for x,c in count_src.most_common() if min_occurences<=c<max_occurences])
        cache_vocab[(src.name, len(src.tokens))] = vocab_src
        
    if (tgt.name, len(tgt.tokens)) in cache_vocab:
        vocab_tgt = cache_vocab[(tgt.name, len(tgt.tokens))]
    else:
        count_tgt = Counter(tgt.get_flat_tokens())
        vocab_tgt = set([x for x,c in count_tgt.most_common() if min_occurences<=c<max_occurences])
        cache_vocab[(tgt.name, len(tgt.tokens))] = vocab_tgt
    overlap = vocab_src.intersection(vocab_tgt)
    
    distance = round(100 * len(overlap) / len(vocab_tgt), 2)
    return distance

def distance_TVeC(src: Dataset, tgt: Dataset, min_occurences=0, max_occurences=float('inf')):
    """Computes the Target Vocabulary Overlap (TVC) for entities only"""
    if (src.name, len(src.tokens)) in cache_vocab_entities:
        vocab_src = cache_vocab_entities[(src.name, len(src.tokens))]
    else:
        count_src = Counter([t for t, l in src.get_flat_tokens(add_labels=True) if l != 'O'])
        vocab_src = set([x for x,c in count_src.most_common() if min_occurences<=c<max_occurences])
        cache_vocab_entities[(src.name, len(src.tokens))] = vocab_src
        
    if (tgt.name, len(tgt.tokens)) in cache_vocab_entities:
        vocab_tgt = cache_vocab_entities[(tgt.name, len(tgt.tokens))]
    else:
        count_tgt = Counter([t for t, l in tgt.get_flat_tokens(add_labels=True) if l != 'O'])
        vocab_tgt = set([x for x,c in count_tgt.most_common() if min_occurences<=c<max_occurences])
        cache_vocab_entities[(tgt.name, len(tgt.tokens))] = vocab_tgt
        
    overlap = vocab_src.intersection(vocab_tgt)
    distance = round(100 * len(overlap) / len(vocab_tgt), 2)
    return distance

def _get_term_distributions(sentences, term2idx, n):
    term_dist = np.zeros(len(term2idx))
    for sentence in sentences:
        sentence = ["<bos>"] * (n - 1) + sentence + ["<eos>"] * (n - 1)
        for i in range(len(sentence) - n + 1):
            term = tuple(sentence[i:i + n])
            term_dist[term2idx[term]] += 1

    term_dist /= np.sum(term_dist)
    return term_dist

def _jensen_shannon_divergence(dist1, dist2):
    avg = 0.5 * (dist1 + dist2)
    sim = 1 - 0.5 * (scipy.stats.entropy(dist1, avg) + scipy.stats.entropy(dist2, avg))
    if np.isinf(sim): return 0
    return sim

def distance_JSD(src, tgt, n=3):
    """Computes Jensen-Shannon-Divergence (JSD) distance; called term distribution in the paper"""
    term2idx = {}
    for sentence in src.tokens + tgt.tokens:
        sentence = ["<bos>"] * (n - 1) + sentence + ["<eos>"] * (n - 1)
        for i in range(len(sentence) - n + 1):
            term = tuple(sentence[i:i + n])
            if term not in term2idx:
                term2idx[term] = len(term2idx)
                
    if (src.name, len(src.tokens), tgt.name, len(tgt.tokens)) in cache_term_distributions:
        src_terms, tgt_terms = cache_term_distributions[(src.name, len(src.tokens), tgt.name, len(tgt.tokens))]
    elif (tgt.name, len(tgt.tokens), src.name, len(src.tokens)) in cache_term_distributions:
        tgt_terms, src_terms = cache_term_distributions[(tgt.name, len(tgt.tokens), src.name, len(src.tokens))]
    else:  
        src_terms = _get_term_distributions(src.tokens, term2idx, n)
        tgt_terms = _get_term_distributions(tgt.tokens, term2idx, n)
        cache_term_distributions[(src.name, len(src.tokens), tgt.name, len(tgt.tokens))] = src_terms, tgt_terms
    similarity = _jensen_shannon_divergence(src_terms, tgt_terms)
    return similarity

##
## Model-based similarity measures
##

def _train_lm_model(src, n):
    if (src.name, len(src.tokens), n) in cache_lm_models:
        return cache_lm_models[(src.name, len(src.tokens), n)]
    src_data, src_padded_sents = padded_everygram_pipeline(n, src.get_flat_tokens())

    model = KneserNeyInterpolated(n) 
    model.fit(src_data, src_padded_sents)

    src_data, src_padded_sents = padded_everygram_pipeline(n, src.get_flat_tokens())
    src_ppl = model.perplexity(src_padded_sents)
    cache_lm_models[(src.name, len(src.tokens), n)] = model, src_ppl
    return model, src_ppl
    
def distance_PPL(src, tgt, n=5):
    """Computes the perpelxity between two domain models on the target dataset"""
    src_copy = src.__copy__()
    src_copy.tokens = [[t.replace('<', '-').replace('>', '-') for t in sent] for sent in src.tokens]
    tgt_copy = tgt.__copy__()
    tgt_copy.tokens = [[t.replace('<', '-').replace('>', '-') for t in sent] for sent in tgt.tokens]
    
    model, src_ppl = _train_lm_model(src_copy, n)
    tgt_data, tgt_padded_sents = padded_everygram_pipeline(n, tgt_copy.get_flat_tokens())
    tgt_ppl = model.perplexity(tgt_padded_sents)
    
    tgt_model, tgt_ref_ppl = _train_lm_model(tgt_copy, n)
    
    distance = tgt_ppl - tgt_ref_ppl
    return distance

def extract_features(model, dataset, average=True, limit=30000):
    if isinstance(model, str):
        model = BertModel.load_model(model).cpu()
        
    features = []
    for sent in dataset.tokens:
        inputs = model.tokenizer(' '.join(sent), return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)
        layers = outputs[1][0]
        for vectors in layers:
            for i, token in enumerate(vectors.detach()):
                features.append(token.numpy())
    features =  np.asarray(features)
    
    if average:
        avg = np.zeros(768, dtype='float32')
        for vec in features:
            avg += vec
        avg /= len(features)
        return avg
    else:
        return features[0:limit]

def cosine_similarity(vA, vB):
    sim = np.dot(vA, vB) / (np.linalg.norm(vA) * np.linalg.norm(vB))
    return sim

def rank_fusion(rankings, k=60):
    assert len(rankings) >= 2
    documents = sorted(rankings[0])
    
    results = {}
    for doc in documents:
        r_sum = 0.0
        for ranking in rankings:
            r_sum += (1 / (k + ranking.index(doc)))
        results[doc] = r_sum
            
    return [k for k, val in sorted(results.items(), key=lambda x: x[1], reverse=True)]

def distance_text_emb(bert_model, src: Dataset, tgt: Dataset):
    """Computes the text embedding difference between two datasets"""
    if isinstance(bert_model, str):
        bert_model = BertModel.load_model(bert_model).cpu()
    features_src = extract_features(bert_model, src)
    features_tgt = extract_features(bert_model, tgt)
    distance = cosine_similarity(features_src, features_tgt)
    return distance

def distance_WVV(src_model, tgt_model, tgt: Dataset):
    """Computes the word vector variance (WVV) taken from two BERT models"""
    if isinstance(src_model, str):
        src_model = BertModel.load_model(src_model).cpu()
    if isinstance(tgt_model, str):
        tgt_model = BertModel.load_model(tgt_model).cpu()
    features_src = extract_features(src_model, tgt)
    features_tgt = extract_features(tgt_model, tgt)
    distance = cosine_similarity(features_src, features_tgt)
    return distance

I = np.eye(768, 768)
def dist_to_matrix(matrix, other_matrix=I):
    return np.linalg.norm(matrix-other_matrix)

def distance_procrustes(src_model, tgt_model, tgt: Dataset):
    """Computes our model similarity measure based on the procrustes method"""
    if isinstance(src_model, str):
        src_model = BertModel.load_model(src_model).cpu()
    if isinstance(tgt_model, str):
        tgt_model = BertModel.load_model(tgt_model).cpu()
    features_src = extract_features(src_model, tgt, average=False)
    features_tgt = extract_features(tgt_model, tgt, average=False)
    p_com = procrustes(features_src, features_tgt)[2]
    distance = dist_to_matrix(p_com)
    return distance

##
## Evaluation methods
##

def get_ranking(sources: Dict[str, Dataset], tgt: Dataset, distance, **kwargs):
    results = {}
    for source_name, src in tqdm(sources.items()):
        if src.name == tgt.name:
            continue
        try:
            results[source_name] = distance(src, tgt, kwargs)
        except:
            results[source_name] = distance(src, tgt)
    return [k for k, val in sorted(results.items(), key=lambda x: x[1], reverse=True)]

def eval_spearman(real_ranking, pred_ranking):
    documents = sorted(real_ranking)
    real = [len(documents) - real_ranking.index(doc) for doc in documents]
    pred = [len(documents) - pred_ranking.index(doc) for doc in documents]
    score = spearmanr(real, pred)
    return score.correlation

def eval_ndcg(real_ranking, pred_ranking):
    documents = sorted(real_ranking)
    real = [len(documents) - real_ranking.index(doc) for doc in documents]
    pred = [len(documents) - pred_ranking.index(doc) for doc in documents]
    score = ndcg_score([real], [pred], ignore_ties=True)
    return score

def eval_average_rank(real_ranking, pred_ranking):
    best_method = real_ranking[0]
    return pred_ranking.index(best_method) + 1