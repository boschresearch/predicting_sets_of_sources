# Prediction Methods for sets of sources
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

from collections import namedtuple

Result = namedtuple('Result', 'src_task, src_domain, tgt_task, tgt_domain, setting, value')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr
from enum import IntEnum, unique
import numpy as np

from abc import ABC, abstractmethod
from typing import Union, Dict, List

TRANSFER_THRESHOLD = 0.5


@unique
class Outcome(IntEnum):
    NEGATIVE = 0
    NEUTRAL = 1
    POSITIVE = 2


class AbstractRanking(ABC):
    TASKS = ['GUM', 'POS', 'TIME', 'NER', 'POS-to-NER', 'TIME-to-NER']

    def __init__(self):
        self.name = None

    @abstractmethod
    def get_ranking(self, task, domain, setting, return_values=False):
        pass

    @abstractmethod
    def get_domains(self, task, setting):
        pass

    def __str__(self):
        return f'Ranking[{self.name}]'

    def __call__(self, task, domain, setting, return_values=False):
        return self.get_ranking(task, domain, setting, return_values)


class Ranking(AbstractRanking):

    def __init__(self, results, name='unknown'):
        super().__init__()
        self.results = results
        self.rankings = {}
        self._process_results(results)
        self.name = name

    def get_ranking(self, task, domain, setting, return_values=False):
        if 'Full' in setting:
            r = self.rankings[(task, domain, 'Full')]
        elif 'Lim' in setting:
            r = self.rankings[(task, domain, 'Limited')]
        else:
            raise KeyError('Unknown Setting')
        if return_values:
            return r
        else:
            return [k for k, val in r]

    def get_domains(self, task, setting):
        domains = [d for t, d, s in self.rankings if t == task and s == setting]
        return domains

    def __str__(self):
        return f'Ranking[{self.name}]'

    def __call__(self, task, domain, setting, return_values=False):
        return self.get_ranking(task, domain, setting, return_values)

    def _process_results(self, results):
        for task in self.TASKS:
            results_full, results_limited = {}, {}
            for key in results:
                stask, sdomain, ttask, tdomain, setting = key

                for our_setting in ['Full-to-Full', 'Limited-to-Limited']:
                    if setting != our_setting:
                        continue
                    if (task == 'NER' and stask == 'NER' and ttask == 'NER') or \
                            (task == 'POS' and stask == 'POS' and ttask == 'POS') or \
                            (task == 'GUM' and stask == 'GUM' and ttask == 'GUM') or \
                            (task == 'TIME' and stask == 'TIME' and ttask == 'TIME') or \
                            (task == 'POS-to-NER' and stask == 'POS' and ttask == 'NER') or \
                            (task == 'TIME-to-NER' and stask == 'TIME' and ttask == 'NER'):
                        if tdomain not in results_full:
                            results_full[tdomain] = {}
                            results_limited[tdomain] = {}
                        if 'Full' in setting:
                            results_full[tdomain][sdomain] = results[key].value
                        else:
                            results_limited[tdomain][sdomain] = results[key].value

            for domain, r in results_limited.items():
                ranking = [(k, val) for k, val in sorted(r.items(), key=lambda x: x[1], reverse=True)]
                self.rankings[(task, domain, 'Limited')] = ranking

            for domain, r in results_full.items():
                ranking = [(k, val) for k, val in sorted(r.items(), key=lambda x: x[1], reverse=True)]
                self.rankings[(task, domain, 'Full')] = ranking

    @classmethod
    def from_file(cls, fname, normalize=False, reverse=False):
        results = {}
        with open(fname, 'r') as fin:
            content = fin.read().splitlines()
            has_score = 'Score' in content[0]
            min_val, max_val = 100000, -100000
            for line in content[1:]:
                if not line.strip():
                    continue
                if has_score:
                    stask, sdomain, ttask, tdomain, setting, _, value = line.split('\t')
                else:
                    stask, sdomain, ttask, tdomain, setting, value = line.split('\t')
                if stask == ttask and sdomain == tdomain:
                    continue
                val = 100 - float(value) if reverse else float(value)
                min_val, max_val = min(min_val, val), max(max_val, val)
                results[stask, sdomain, ttask, tdomain, setting] = float(val)
            for key in results:
                stask, sdomain, ttask, tdomain, setting = key
                val = results[key]
                if normalize:
                    val = (val - min_val) * (100 / (max_val - min_val))
                r = Result(stask, sdomain, ttask, tdomain, setting, val)
                results[stask, sdomain, ttask, tdomain, setting] = r
        return cls(results, fname)


class RandomRanking(Ranking):

    def __init__(self, results, name='random'):
        super().__init__(results, name)
        self.rg = np.random.Generator(np.random.PCG64())

    def __copy__(self):
        x = RandomRanking(self.results, self.name)
        return x

    def set_seed(self, seed: int = None):
        self.rg = np.random.Generator(np.random.PCG64(seed))
        self.name = f'Random Ranking ({seed})'
        return self

    def get_ranking(self, task, domain, setting, return_values=False):
        assert not return_values, 'Disabled for RandomRankings'
        if 'Full' in setting:
            r = self.rankings[(task, domain, 'Full')]
        elif 'Lim' in setting:
            r = self.rankings[(task, domain, 'Limited')]
        else:
            raise KeyError('Unknown Setting')
        self.rg.shuffle(r)
        return [k for k, val in r]


class MultiRanking(AbstractRanking):

    def __init__(self, rankings: List[AbstractRanking], name='unknown', k=60):
        super().__init__()
        self.rankings = rankings
        self.name = name
        self.k = k

    def get_ranking(self, task, domain, setting, return_values=False):
        assert not return_values, 'Disabled for MultiRankings'
        ranked_predictions = []
        for rank in self.rankings:
            ranked_predictions.append(rank(task, domain, setting))
        return MultiRanking.rank_fusion(ranked_predictions, self.k)

    def get_domains(self, task, setting):
        return self.rankings[0].get_domains(task, setting)

    @staticmethod
    def rank_fusion(rankings: List[List[str]], k: int = 60):
        """Computes the Combination using Reciprocal Rank Fusion (RFR)
           $RFRscore (d \in D) = \sum_{r\in R} \frac{1}{k+r(d)}$
           with $k=60$ for documents $D$ and rankings $R$"""
        assert len(rankings) >= 2
        documents = sorted(rankings[0])

        results = {}
        for doc in documents:
            r_sum = 0.0
            for ranking in rankings:
                r_sum += (1 / (k + ranking.index(doc)))
            results[doc] = r_sum

        return [k for k, val in sorted(results.items(), key=lambda x: x[1], reverse=True)]


class Predictor(ABC):

    @abstractmethod
    def get_predictions(self, task: str, domain: str, setting: str):
        # rankings: Union[Ranking, List[Ranking]],
        pass

    def __call__(self, task: str, domain: str, setting: str):
        return self.get_predictions(task, domain, setting)


class TopN(Predictor):
    def __init__(self, ranking: Ranking, n):
        self.ranking = ranking
        self.n = n

    def get_predictions(self, task: str, domain: str, setting: str):
        r = self.ranking(task, domain, setting)
        if self.n < 0:
            return r
        return r[:self.n]


class Top1(TopN):
    def __init__(self, ranking: Ranking):
        super().__init__(ranking, n=1)


class TopDynamic(Predictor):
    def __init__(self, ranking: Ranking, transfer_information: List[Result]):
        self.min_dist = self.find_minimal_distance(ranking, transfer_information)
        self.ranking = ranking

    def get_predictions(self, task: str, domain: str, setting: str):
        r = self.ranking(task, domain, setting, True)
        if self.min_dist is not None:
            return [k for k, dist in r if dist >= self.min_dist]
        return []

    @staticmethod
    def find_minimal_distance(ranking: Ranking, transfer_information: List[Result]):
        minimum = float('-inf')
        positive_transfer = []
        for info in transfer_information:
            task = info.tgt_task if info.tgt_task == info.src_task else info.src_task + '-to-' + info.tgt_task
            ranking_for_target = ranking(task, info.tgt_domain, info.setting, True)
            dist = [val for k, val in ranking_for_target if k == info.src_domain][0]
            score = info.value

            if score <= -TRANSFER_THRESHOLD:
                minimum = max(minimum, dist)
            else:
                positive_transfer.append(dist)
        suitable_distances = [dist for dist in positive_transfer if dist > minimum]
        return min(suitable_distances) if len(suitable_distances) > 0 else None


class ClassifierPredictor(Predictor):

    def get_predictions(self, task: str, domain: str, setting: str):
        distances = {}
        for rank in self.rankings:
            r = rank(task, domain, setting, True)
            for src_domain, dist in r:
                if src_domain not in distances:
                    distances[src_domain] = []
                distances[src_domain].append(dist)

        predictions = []
        for src_domain, X in distances.items():
            pred = self.classifier.predict([X])
            if pred == Outcome.POSITIVE:
                predictions.append(src_domain)
        return predictions

    def prepare_data(self, transfer_information: List[Result]):
        X, y = [], []
        for info in transfer_information:
            distances = []
            for rank in self.rankings:
                task = info.tgt_task if info.tgt_task == info.src_task else info.src_task + '-to-' + info.tgt_task
                ranking_for_target = rank(task, info.tgt_domain, info.setting, True)
                dist = [val for k, val in ranking_for_target if k == info.src_domain][0]
                distances.append(dist)

            score = info.value
            if score > TRANSFER_THRESHOLD:
                for _ in range(self.weights[Outcome.POSITIVE]):
                    X.append(distances)
                    y.append(Outcome.POSITIVE.value)
            elif score < -TRANSFER_THRESHOLD:
                for _ in range(self.weights[Outcome.NEGATIVE]):
                    X.append(distances)
                    y.append(Outcome.NEGATIVE.value)
            else:
                for _ in range(self.weights[Outcome.NEUTRAL]):
                    X.append(distances)
                    y.append(Outcome.NEUTRAL.value)
        assert len(X) == len(y)
        return X, y


class RegressionPredictor(Predictor):

    def get_predictions(self, task: str, domain: str, setting: str):
        distances = {}
        for rank in self.rankings:
            r = rank(task, domain, setting, True)
            for src_domain, dist in r:
                if src_domain not in distances:
                    distances[src_domain] = []
                distances[src_domain].append(dist)

        predictions = []
        for src_domain, X in distances.items():
            pred = self.classifier.predict([X])
            if pred >= TRANSFER_THRESHOLD: # Outcome.POSITIVE:
                predictions.append(src_domain)
        return predictions

    def prepare_data(self, transfer_information: List[Result]):
        X, y = [], []
        for info in transfer_information:
            distances = []
            for rank in self.rankings:
                task = info.tgt_task if info.tgt_task == info.src_task else info.src_task + '-to-' + info.tgt_task
                ranking_for_target = rank(task, info.tgt_domain, info.setting, True)
                dist = [val for k, val in ranking_for_target if k == info.src_domain][0]
                distances.append(dist)

            score = info.value
            X.append(distances)
            y.append(score)
        assert len(X) == len(y)
        return X, y


class NearestNeighbor(ClassifierPredictor):
    def __init__(self, rankings: Union[Ranking, List[Ranking]], transfer_information: List[Result],
                 k=3, weights=None):
        if weights is None:
            weights = {Outcome.NEGATIVE: 1, Outcome.NEUTRAL: 1, Outcome.POSITIVE: 1}
        elif isinstance(weights, List):
            assert len(weights) == 3
            weights = {Outcome.NEGATIVE: weights[0], Outcome.NEUTRAL: weights[1], Outcome.POSITIVE: weights[0]}
        if isinstance(rankings, Ranking):
            rankings = [rankings]

        self.k = k
        self.weights = weights
        self.rankings = rankings
        self.classifier = self.fit_classifier(transfer_information)

    def fit_classifier(self, transfer_information: List[Result]):
        X, y = self.prepare_data(transfer_information)
        classifier = KNeighborsClassifier(n_neighbors=self.k)
        classifier.fit(X, y)
        return classifier


class SVM_Pred(ClassifierPredictor):
    def __init__(self, rankings: Union[Ranking, List[Ranking]], transfer_information: List[Result],
                 weights=None):
        if weights is None:
            weights = {Outcome.NEGATIVE: 1, Outcome.NEUTRAL: 1, Outcome.POSITIVE: 1}
        elif isinstance(weights, List):
            assert len(weights) == 3
            weights = {Outcome.NEGATIVE: weights[0], Outcome.NEUTRAL: weights[1], Outcome.POSITIVE: weights[0]}
        if isinstance(rankings, Ranking):
            rankings = [rankings]

        self.weights = weights
        self.rankings = rankings
        self.classifier = self.fit_classifier(transfer_information)

    def fit_classifier(self, transfer_information: List[Result]):
        X, y = self.prepare_data(transfer_information)
        classifier = SVC()
        classifier.fit(X, y)
        return classifier


class Log_Pred(ClassifierPredictor):
    def __init__(self, rankings: Union[Ranking, List[Ranking]], transfer_information: List[Result],
                 weights=None):
        if weights is None:
            weights = {Outcome.NEGATIVE: 1, Outcome.NEUTRAL: 1, Outcome.POSITIVE: 1}
        elif isinstance(weights, List):
            assert len(weights) == 3
            weights = {Outcome.NEGATIVE: weights[0], Outcome.NEUTRAL: weights[1], Outcome.POSITIVE: weights[0]}
        if isinstance(rankings, Ranking):
            rankings = [rankings]

        self.weights = weights
        self.rankings = rankings
        self.classifier = self.fit_classifier(transfer_information)

    def fit_classifier(self, transfer_information: List[Result]):
        X, y = self.prepare_data(transfer_information)
        classifier = LogisticRegression()
        classifier.fit(X, y)
        return classifier


class SVM_Reg(RegressionPredictor):
    def __init__(self, rankings: Union[Ranking, List[Ranking]], transfer_information: List[Result]):
        if isinstance(rankings, Ranking):
            rankings = [rankings]

        self.rankings = rankings
        self.classifier = self.fit_classifier(transfer_information)

    def fit_classifier(self, transfer_information: List[Result]):
        X, y = self.prepare_data(transfer_information)
        classifier = SVR()
        classifier.fit(X, y)
        return classifier


class Lin_Reg(RegressionPredictor):
    def __init__(self, rankings: Union[Ranking, List[Ranking]], transfer_information: List[Result]):
        if isinstance(rankings, Ranking):
            rankings = [rankings]

        self.rankings = rankings
        self.classifier = self.fit_classifier(transfer_information)

    def fit_classifier(self, transfer_information: List[Result]):
        X, y = self.prepare_data(transfer_information)
        classifier = LinearRegression()
        classifier.fit(X, y)
        return classifier


def filter_results(results: List[Result], src_task: str, tgt_task: str, setting: str, exclude=None):
    if exclude is None:
        exclude = []
    tmp = []
    for obj in results:
        if obj.setting == 'Full-to-Limited' or setting not in obj.setting:
            continue
        if obj.src_task == src_task and obj.tgt_task == tgt_task:
            if obj.tgt_domain not in exclude and obj.src_domain not in exclude:
                tmp.append(obj)
    return tmp


def get_sources(results: List[Result], src_task: str, tgt_task: str, tgt_domain: str, setting: str):
    positive, negative, neutral = [], [], []
    for obj in results:
        if obj.setting == 'Full-to-Limited' or setting not in obj.setting:
            continue
        if obj.src_task == src_task and obj.tgt_task == tgt_task:
            if obj.tgt_domain == tgt_domain and obj.src_domain != tgt_domain:
                if obj.value >= TRANSFER_THRESHOLD:
                    positive.append(obj.src_domain)
                elif obj.value <= -TRANSFER_THRESHOLD:
                    negative.append(obj.src_domain)
                else:
                    neutral.append(obj.src_domain)
    return frozenset(negative), frozenset(neutral), frozenset(positive)


## ################# ##
## Evaluate Rankings ##
## ################# ##


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



def eval_ranking(performance, ranking, print_outputs=False):
    if print_outputs:
        print(f'Distance: {ranking.name}')
    latex_line = ranking.name + ' '
    for setting in ['Full']:
        a_rank, a_ndcg, a_corr, a_num = 0, 0, 0, 0
        for task in ranking.TASKS:
            s_rank, s_ndcg, s_corr, s_num = 0, 0, 0, 0
            a_num += 1

            for domain in ranking.get_domains(task, setting):
                real_ranking = performance(task, domain, setting)
                pred_ranking = ranking(task, domain, setting)

                s_rank += eval_average_rank(real_ranking, pred_ranking)
                s_ndcg += eval_ndcg(real_ranking, pred_ranking)
                s_corr += eval_spearman(real_ranking, pred_ranking)
                s_num += 1

            s_rank /= s_num
            s_ndcg /= s_num
            s_corr /= s_num

            a_rank += s_rank
            a_ndcg += s_ndcg
            a_corr += s_corr

            if print_outputs:
                print(f'Task: {task:10s}\tRank: {round(s_rank, 1)}\t'
                      f'NDCG: {round(s_ndcg * 100, 1)}\tCorr: {round(s_corr, 3)}')
            latex_line += f'& {round(s_rank,1)} & {round(s_ndcg * 100, 1)} '

        a_rank /= a_num
        a_ndcg /= a_num
        a_corr /= a_num
        if print_outputs:
            print(f'{"":16s}\tRank: {round(a_rank, 1)}\tNDCG: {round(a_ndcg * 100, 1)}\tCorr: {round(a_corr, 3)}')
            print()
        latex_line += '\\\\'
        print(latex_line)
    return a_rank, a_ndcg, a_corr

