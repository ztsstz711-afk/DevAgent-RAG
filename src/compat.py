from __future__ import annotations

import math
import pickle
import re
from collections import Counter
from pathlib import Path


def dump_pickle(payload, path):
    with Path(path).open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


class SimpleTfidfVectorizer:
    def __init__(self, **_kwargs):
        self.vocabulary_: dict[str, int] = {}
        self.idf_: list[float] = []

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]+", text.lower())

    def fit_transform(self, texts):
        texts = list(texts)
        tokenized = [self._tokens(text) for text in texts]
        terms = sorted({term for tokens in tokenized for term in tokens})
        self.vocabulary_ = {term: index for index, term in enumerate(terms)}
        self.idf_ = [math.log((1 + len(texts)) / (1 + sum(term in tokens for tokens in tokenized))) + 1 for term in terms]
        return [self._vector(tokens) for tokens in tokenized]

    def transform(self, texts):
        return [self._vector(self._tokens(text)) for text in texts]

    def _vector(self, tokens):
        counts = Counter(tokens)
        vector = [0.0] * len(self.vocabulary_)
        for term, count in counts.items():
            if term in self.vocabulary_:
                index = self.vocabulary_[term]
                vector[index] = count * self.idf_[index]
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


def simple_cosine(query_matrix, document_matrix):
    query = query_matrix[0]
    return [[sum(a * b for a, b in zip(query, document)) for document in document_matrix]]


class SimpleCompiledGraph:
    def __init__(self, nodes, route):
        self.nodes = nodes
        self.route = route

    def invoke(self, state):
        state = dict(state)
        state.update(self.nodes["route_task"](state))
        if self.route(state) == "error":
            state.update(self.nodes["parse_error"](state))
        for name in ("retrieve_docs", "find_code_snippets", "generate_answer", "check_quality"):
            state.update(self.nodes[name](state))
        return state


class SimpleStateGraph:
    def __init__(self, _state_type):
        self.nodes = {}
        self.route = lambda state: state["route"]

    def add_node(self, name, function):
        self.nodes[name] = function

    def add_edge(self, *_args):
        return None

    def add_conditional_edges(self, _name, route, _mapping):
        self.route = route

    def compile(self):
        return SimpleCompiledGraph(self.nodes, self.route)


START = "__start__"
END = "__end__"
