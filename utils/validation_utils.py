#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
 Set of utilities for Q&A results validation tasks - Retriver passage validation and Reader predicted answer validation
"""

import collections
import logging
import string
import unicodedata
import zlib
from functools import partial
from multiprocessing import Pool as ProcessPool
from typing import Tuple, List, Dict

from datasets import Dataset        

import regex as re

import copy
import logging

import regex
import spacy

logger = logging.getLogger(__name__)

QAMatchStats = collections.namedtuple("QAMatchStats", ["top_k_hits", "questions_doc_hits"])

QATableMatchStats = collections.namedtuple(
    "QAMatchStats", ["top_k_chunk_hits", "top_k_table_hits", "questions_doc_hits"]
)

TableChunk = collections.namedtuple("TableChunk", ["text", "title", "table_id"])

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def calculate_matches(
    wiki_data: Dataset,
    answers: List[List[str]],
    closest_docs: List[Tuple[List[object], List[float]]],   
    workers_num: int,       
    match_type: str,
) -> QAMatchStats:  
    """
    Evaluates answers presence in the set of documents. This function is supposed to be used with a large collection of
    documents and results. It internally forks multiple sub-processes for evaluation and then merges results
    :param all_docs: dictionary of the entire documents database. doc_id -> (doc_text, title)
    :param answers: list of answers's list. One list per question
    :param closest_docs: document ids of the top results along with their scores
    :param workers_num: amount of parallel threads to process data
    :param match_type: type of answer matching. Refer to has_answer code for available options
    :return: matching information tuple.
    top_k_hits - a list where the index is the amount of top documents retrieved and the value is the total amount of
    valid matches across an entire dataset.
    questions_doc_hits - more detailed info with answer matches for every question and every retrieved document
    """
    # logger.info("wiki_data size %d", len(wiki_data))
    global dpr_all_documents
    dpr_all_documents = wiki_data
    # logger.info("dpr_all_documents size %d", len(dpr_all_documents))

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    processes = ProcessPool(processes=workers_num)
    # logger.info("Matching answers in top docs...")
    get_score_partial = partial(check_answer, match_type=match_type, tokenizer=tokenizer)

    questions_answers_docs = zip(answers, closest_docs)
    scores = processes.map(get_score_partial, questions_answers_docs)

    # logger.info("Per question validation results len=%d", len(scores))

    n_docs = len(closest_docs[0])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]
    return QAMatchStats(top_k_hits, scores)


def calculate_matches_from_meta(
    answers: List[List[str]],
    closest_docs: List[Tuple[List[object], List[float]]],
    workers_num: int,
    match_type: str,
    use_title: bool = False,
    meta_compressed: bool = False,
) -> QAMatchStats:

    tok_opts = {}
    tokenizer = SimpleTokenizer(**tok_opts)

    processes = ProcessPool(processes=workers_num)
    logger.info("Matching answers in top docs...")
    get_score_partial = partial(
        check_answer_from_meta,
        match_type=match_type,
        tokenizer=tokenizer,
        use_title=use_title,
        meta_compressed=meta_compressed,
    )

    questions_answers_docs = zip(answers, closest_docs)
    scores = processes.map(get_score_partial, questions_answers_docs)

    logger.info("Per question validation results len=%d", len(scores))

    n_docs = len(closest_docs[0][0])
    top_k_hits = [0] * n_docs
    for question_hits in scores:
        best_hit = next((i for i, x in enumerate(question_hits) if x), None)
        if best_hit is not None:
            top_k_hits[best_hit:] = [v + 1 for v in top_k_hits[best_hit:]]

    return QAMatchStats(top_k_hits, scores)


def check_answer(questions_answers_docs, tokenizer, match_type) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers, doc_ids = questions_answers_docs

    global dpr_all_documents
    hits = []

    for i, doc_id in enumerate(doc_ids):
        doc = dpr_all_documents[doc_id]
        text = doc['text']

        answer_found = False
        if text is None:  # cannot find the document for some reason
            logger.warning("no doc in db")
            hits.append(False)
            continue
        if match_type == "kilt":
            if has_answer_kilt(answers, text):
                answer_found = True
        elif has_answer(answers, text, tokenizer, match_type):
            answer_found = True
        hits.append(answer_found)
    return hits


def check_answer_from_meta(
    questions_answers_docs,
    tokenizer,
    match_type,
    meta_body_idx: int = 1,
    meta_title_idx: int = 2,
    use_title: bool = False,
    meta_compressed: bool = False,
) -> List[bool]:
    """Search through all the top docs to see if they have any of the answers."""
    answers, (docs_meta, doc_scores) = questions_answers_docs

    hits = []

    for i, doc_meta in enumerate(docs_meta):

        text = doc_meta[meta_body_idx]
        title = doc_meta[meta_title_idx] if len(doc_meta) > meta_title_idx else ""
        if meta_compressed:
            text = zlib.decompress(text).decode()
            title = zlib.decompress(title).decode()

        if use_title:
            text = title + " . " + text
        answer_found = False
        if has_answer(answers, text, tokenizer, match_type):
            answer_found = True
        hits.append(answer_found)
    return hits


def has_answer(answers, text, tokenizer, match_type) -> bool:
    """Check if a document contains an answer string.
    If `match_type` is string, token matching is done between the text and answer.
    If `match_type` is regex, we search the whole text with the regex.
    """
    text = _normalize(text)

    if match_type == "string":
        # Answer is a list of possible strings
        text = tokenizer.tokenize(text).words(uncased=True)

        for single_answer in answers:
            single_answer = _normalize(single_answer)
            single_answer = tokenizer.tokenize(single_answer)
            single_answer = single_answer.words(uncased=True)

            for i in range(0, len(text) - len(single_answer) + 1):
                if single_answer == text[i : i + len(single_answer)]:
                    return True

    elif match_type == "regex":
        # Answer is a regex
        for single_answer in answers:
            single_answer = _normalize(single_answer)
            if regex_match(text, single_answer):
                return True
    return False


def regex_match(text, pattern):
    """Test if a regex pattern is contained within a text."""
    try:
        pattern = re.compile(pattern, flags=re.IGNORECASE + re.UNICODE + re.MULTILINE)
    except BaseException:
        return False
    return pattern.search(text) is not None


# function for the reader model answer validation
def exact_match_score(prediction, ground_truth):
    return _normalize_answer(prediction) == _normalize_answer(ground_truth)


def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _normalize(text):
    return unicodedata.normalize("NFD", text)


# -------------------- KILT eval ---------------------------------


def has_answer_kilt(answers, text) -> bool:
    text = normalize_kilt(text)
    for single_answer in answers:
        single_answer = normalize_kilt(single_answer)
        if single_answer in text:
            return True
    return False


# answer normalization
def normalize_kilt(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class Tokens(object):
    """A class to represent a list of tokenized text."""

    TEXT = 0
    TEXT_WS = 1
    SPAN = 2
    POS = 3
    LEMMA = 4
    NER = 5

    def __init__(self, data, annotators, opts=None):
        self.data = data
        self.annotators = annotators
        self.opts = opts or {}

    def __len__(self):
        """The number of tokens."""
        return len(self.data)

    def slice(self, i=None, j=None):
        """Return a view of the list of tokens from [i, j)."""
        new_tokens = copy.copy(self)
        new_tokens.data = self.data[i:j]
        return new_tokens

    def untokenize(self):
        """Returns the original text (with whitespace reinserted)."""
        return "".join([t[self.TEXT_WS] for t in self.data]).strip()

    def words(self, uncased=False):
        """Returns a list of the text of each token

        Args:
            uncased: lower cases text
        """
        if uncased:
            return [t[self.TEXT].lower() for t in self.data]
        else:
            return [t[self.TEXT] for t in self.data]

    def offsets(self):
        """Returns a list of [start, end) character offsets of each token."""
        return [t[self.SPAN] for t in self.data]

    def pos(self):
        """Returns a list of part-of-speech tags of each token.
        Returns None if this annotation was not included.
        """
        if "pos" not in self.annotators:
            return None
        return [t[self.POS] for t in self.data]

    def lemmas(self):
        """Returns a list of the lemmatized text of each token.
        Returns None if this annotation was not included.
        """
        if "lemma" not in self.annotators:
            return None
        return [t[self.LEMMA] for t in self.data]

    def entities(self):
        """Returns a list of named-entity-recognition tags of each token.
        Returns None if this annotation was not included.
        """
        if "ner" not in self.annotators:
            return None
        return [t[self.NER] for t in self.data]

    def ngrams(self, n=1, uncased=False, filter_fn=None, as_strings=True):
        """Returns a list of all ngrams from length 1 to n.

        Args:
            n: upper limit of ngram length
            uncased: lower cases text
            filter_fn: user function that takes in an ngram list and returns
              True or False to keep or not keep the ngram
            as_string: return the ngram as a string vs list
        """

        def _skip(gram):
            if not filter_fn:
                return False
            return filter_fn(gram)

        words = self.words(uncased)
        ngrams = [
            (s, e + 1)
            for s in range(len(words))
            for e in range(s, min(s + n, len(words)))
            if not _skip(words[s : e + 1])
        ]

        # Concatenate into strings
        if as_strings:
            ngrams = ["{}".format(" ".join(words[s:e])) for (s, e) in ngrams]

        return ngrams

    def entity_groups(self):
        """Group consecutive entity tokens with the same NER tag."""
        entities = self.entities()
        if not entities:
            return None
        non_ent = self.opts.get("non_ent", "O")
        groups = []
        idx = 0
        while idx < len(entities):
            ner_tag = entities[idx]
            # Check for entity tag
            if ner_tag != non_ent:
                # Chomp the sequence
                start = idx
                while idx < len(entities) and entities[idx] == ner_tag:
                    idx += 1
                groups.append((self.slice(start, idx).untokenize(), ner_tag))
            else:
                idx += 1
        return groups


class Tokenizer(object):
    """Base tokenizer class.
    Tokenizers implement tokenize, which should return a Tokens class.
    """

    def tokenize(self, text):
        raise NotImplementedError

    def shutdown(self):
        pass

    def __del__(self):
        self.shutdown()


class SimpleTokenizer(Tokenizer):
    ALPHA_NUM = r"[\p{L}\p{N}\p{M}]+"
    NON_WS = r"[^\p{Z}\p{C}]"

    def __init__(self, **kwargs):
        """
        Args:
            annotators: None or empty set (only tokenizes).
        """
        self._regexp = regex.compile(
            "(%s)|(%s)" % (self.ALPHA_NUM, self.NON_WS),
            flags=regex.IGNORECASE + regex.UNICODE + regex.MULTILINE,
        )
        if len(kwargs.get("annotators", {})) > 0:
            logger.warning(
                "%s only tokenizes! Skipping annotators: %s" % (type(self).__name__, kwargs.get("annotators"))
            )
        self.annotators = set()

    def tokenize(self, text):
        data = []
        matches = [m for m in self._regexp.finditer(text)]
        for i in range(len(matches)):
            # Get text
            token = matches[i].group()

            # Get whitespace
            span = matches[i].span()
            start_ws = span[0]
            if i + 1 < len(matches):
                end_ws = matches[i + 1].span()[0]
            else:
                end_ws = span[1]

            # Format data
            data.append(
                (
                    token,
                    text[start_ws:end_ws],
                    span,
                )
            )
        return Tokens(data, self.annotators)


class SpacyTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        """
        Args:
            annotators: set that can include pos, lemma, and ner.
            model: spaCy model to use (either path, or keyword like 'en').
        """
        model = kwargs.get("model", "en_core_web_sm")  # TODO: replace with en ?
        self.annotators = copy.deepcopy(kwargs.get("annotators", set()))
        nlp_kwargs = {"parser": False}
        if not any([p in self.annotators for p in ["lemma", "pos", "ner"]]):
            nlp_kwargs["tagger"] = False
        if "ner" not in self.annotators:
            nlp_kwargs["entity"] = False
        self.nlp = spacy.load(model, **nlp_kwargs)

    def tokenize(self, text):
        # We don't treat new lines as tokens.
        clean_text = text.replace("\n", " ")
        tokens = self.nlp.tokenizer(clean_text)
        if any([p in self.annotators for p in ["lemma", "pos", "ner"]]):
            self.nlp.tagger(tokens)
        if "ner" in self.annotators:
            self.nlp.entity(tokens)

        data = []
        for i in range(len(tokens)):
            # Get whitespace
            start_ws = tokens[i].idx
            if i + 1 < len(tokens):
                end_ws = tokens[i + 1].idx
            else:
                end_ws = tokens[i].idx + len(tokens[i].text)

            data.append(
                (
                    tokens[i].text,
                    text[start_ws:end_ws],
                    (tokens[i].idx, tokens[i].idx + len(tokens[i].text)),
                    tokens[i].tag_,
                    tokens[i].lemma_,
                    tokens[i].ent_type_,
                )
            )

        # Set special option for non-entity tag: '' vs 'O' in spaCy
        return Tokens(data, self.annotators, opts={"non_ent": ""})
