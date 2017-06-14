#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automatically detect common phrases (multiword expressions) from a stream of sentences.

The phrases are collocations (frequently co-occurring tokens). See [1]_ for the
exact formula.

For example, if your input stream (=an iterable, with each value a list of token strings) looks like:

>>> print(list(sentence_stream))
[[u'the', u'mayor', u'of', u'new', u'york', u'was', u'there'],
 [u'machine', u'learning', u'can', u'be', u'useful', u'sometimes'],
 ...,
]

you'd train the detector with:

>>> phrases = Phrases(sentence_stream)

and then create a performant Phraser object to transform any sentence (list of token strings) using the standard gensim syntax:

>>> bigram = Phraser(phrases)
>>> sent = [u'the', u'mayor', u'of', u'new', u'york', u'was', u'there']
>>> print(bigram[sent])
[u'the', u'mayor', u'of', u'new_york', u'was', u'there']

(note `new_york` became a single token). As usual, you can also transform an entire
sentence stream using:

>>> print(list(bigram[any_sentence_stream]))
[[u'the', u'mayor', u'of', u'new_york', u'was', u'there'],
 [u'machine_learning', u'can', u'be', u'useful', u'sometimes'],
 ...,
]

You can also continue updating the collocation counts with new sentences, by:

>>> bigram.add_vocab(new_sentence_stream)

These **phrase streams are meant to be used during text preprocessing, before
converting the resulting tokens into vectors using `Dictionary`**. See the
:mod:`gensim.models.word2vec` module for an example application of using phrase detection.

The detection can also be **run repeatedly**, to get phrases longer than
two tokens (e.g. `new_york_times`):

>>> trigram = Phrases(bigram[sentence_stream])
>>> sent = [u'the', u'new', u'york', u'times', u'is', u'a', u'newspaper']
>>> print(trigram[bigram[sent]])
[u'the', u'new_york_times', u'is', u'a', u'newspaper']

The common_terms parameter add a way to give special treatment to common terms (aka stop words)
such that their presence between two words
won't prevent bigram detection.
It allows to detect expressions like "bank of america" or "eye of the beholder".

>>> common_terms = ["of", "with", "without", "and", "or", "the", "a"]
>>> ct_phrases = Phrases(sentence_stream, common_terms=common_terms)

The phraser will of course inherit the common_terms from Phrases.

>>> ct_bigram = Phraser(ct_phrases)
>>> sent = [u'the', u'mayor', u'shows', u'his', u'lack', u'of', u'interest']
>>> print(bigram[sent])
[u'the', u'mayor', u'shows', u'his', u'lack_of_interest']


.. [1] Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, and Jeffrey Dean.
       Distributed Representations of Words and Phrases and their Compositionality.
       In Proceedings of NIPS, 2013.

"""

import sys
import os
import logging
import warnings
from collections import defaultdict
import itertools as it
import functools as ft

from six import iteritems, string_types, next

from gensim import utils, interfaces

logger = logging.getLogger(__name__)


def _is_single(obj):
    """
    Check whether `obj` is a single document or an entire corpus.
    Returns (is_single, new) 2-tuple, where `new` yields the same
    sequence as `obj`.

    `obj` is a single document if it is an iterable of strings.  It
    is a corpus if it is an iterable of documents.
    """
    obj_iter = iter(obj)
    try:
        peek = next(obj_iter)
        obj_iter = it.chain([peek], obj_iter)
    except StopIteration:
        # An empty object is a single document
        return True, obj
    if isinstance(peek, string_types):
        # It's a document, return the iterator
        return True, obj_iter
    else:
        # If the first item isn't a string, assume obj is a corpus
        return False, obj_iter


# def lrsentence(sentence, common_terms):
#     """given a sentence craft a first sentence where common_terms where removed
#     and a left one where they are merged with the following "uncommon" term.
# 
#     :param delimiter: delimiter to glue common terms to following term.
#       if None, tuples are returned instead
# 
#     :return tuple: sentence without common_terms, sentence with merged common_terms
#     """
#     # first part does not use stop words
#     lsentence = [w for w in sentence if w not in common_terms]
#     # second sentence join stop words and following word
#     rsentence = []
#     cterms = []
#     for w in sentence:
#         if w in common_terms:
#             cterms.append(w)
#         else:
#             if delimiter:
#                 w = delimiter.join(cterms + [w])
#             rsentence.append(w)
#             cterms = []
#     return lsentence, rsentence


class SentenceAnalyzer:

    def score(self, vocab, word_a, word_b, bigram_word, min_count):
        raise NotImplementedError("implement in sub-classes")

    def analyze_sentence(self, sentence, vocab, delimiter, threshold, min_count, common_terms):
        s = [utils.any2utf8(w) for w in sentence]
        previous_common = None
        last_uncommon = None
        in_between = []
        # adding None is a trick that helps getting an automatic happy ending
        # has it won't be a common_word, nor score
        for word in s + [None]:
            is_common = word in common_terms
            if not is_common and last_uncommon:
                chain = [last_uncommon] + in_between + [word]
                # test between last_uncommon
                bigram_word = delimiter.join(chain) if word is not None else None
                score = self.score(vocab, last_uncommon, word, bigram_word, min_count)
                if score > threshold:
                    if previous_common is not None:
                        yield (previous_common, None)
                    yield (chain, score)
                    previous_common = None
                    last_uncommon = None
                    in_between = []
                else:
                    # search with common words
                    # we don't add word, as it may participate in a future bigram
                    chain = []
                    if previous_common:
                        chain.append(previous_common)
                    chain.append(last_uncommon)
                    chain.extend(in_between)
                    bigram_yielded = False
                    for w1, w2 in zip(chain, chain[1:]):
                        bigram_word = delimiter.join((w1, w2))
                        score = self.score(vocab, w1, w2, bigram_word, min_count)
                        if score > threshold:
                            yield ((w1, w2), score)
                            bigram_yielded = True
                        else:
                            yield (w1, None)
                            bigram_yielded = False
                    previous_common = chain[-1] if not bigram_yielded else None
                    last_uncommon = word
                    in_between = []
            elif not is_common:
                # first common term
                last_uncommon = word
            else:  # common term
                if last_uncommon:
                    # wait for uncommon resolution
                    in_between.append(word)
                else:
                    # yield previous common and take its place
                    if previous_common is not None:
                        yield (previous_common, None)
                    previous_common = word

        if previous_common is not None:
            yield (previous_common, None)

#        if common_terms:
#            lsentence, rsentence = lrsentence(s, None, common_terms)
#        else:
#            lsentence, rsentence = s, s
#            bigrams = []
#        for word_a, words_b, orig_b in zip(lsentence, rsentence[1:], lsentence[1:]):
#            tail = [word_a] + words_b
#            # try whole bigram
#            bigram_word = delimiter.join(tail)
#            if not last_bigram:
#                score = self.score(vocab, word_a, orig_b, bigram_word, min_count)
#                if score > threshold:
#                    bigrams.append((tail, score))
#                    tail = []
#                    last_bigram = True
#            if not last_bigram and len(tail) > 3:
#                # give a chance to common terms
#                # with word a first
#                ct = tail[1]
#                bigram_word = delimiter.join((word_a, ct))
#                score = self.score(word_a, ct, bigram_word, min_count)
#                if score > threshold:
#                    bigrams.append(((word_a, ct), score))
#                    tail = tail[2:]
#                else:
#                    bigrams.append((word_a, None))
#                    tail = tail[1:]
#            if len(tail) > 2:
#                # also give a chance to b
#                ct = tail[-2]
#                bigram_word = delimiter.join((ct, orig_b))
#                score = self.score(ct, orig_b, bigram_word, min_count)
#                if score > threshold:
#                    bigrams.extend((w, None) for w in tail[:-2])
#                    bigrams.append(((ct, orig_b), score))
#                    tail = []
#                    last_bigram = True
#            if tail:
#                # add tail but word_b, it would be in next round
#                bigrams.extend((w, None) for w in tail[:-1])
#        if not last_bigram:
#            # add last word skipped by last loop
#            bigrams.append((s[-1], None))
#        return bigrams


class Phrases(SentenceAnalyzer, interfaces.TransformationABC):
    """
    Detect phrases, based on collected collocation counts. Adjacent words that appear
    together more frequently than expected are joined together with the `_` character.

    It can be used to generate phrases on the fly, using the `phrases[sentence]`
    and `phrases[corpus]` syntax.

    """
    def __init__(self, sentences=None, min_count=5, threshold=10.0,
                 max_vocab_size=40000000, delimiter=b'_', progress_per=10000,
                 common_terms=frozenset()):
        """
        Initialize the model from an iterable of `sentences`. Each sentence must be
        a list of words (unicode strings) that will be used for training.

        The `sentences` iterable can be simply a list, but for larger corpora,
        consider a generator that streams the sentences directly from disk/network,
        without storing everything in RAM. See :class:`BrownCorpus`,
        :class:`Text8Corpus` or :class:`LineSentence` in the :mod:`gensim.models.word2vec`
        module for such examples.

        `min_count` ignore all words and bigrams with total collected count lower
        than this.

        `threshold` represents a threshold for forming the phrases (higher means
        fewer phrases). A phrase of words `a` and `b` is accepted if
        `(cnt(a, b) - min_count) * N / (cnt(a) * cnt(b)) > threshold`, where `N` is the
        total vocabulary size.

        `max_vocab_size` is the maximum size of the vocabulary. Used to control
        pruning of less common words, to keep memory under control. The default
        of 40M needs about 3.6GB of RAM; increase/decrease `max_vocab_size` depending
        on how much available memory you have.

        `delimiter` is the glue character used to join collocation tokens, and
        should be a byte string (e.g. b'_').

        `common_terms` is an optionnal list of "stop words" that won't affect frequency count
        of expressions containing them.
        """
        if min_count <= 0:
            raise ValueError("min_count should be at least 1")

        if threshold <= 0:
            raise ValueError("threshold should be positive")

        self.min_count = min_count
        self.threshold = threshold
        self.max_vocab_size = max_vocab_size
        self.vocab = defaultdict(int)  # mapping between utf8 token => its count
        self.min_reduce = 1  # ignore any tokens with count smaller than this
        self.delimiter = delimiter
        self.progress_per = progress_per
        self.common_terms = frozenset(utils.any2utf8(w) for w in common_terms)

        if sentences is not None:
            self.add_vocab(sentences)

    def __str__(self):
        """Get short string representation of this phrase detector."""
        return "%s<%i vocab, min_count=%s, threshold=%s, max_vocab_size=%s>" % (
            self.__class__.__name__, len(self.vocab), self.min_count,
            self.threshold, self.max_vocab_size)

    @staticmethod
    def learn_vocab(sentences, max_vocab_size, delimiter=b'_', progress_per=10000,
                    common_terms=frozenset()):
        """Collect unigram/bigram counts from the `sentences` iterable."""
        sentence_no = -1
        total_words = 0
        logger.info("collecting all words and their counts")
        vocab = defaultdict(int)
        min_reduce = 1
        for sentence_no, sentence in enumerate(sentences):
            if sentence_no % progress_per == 0:
                logger.info("PROGRESS: at sentence #%i, processed %i words and %i word types" %
                            (sentence_no, total_words, len(vocab)))
            s = [utils.any2utf8(w) for w in sentence]
            last_uncommon = None
            last = None
            in_between = []
            for word in s:
                vocab[word] += 1
                if last is not None:
                    vocab[delimiter.join((last, word))] += 1
                if word not in common_terms:
                    # note: we check if there are common_terms,
                    # else it has already been done above
                    if last_uncommon is not None and in_between:
                        vocab[delimiter.join([last_uncommon] + in_between + [word])] += 1
                    last_uncommon = word
                    in_between = []
                elif last_uncommon is not None:
                    in_between.append(word)
                last = word
                total_words += 1

            if len(vocab) > max_vocab_size:
                utils.prune_vocab(vocab, min_reduce)
                min_reduce += 1

        logger.info("collected %i word types from a corpus of %i words (unigram + bigrams) and %i sentences" %
                    (len(vocab), total_words, sentence_no + 1))
        return min_reduce, vocab

    def add_vocab(self, sentences):
        """
        Merge the collected counts `vocab` into this phrase detector.

        """
        # uses a separate vocab to collect the token counts from `sentences`.
        # this consumes more RAM than merging new sentences into `self.vocab`
        # directly, but gives the new sentences a fighting chance to collect
        # sufficient counts, before being pruned out by the (large) accummulated
        # counts collected in previous learn_vocab runs.
        min_reduce, vocab = self.learn_vocab(
            sentences, self.max_vocab_size, self.delimiter, self.progress_per, self.common_terms)

        if len(self.vocab) > 0:
            logger.info("merging %i counts into %s", len(vocab), self)
            self.min_reduce = max(self.min_reduce, min_reduce)
            for word, count in iteritems(vocab):
                self.vocab[word] += count
            if len(self.vocab) > self.max_vocab_size:
                utils.prune_vocab(self.vocab, self.min_reduce)
                self.min_reduce += 1
            logger.info("merged %s", self)
        else:
            # in common case, avoid doubling gigantic dict
            logger.info("using %i counts as vocab in %s", len(vocab), self)
            self.vocab = vocab

    def score(self, vocab, word_a, word_b, bigram, min_count):
        if word_a in vocab and word_b in vocab and bigram in vocab:
            pa = float(vocab[word_a])
            pb = float(vocab[word_b])
            pab = float(vocab[bigram])
            return (pab - min_count) / pa / pb * len(vocab)
        else:
            return -1

    def export_phrases(self, sentences, out_delimiter=b' ', as_tuples=False):
        """
        Generate an iterator that contains all phrases in given 'sentences'

        Example::

          >>> sentences = Text8Corpus(path_to_corpus)
          >>> bigram = Phrases(sentences, min_count=5, threshold=100)
          >>> for phrase, score in bigram.export_phrases(sentences):
          ...     print(u'{0}\t{1}'.format(phrase, score))

            then you can debug the threshold with generated tsv
        """
        analyze_sentence = ft.partial(
            self.analyze_sentence,
            vocab=self.vocab,
            delimiter=self.delimiter,
            threshold=self.threshold,
            min_count=self.min_count,
            common_terms=self.common_terms)
        for sentence in sentences:
            bigrams = analyze_sentence(sentence)
            # keeps only not None scores
            filtered = ((words, score) for words, score in bigrams if score is not None)
            for words, score in filtered:
                if as_tuples:
                    yield (tuple(words), score)
                else:
                    yield (out_delimiter.join(words), score)

    def __getitem__(self, sentence):
        """
        Convert the input tokens `sentence` (=list of unicode strings) into phrase
        tokens (=list of unicode strings, where detected phrases are joined by u'_').

        If `sentence` is an entire corpus (iterable of sentences rather than a single
        sentence), return an iterable that converts each of the corpus' sentences
        into phrases on the fly, one after another.

        Example::

          >>> sentences = Text8Corpus(path_to_corpus)
          >>> bigram = Phrases(sentences, min_count=5, threshold=100)
          >>> for sentence in phrases[sentences]:
          ...     print(u' '.join(s))
            he refuted nechaev other anarchists sometimes identified as pacifist anarchists advocated complete
            nonviolence leo_tolstoy

        """
        warnings.warn("For a faster implementation, use the gensim.models.phrases.Phraser class")

        is_single, sentence = _is_single(sentence)
        if not is_single:
            # if the input is an entire corpus (rather than a single sentence),
            # return an iterable stream.
            return self._apply(sentence)

        delimiter = self.delimiter
        bigrams = self.analyze_sentence(
            sentence,
            vocab=self.vocab,
            delimiter=delimiter,
            threshold=self.threshold,
            min_count=self.min_count,
            common_terms=self.common_terms)
        new_s = []
        for words, score in bigrams:
            if score is not None:
                words = delimiter.join(words)
            new_s.append(words)
        return [utils.to_unicode(w) for w in new_s]


def pseudocorpus(source_vocab, sep, common_terms=frozenset()):
    """Feeds source_vocab's compound keys back to it, to discover phrases"""
    for k in source_vocab:
        if sep not in k:
            continue
        unigrams = k.split(sep)
        for i in range(1, len(unigrams)):
            if unigrams[i-1] not in common_terms:
                # do not join common terms
                cterms = list(it.takewhile(lambda w: w in common_terms, unigrams[i:]))
                tail = unigrams[i + len(cterms):]
                components = [sep.join(unigrams[:i])] + cterms
                if tail:
                    components.append(sep.join(tail))
                yield components


class Phraser(SentenceAnalyzer, interfaces.TransformationABC):
    """
    Minimal state & functionality to apply results of a Phrases model to tokens.

    After the one-time initialization, a Phraser will be much smaller and
    somewhat faster than using the full Phrases model.

    Reflects the results of the source model's `min_count` and `threshold`
    settings. (You can tamper with those & create a new Phraser to try
    other values.)

    """
    def __init__(self, phrases_model):
        self.threshold = phrases_model.threshold
        self.min_count = phrases_model.min_count
        self.delimiter = phrases_model.delimiter
        self.common_terms = phrases_model.common_terms
        self.phrasegrams = {}
        corpus = self.pseudocorpus(phrases_model)
        logger.info('source_vocab length %i', len(phrases_model.vocab))
        count = 0
        for bigram, score in phrases_model.export_phrases(corpus, self.delimiter, as_tuples=True):
            if bigram in self.phrasegrams:
                logger.info('Phraser repeat %s', bigram)
            self.phrasegrams[bigram] = (phrases_model.vocab[self.delimiter.join(bigram)], score)
            count += 1
            if not count % 50000:
                logger.info('Phraser added %i phrasegrams', count)
        logger.info('Phraser built with %i %i phrasegrams', count, len(self.phrasegrams))

    def pseudocorpus(self, phrases_model):
        return pseudocorpus(phrases_model.vocab, phrases_model.delimiter,
                            phrases_model.common_terms)

    def score(self, vocab, word_a, word_b, bigram, min_count):
        try:
            return vocab[bigram]
        except KeyError:
            return -1

    def __getitem__(self, sentence):
        """
        Convert the input tokens `sentence` (=list of unicode strings) into phrase
        tokens (=list of unicode strings, where detected phrases are joined by u'_'
        (or other configured delimiter-character).

        If `sentence` is an entire corpus (iterable of sentences rather than a single
        sentence), return an iterable that converts each of the corpus' sentences
        into phrases on the fly, one after another.

        """
        is_single, sentence = _is_single(sentence)
        if not is_single:
            # if the input is an entire corpus (rather than a single sentence),
            # return an iterable stream.
            return self._apply(sentence)

        delimiter = self.delimiter
        bigrams = self.analyze_sentence(
            sentence,
            vocab=self.phrasegrams,
            delimiter=delimiter,
            threshold=self.threshold,
            min_count=self.min_count,
            common_terms=self.common_terms)
        new_s = []
        for words, score in bigrams:
            if score is not None:
                words = delimiter.join(words)
            new_s.append(words)
        return [utils.to_unicode(w) for w in new_s]


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.info("running %s" % " ".join(sys.argv))

    # check and process cmdline input
    program = os.path.basename(sys.argv[0])
    if len(sys.argv) < 2:
        print(globals()['__doc__'] % locals())
        sys.exit(1)
    infile = sys.argv[1]

    from gensim.models import Phrases  # for pickle
    from gensim.models.word2vec import Text8Corpus
    sentences = Text8Corpus(infile)

    # test_doc = LineSentence('test/test_data/testcorpus.txt')
    bigram = Phrases(sentences, min_count=5, threshold=100)
    for s in bigram[sentences]:
        print(utils.to_utf8(u' '.join(s)))
