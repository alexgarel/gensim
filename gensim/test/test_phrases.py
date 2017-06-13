#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

"""
Automated tests for checking transformation algorithms (the models package).
"""


import logging
import unittest
import os
import sys

from gensim import utils
from gensim.models.phrases import Phrases, Phraser, lrsentence, pseudocorpus

if sys.version_info[0] >= 3:
    unicode = str

module_path = os.path.dirname(__file__)  # needed because sample data files are located in the same folder
datapath = lambda fname: os.path.join(module_path, 'test_data', fname)


class TestUtils(unittest.TestCase):

    def test_pseudocorpus_no_common_terms(self):
        vocab = [
            "prime_minister",
            "gold",
            "chief_technical_officer",
            "effective"]
        result = list(pseudocorpus(vocab, "_"))
        self.assertEqual(
            result,
            [["prime", "minister"],
             ["chief", "technical_officer"],
             ["chief_technical", "officer"]])

    def test_pseudocorpus_with_common_terms(self):
        vocab = [
            "hall_of_fame",
            "gold",
            "chief_of_political_bureau",
            "effective",
            "beware_of_the_dog_in_the_yard"]
        common_terms=frozenset(["in", "the", "of"])
        result = list(pseudocorpus(vocab, "_", common_terms=common_terms))
        self.assertEqual(
            result,
            [["hall", "of", "fame"],
             ["chief", "of",  "political_bureau"],
             ["chief_of_political", "bureau"],
             ["beware", "of", "the", "dog_in_the_yard"],
             ["beware_of_the_dog", "in", "the", "yard"]])

    def test_lrsentence(self):
        common_terms=frozenset(["is", "in", "the", "of"])
        sentence = "the prime minister is in the yard of webminster".split()
        lsent, rsent = lrsentence(sentence, "_", common_terms=common_terms)
        self.assertEqual(lsent, ["prime", "minister",  "yard",  "webminster"])
        self.assertEqual(rsent, ["the_prime", "minister", "is_in_the_yard", "of_webminster"])


class PhrasesData:
    sentences = [
        ['human', 'interface', 'computer'],
        ['survey', 'user', 'computer', 'system', 'response', 'time'],
        ['eps', 'user', 'interface', 'system'],
        ['system', 'human', 'system', 'eps'],
        ['user', 'response', 'time'],
        ['trees'],
        ['graph', 'trees'],
        ['graph', 'minors', 'trees'],
        ['graph', 'minors', 'survey'],
        ['graph', 'minors', 'survey','human','interface'] #test bigrams within same sentence
    ]
    unicode_sentences = [[utils.to_unicode(w) for w in sentence] for sentence in sentences]
    common_terms = frozenset()

    bigram1 = u'response_time'
    bigram2 = u'graph_minors'
    bigram3 = u'human_interface'

    def gen_sentences(self):
        return ((w for w in sentence) for sentence in self.sentences)


class PhrasesCommon:
    """ Tests that need to be run for both Prases and Phraser classes."""
    def setUp(self):
        self.bigram = Phrases(
            self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)
        self.bigram_default = Phrases(
            self.sentences, common_terms=self.common_terms)
        self.bigram_utf8 = Phrases(
            self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)
        self.bigram_unicode = Phrases(
            self.unicode_sentences, min_count=1, threshold=1, common_terms=self.common_terms)

    def testEmptyInputsOnBigramConstruction(self):
        """Test that empty inputs don't throw errors and return the expected result."""
        # Empty list -> empty list
        self.assertEqual(list(self.bigram_default[[]]), [])
        # Empty iterator -> empty list
        self.assertEqual(list(self.bigram_default[iter(())]), [])
        # List of empty list -> list of empty list
        self.assertEqual(list(self.bigram_default[[[], []]]), [[], []])
        # Iterator of empty list -> list of empty list
        self.assertEqual(list(self.bigram_default[iter([[], []])]), [[], []])
        # Iterator of empty iterator -> list of empty list
        self.assertEqual(list(self.bigram_default[(iter(()) for i in range(2))]), [[], []])

    def testSentenceGeneration(self):
        """Test basic bigram using a dummy corpus."""
        # test that we generate the same amount of sentences as the input
        self.assertEqual(len(self.sentences), len(list(self.bigram_default[self.sentences])))

    def testSentenceGenerationWithGenerator(self):
        """Test basic bigram production when corpus is a generator."""
        self.assertEqual(len(list(self.gen_sentences())),
                         len(list(self.bigram_default[self.gen_sentences()])))

    def testBigramConstruction(self):
        """Test Phrases bigram construction building."""
        # with this setting we should get response_time and graph_minors
        bigram1_seen = False
        bigram2_seen = False

        for s in self.bigram[self.sentences]:
            if not bigram1_seen and self.bigram1 in s:
                bigram1_seen = True
            if not bigram2_seen and self.bigram2 in s:
                bigram2_seen = True
            if bigram1_seen and bigram2_seen:
                break

        self.assertTrue(bigram1_seen and bigram2_seen)

        # check the same thing, this time using single doc transformation
        # last sentence should contain both graph_minors and human_interface
        self.assertTrue(self.bigram1 in self.bigram[self.sentences[1]])
        self.assertTrue(self.bigram1 in self.bigram[self.sentences[4]])
        self.assertTrue(self.bigram2 in self.bigram[self.sentences[-2]])
        self.assertTrue(self.bigram2 in self.bigram[self.sentences[-1]])
        self.assertTrue(self.bigram3 in self.bigram[self.sentences[-1]])

    def testBigramConstructionFromGenerator(self):
        """Test Phrases bigram construction building when corpus is a generator"""
        bigram1_seen = False
        bigram2_seen = False

        for s in self.bigram[self.gen_sentences()]:
            if not bigram1_seen and self.bigram1 in s:
                bigram1_seen = True
            if not bigram2_seen and self.bigram2 in s:
                bigram2_seen = True
            if bigram1_seen and bigram2_seen:
                break
        self.assertTrue(bigram1_seen and bigram2_seen)

    def testEncoding(self):
        """Test that both utf8 and unicode input work; output must be unicode."""
        expected = [u'survey', u'user', u'computer', u'system', u'response_time']

        self.assertEqual(self.bigram_utf8[self.sentences[1]], expected)
        self.assertEqual(self.bigram_unicode[self.sentences[1]], expected)

        transformed = ' '.join(self.bigram_utf8[self.sentences[1]])
        self.assertTrue(isinstance(transformed, unicode))


class TestPhrasesModel(PhrasesData, PhrasesCommon, unittest.TestCase):

    def testExportPhrases(self):
        """Test Phrases bigram export_phrases functionality."""
        bigram = Phrases(self.sentences, min_count=1, threshold=1)

        seen_bigrams = set()

        for phrase, score in bigram.export_phrases(self.sentences):
            seen_bigrams.add(phrase)

        assert seen_bigrams == set([
            b'response time',
            b'graph minors',
            b'human interface',
        ])

    def test_multiple_bigrams_single_entry(self):
        """ a single entry should produce multiple bigrams. """
        bigram = Phrases(self.sentences, min_count=1, threshold=1)
        seen_bigrams = set()

        test_sentences = [['graph', 'minors', 'survey', 'human', 'interface']]
        for phrase, score in bigram.export_phrases(test_sentences):
            seen_bigrams.add(phrase)

        assert seen_bigrams == set([
            b'graph minors',
            b'human interface'
        ])

    def testBadParameters(self):
        """Test the phrases module with bad parameters."""
        # should fail with something less or equal than 0
        self.assertRaises(ValueError, Phrases, self.sentences, min_count=0)

        # threshold should be positive
        self.assertRaises(ValueError, Phrases, self.sentences, threshold=-1)

    def testPruning(self):
        """Test that max_vocab_size parameter is respected."""
        bigram = Phrases(self.sentences, max_vocab_size=5)
        self.assertTrue(len(bigram.vocab) <= 5)
#endclass TestPhrasesModel


class TestPhraserModel(PhrasesData, PhrasesCommon, unittest.TestCase):
    """ Test Phraser models."""
    def setUp(self):
        """Set up Phraser models for the tests."""
        bigram_phrases = Phrases(
            self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)
        self.bigram = Phraser(bigram_phrases)

        bigram_default_phrases = Phrases(self.sentences, common_terms=self.common_terms)
        self.bigram_default = Phraser(bigram_default_phrases)

        bigram_utf8_phrases = Phrases(
            self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)
        self.bigram_utf8 = Phraser(bigram_utf8_phrases)

        bigram_unicode_phrases = Phrases(
            self.unicode_sentences, min_count=1, threshold=1, common_terms=self.common_terms)
        self.bigram_unicode = Phraser(bigram_unicode_phrases)


class CommonTermsPhrasesData:
    """This mixin permits to reuse the test, using, this time the common_terms option
    """

    sentences = [
        ['human', 'interface', 'with', 'computer'],
        ['survey', 'of', 'user', 'computer', 'system', 'lack', 'of', 'interest'],
        ['eps', 'user', 'interface', 'system'],
        ['system', 'and', 'human', 'system', 'eps'],
        ['user', 'lack', 'of', 'interest'],
        ['trees'],
        ['graph', 'of', 'trees'],
        ['data', 'and', 'graph', 'of', 'trees'],
        ['data', 'and', 'graph', 'survey'],
        ['data', 'and', 'graph', 'survey', 'for', 'human','interface'] #test bigrams within same sentence
    ]
    unicode_sentences = [[utils.to_unicode(w) for w in sentence] for sentence in sentences]
    common_terms = ['of', 'and', 'for']

    bigram1 = u'lack_of_interest'
    bigram2 = u'data_and_graph'
    bigram3 = u'human_interface'
    expression1 = u'lack of interest'
    expression2 = u'data and graph'
    expression3 = u'human interface'

    def gen_sentences(self):
        return ((w for w in sentence) for sentence in self.sentences)


class TestPhrasesModelCommonTerms(CommonTermsPhrasesData, TestPhrasesModel):
    """ Test Phrases models with common terms"""

    def testEncoding(self):
        """Test that both utf8 and unicode input work; output must be unicode."""
        expected = [u'survey', u'of', u'user', u'computer', u'system', u'lack_of_interest']

        self.assertEqual(self.bigram_utf8[self.sentences[1]], expected)
        self.assertEqual(self.bigram_unicode[self.sentences[1]], expected)

        transformed = ' '.join(self.bigram_utf8[self.sentences[1]])
        self.assertTrue(isinstance(transformed, unicode))

    def test_multiple_bigrams_single_entry(self):
        """ a single entry should produce multiple bigrams. """
        bigram = Phrases(self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)

        seen_bigrams = set()
        test_sentences = [['data', 'and', 'graph', 'survey', 'for', 'human','interface']]
        for phrase, score in bigram.export_phrases(test_sentences):
            seen_bigrams.add(phrase)
        assert seen_bigrams == set([
            b'data and graph',
            b'human interface',
        ])

    def testExportPhrases(self):
        """Test Phrases bigram export_phrases functionality."""
        bigram = Phrases(self.sentences, min_count=1, threshold=1, common_terms=self.common_terms)

        seen_bigrams = set()

        for phrase, score in bigram.export_phrases(self.sentences):
            seen_bigrams.add(phrase)

        assert seen_bigrams == set([
            b'human interface',
            b'graph of trees',
            b'data and graph',
            b'lack of interest',
        ])


class TestPhraserModelCommonTerms(CommonTermsPhrasesData, TestPhraserModel):

    def testEncoding(self):
        """Test that both utf8 and unicode input work; output must be unicode."""
        expected = [u'survey', u'of', u'user', u'computer', u'system', u'lack_of_interest']

        self.assertEqual(self.bigram_utf8[self.sentences[1]], expected)
        self.assertEqual(self.bigram_unicode[self.sentences[1]], expected)

        transformed = ' '.join(self.bigram_utf8[self.sentences[1]])
        self.assertTrue(isinstance(transformed, unicode))


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()
