#!/usr/bin/python
# -*- coding: utf-8 -*-

'''

@author: Raquel G. Alhama

Generate stimuli following Frank et al. 2010 (Cognition)

Vocabulary: 6 words, 2 with 2 syllables, 2 with 3 syllables, 2 with 4 syllables
The stimuli is composed of sentences.
Inside a sentence there can not be two consecutive tokens of the same word.

The length (number of words) of the sentences and the number of tokens of each word are given by command line options.

Experiments / Conditions:
1. Sentence length: 1,2,3,4,6,8,12,24.
2. Amount of exposure: 48,100,300,600,900,1200
3. Number of word types: 3,4,5,6,9

'''

import sys

import time
import re
import RnR as RnR
import math
import scipy.stats.stats as stats
import Frank2010Results as frr
from random import *

verbose = False

myRandom = Random(time.time())
EXP_COND = {1: (1, 2, 3, 4, 6, 8, 12, 24), 2: (48, 100, 300, 600, 900, 1200), 3: (3, 4, 5, 6, 9)}
CONDITION = {1: "sentence length", 2: "number of tokens", 3:"size of vocabulary"}
COLUMNS = ("sbjId", "condition", "timestamp", "wordlen", "rt", "keypressed", "correct")

''' Print if verbose mode. '''


def vprint(string):
    if verbose:
        print >> sys.stderr, string
        sys.stderr.flush()


''' A whole experiment, with all its conditions, streams, test items, etc.\
    It also loads the real data of the experiment.
    Structure: experiment.cnd[condition].stream, test, ...
'''


class experiment:
    ''' Constructor '''

    def __init__(self, expId):
        self.expId = expId
        self.CONDS_LIST = EXP_COND[expId]
        # Generate experiment for all conditions
        self.cnd = dict.fromkeys(EXP_COND[expId])
        for condition in EXP_COND[expId]:
            self.cnd[condition] = exp_cond(expId, condition)
        # Load human data of real experiment
        self.expResults = frr.experimentResults(expId)



    ''' PearsonR between human data and some model simulation'''

    def pearsonR(self, modelData):
        values_model = [v for (_, v) in sorted(modelData.items())]
        values_humans = [v for (_, v) in sorted(self.expResults.avg_performance.items())]
        values_humans = map(lambda x: x * 100, values_humans)
        pearsonr = stats.pearsonr(values_humans, values_model)[0]
        if math.isnan(pearsonr):
            pearsonr = 0.0
        return pearsonr



''' Each experiment_condition complete.
    When "shuffle" is False, the order of the test pairs is not randomized. This is just to make it more efficient since the model is blind to the effect of order.
    If used for generating the experiment for humans, then shuffle must be set to True.
'''
class exp_cond:
    def __init__(self, expId, condition, shuffle=False):

        self.SYLLABLES = (
        "ba", "bi", "da", "du", "ti", "tu", "ka", "ki", "la", "lu", "gi", "gu", "pa", "pi", "va", "vu", "zi", "zu")
        self.expId = expId
        self.condition = condition
        self.TEST_LENGTH = 30
        self.shuffle = shuffle

        if self.expId == 1:
            self.tokens = 100
            self.types = 6
            self.sentence_length = condition
        elif self.expId == 2:
            self.tokens = condition
            self.types = 6
            self.sentence_length = 4
        else:
            self.tokens = 600 / condition
            self.types = condition
            self.sentence_length = 4

        self.words = self.__randomVocabulary()
        self.stimuli = self.generateStimuli()
        self.stream = "##".join(self.stimuli)
        self.distractors = self.generateDistractors()
        self.test = self.generateTest()

    ''' Generate random stimuli and break it into sentences '''

    def generateStimuli(self):

        stream = ""
        # Repeat until we find a stimuli that satisfies constraints
        while stream == "":
            stream = self.__randomizedStimuli()

        # Break it into sentences
        sentences = self.__groupIntoSentences(tuple(stream))

        # Return
        vprint("Sentences:")
        vprint(sentences)
        return sentences

    ''' Generate test
        30 test pairs, with a word and a partword.
    '''

    def generateTest(self):
        test = []
        vprint("Test: ")
        for _ in range(0, self.TEST_LENGTH):
            r = myRandom.randint(0, len(self.words) - 1)
            w = self.words[r]
            l = len(w)
            pw = self.__getRandomDistractor(l)
            pair = [w, pw]
            if (self.shuffle):
                myRandom.shuffle(pair)
            test.append(pair)
        if (self.shuffle):
            self.test = myRandom.shuffle(test)
        vprint(test)
        return test

    def generateDistractors(self):
        self.distractors = []
        stream = "##".join(self.stimuli)
        syllables = re.findall('..', stream)
        vprint("Distractors: ")

        buff = {}
        for i in (2, 3, 4):
            buff[i] = ["##"] * i
        while len(syllables) > 0:

            # Read next syllable
            if len(syllables) > 0:
                nextSyllable = syllables.pop(0)

            # Add syllable to buffers of all window size
            for i in (2, 3, 4):
                buff[i].pop(0)
                buff[i].append(nextSyllable)

            # See if we have partwords
            for i in (2, 3, 4):
                sequence = "".join(buff[i])
                if ("##" not in sequence) and (sequence not in self.words):
                    self.distractors.append(sequence)
                # Exceptional case of 1 word sentences
                if self.expId == 1 and self.condition == 1 and sequence not in self.words:
                    self.distractors.append(sequence)
        self.distractors = list(set(self.distractors))
        vprint(self.distractors)
        return self.distractors

    ''' Return a random distractor of a certain length'''

    def __getRandomDistractor(self, length):
        vprint("Generating a distractor of length " + str(length))
        distractor = ""
        while len(distractor) != length:
            r = myRandom.randint(0, len(self.distractors) - 1)
            distractor = self.distractors[r]
        return distractor

    '''Returns next randomly chosen word'''

    def __choose_next(self, block):
        r = myRandom.randrange(0, len(block))
        return block[r]

    '''Applies stimuli constraints; both within and between blocks'''

    def __satisfies_constraints(self, stream, word):

        if len(stream) == 0:
            return True
        else:
            # Constraint: actual word cannot be the same as previous
            prevword = stream[-1]
            if word == prevword:
                return False
        return True

    '''Generates the whole stream. Sentences are implicit; the function "breakIntoSentences" will mark the sentence boundaries of this stream.'''

    def __randomizedStimuli(self):
        finalStimuli = []
        unorderedwords = self.words * self.tokens

        iterationnext = unorderedwords[:]
        actsl = 0

        while len(iterationnext) > 0:

            next = self.__choose_next(iterationnext)
            if actsl % self.sentence_length == 0 or self.satisfies_constraints(finalStimuli,
                                                                               next):  # there are no constraints at the beginning of the sentence
                finalStimuli.append(next)
                unorderedwords.remove(next)
                iterationnext = unorderedwords[:]
            else:
                iterationnext.remove(next)

        if len(finalStimuli) == len(self.words) * self.tokens:
            return finalStimuli
        else:
            return ""

    def __randomVocabulary(self):

        lengths = [2, 3, 4]
        if self.types % 3 == 0:
            lengths = lengths * (self.types / 3)
        else:

            # Special case of 4 types
            r = myRandom.randint(0, 2)
            lengths.append(lengths[r])

            # Special case of 5 types
            r = myRandom.randint(0, 2)
            lengths.append(lengths[r])

        vocabulary = []

        for l in lengths:
            word = ""
            for i in range(0, l):
                r = myRandom.randrange(0, len(self.SYLLABLES))
                syllable = self.SYLLABLES[r]
                word += syllable
            vocabulary.append(word)
        vprint("\nWords: ")
        vprint(vocabulary)
        self.words = vocabulary
        return vocabulary

    def __groupIntoSentences(self, stream):
        sentences = []
        stream = list(stream)
        while len(stream) != 0:
            sentence = ""
            for i in range(0, self.sentence_length):
                if len(stream) > 0:
                    sentence += stream.pop(0)
            sentences.append(sentence)
        return sentences


if __name__ == "__main__":
    seed = time.time()
    expId = 1
    exp = experiment(expId)
    for cnd, expcnd in exp.cnd.iteritems():
        print cnd, expcnd.words, expcnd.stimuli
    mm = RnR.RnRv3(A=0.0053, B=0.9488, D=0.8275, munp=1.0, muwp=0.2849, nmax=4)
    performance_exp = dict.fromkeys(EXP_COND[expId])
    for cnd, expcnd in exp.cnd.iteritems():
        print expcnd.stream
        mm.memorizeOnline("##".join(expcnd.stimuli))
        correct = incorrect = 0
        for pair in expcnd.test:
            (prob_x, prob_y, chosen) = mm.Luce(pair[0], pair[1])
            if chosen in expcnd.words:
                correct += 1
            else:
                incorrect += 1
        percentage_correct = correct * 100.0 / (correct + incorrect)
        performance_exp[cnd] = percentage_correct

    personR = exp.pearsonR(performance_exp)
