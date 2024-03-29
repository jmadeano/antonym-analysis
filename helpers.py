from wordfreq import zipf_frequency
import gensim.downloader as api
import numpy as np
import csv
import json
from scipy import stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

def find_similar_words(word_embedding, words):
    '''
    Given a list of a words, find the ten most similar words to each. Return a dictionary
    mapping each word in the input list to the similar words along with their similarity.
    Common word embeddings include 'fasttext-wiki-news-subwords-300' and 'glove-wiki-gigaword-200'.

    Example:
        {'adventurous':              [['adventuresome', 0.673180103302002],
                                     ['inventive', 0.5974040627479553],
                                     ['imaginative', 0.5858909487724304],
                                     ['enterprising', 0.5562216639518738],
                                     ['musically', 0.5521135330200195],
                                     ['impetuous', 0.5404343008995056],
                                     ['inquisitive', 0.5328224897384644],
                                     ['venturesome', 0.5321169495582581],
                                     ['enjoyable', 0.5309233069419861],
                                     ['offbeat', 0.5194555521011353]],
         'affectionate':              [['playful', 0.6456809639930725],
                                      ['respectful', 0.6125648021697998],
                                      ['sarcastic', 0.6028381586074829],
                                      ['affection', 0.5752658247947693],
                                      ['sardonic', 0.5718863010406494],
                                      ['loving', 0.5700308084487915],
                                      ['endearing', 0.5636808276176453],
                                      ['polite', 0.5526844263076782],
                                      ['wry', 0.5466963648796082],
                                      ['irreverent', 0.5442217588424683]]}

    Keyword arguments:
    word_embedding -- the gensim word embedding model, see https://github.com/RaRe-Technologies/gensim-data
    words -- the list of words to find similar words of
    '''

    info = api.info()  # show info about available models/datasets
    model = api.load(word_embedding)  # download the model and return as object ready for use

    embedding_similarity = {}

    for word in words:
        embedding_similarity[word] = model.most_similar(word)

    with open('embedding-data.json', 'w') as outfile:
        json.dump(embedding_similarity, outfile)

    return embedding_similarity



def convert_similar_words_to_map(gensim_similar_list):
    '''
    Given a mapping of similar words (as returned by find_similar_words), change the structure such that
    each similar word maps to its similarity rating (rather than the original list structure).
    Return a dictionary of dictionaries.

    Example:
        {'adventurous': {'adventuresome': 0.673180103302002,
                                 'enjoyable': 0.5309233069419861,
                                 'enterprising': 0.5562216639518738,
                                 'imaginative': 0.5858909487724304,
                                 'impetuous': 0.5404343008995056,
                                 'inquisitive': 0.5328224897384644,
                                 'inventive': 0.5974040627479553,
                                 'musically': 0.5521135330200195,
                                 'offbeat': 0.5194555521011353,
                                 'venturesome': 0.5321169495582581},
         'affectionate': {'affection': 0.5752658247947693,
                                  'endearing': 0.5636808276176453,
                                  'irreverent': 0.5442217588424683,
                                  'loving': 0.5700308084487915,
                                  'playful': 0.6456809639930725,
                                  'polite': 0.5526844263076782,
                                  'respectful': 0.6125648021697998,
                                  'sarcastic': 0.6028381586074829,
                                  'sardonic': 0.5718863010406494,
                                  'wry': 0.5466963648796082}}

    Keyword arguments:
    gensim_similar_list -- the similar words data structure returned by find_similar_words
    '''


    for word, sim_words in gensim_similar_list.items():
        gensim_similar_list[word] = {sim_word[0]: sim_word[1] for sim_word in sim_words}

    return gensim_similar_list


def generate_antonym_list(filename, generation = 0, preprocess = True):
    '''
    Given the filename of a csv containing experiment results, generate and return a map from all
    of the input words to a list of all the antonyms that were generated from them (contains duplicates).
    Optionally specify which generation to look at (otherwise compile data from all generation).
    All words are preprocessed (convert to lowercase + remove leading/trailing whitespace) by default.

    Example:
        {'adventurous': ['cautious',
                         'boring',
                         'boring',
                         'homebody',
                         'unadventurous',
                         'shy',
                         'lame',
                         'reserved',
                         'timid',
                         'sheltered',
                         'safe',
                         'home',
                         'prudent',
                         'conventional',
                         'cautious',
                         'boring',
                         'withdrawn',
                         'dull'],
         'affectionate': ['mean',
                          'cold',
                          'cold',
                          'glum',
                          'unaffectionate',
                          'cold',
                          'cold',
                          'aloof',
                          'cold',
                          'cold',
                          'unaffectionate',
                          'hateful',
                          'cold',
                          'unaffectionate',
                          'unaffectionate',
                          'cold',
                          'distant',
                          'unaffectionate']}

    Keyword arguments:
    filename -- the filename of the csv data
    generation -- optional parameter to only parse antonyms generated during the nth generation,
                  the function will look at data from all generations if the parameter is unspecified
    preprocess -- optional parameter, False => disable preprocessing
    '''

    antonym_list = {}

    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:


            word = row['positive'] # presented word
            ant = row['response'] # antonym response

            # lowercase and strip whitespace from response (standardize)
            if preprocess:
                word = word.lower().strip()
                ant = ant.lower().strip()

            # make a list of all antonyms for a each word (w/ repeats)
            antonym_list[word] = antonym_list.get(word, [])
            antonym_list[word].append(ant)

    return antonym_list


def calculate_transition_prob(ant_list):
    '''
    Given a map of antonyms generated with generate_antonym_list, calculate the transition probabilties for all
    presented words. This process removes duplicate antonyms in and return a dictionary of dictionaries.
    The outer dictionary maps a word to its antonyms and the inner dictionary maps each antonym to high
    likely it was to occur given the original word. For each antonym, the transition probabilities
    should sum to 1.

    Example:
        {'adventurous': {'boring': 0.16666666666666666,
                         'cautious': 0.1111111111111111,
                         'conventional': 0.05555555555555555,
                         'dull': 0.05555555555555555,
                         'home': 0.05555555555555555,
                         'homebody': 0.05555555555555555,
                         'lame': 0.05555555555555555,
                         'prudent': 0.05555555555555555,
                         'reserved': 0.05555555555555555,
                         'safe': 0.05555555555555555,
                         'sheltered': 0.05555555555555555,
                         'shy': 0.05555555555555555,
                         'timid': 0.05555555555555555,
                         'unadventurous': 0.05555555555555555,
                         'withdrawn': 0.05555555555555555},
          'affectionate': {'aloof': 0.05555555555555555,
                          'cold': 0.4444444444444444,
                          'distant': 0.05555555555555555,
                          'glum': 0.05555555555555555,
                          'hateful': 0.05555555555555555,
                          'mean': 0.05555555555555555,
                          'unaffectionate': 0.2777777777777778}}


    Keyword arguments:
    ant_list -- a mapping from word to all of its antonyms that were generated, see generate_antonyms_list
    '''
    transition_map = {}

    for word, ants in ant_list.items():
        probabilities = {}

        for ant in ants:
            # transition probability is (# occurences of antonym i / total number of antonyms)
            probabilities[ant] = probabilities.get(ant, ants.count(ant)/len(ants))

        transition_map[word] = probabilities

    return transition_map


def find_most_likely_transition(trans_map):
    '''
    Given a map transition probabilties generated using calculate_transition_prob, map each word to
    its most likely antonym. If multiple antonyms are equally likely, only one of them will be returned.

    Example:
        {'adventurous': 'boring',
         'affectionate': 'cold'}


    Keyword arguments:
    trans_map -- a mapping of transition probabilities, see calculate_transition_prob
    '''
    likely_trans = {}

    for word, ants in trans_map.items():
        likely_trans[word] = max(ants, key=ants.get)

    return likely_trans


def is_morphological_negation(word, antonym, lowercase = True):
    '''
    Given an original word and an antonym, check if the antonym is a morphological negation
    of the word (in-, im-, un-, dis-, ir-, ab-, a-  + word). Returns a boolean.

    Keyword arguments:
    word -- the origial word
    antonym -- an antonym of word that may be a morphological antonym of word
    lowercase -- optional parameter, when lowercase = false the words are not lowercased
    '''
    if lowercase:
        antonym = antonym.lower()
        word = word.lower()

    for prefix in ('ir', 'in', 'im', 'un', 'dis', 'ab', 'a'):
        p_len = len(prefix)

        # antonym - prefix = word and antonym starts with prefix
        if antonym[p_len:] == word and antonym[:p_len] == prefix:
            return True

    return False

    # # TESTS
    # tests = [('heaLthy', 'Unhealthy'), ('rational', 'irrational'),
    #          ('honest', 'dishonest'), ('possible', 'impossible'),
    #          ('different', 'indifferent'), ('honest', 'disshonest'),
    #          ('normal', 'abnormal'), ('typical', 'atypical')]


def find_morphological_antonyms(data, all = False):
    '''
    Given some data (either model similarities generated by convert_similar_words_to_map() or
    experimental data generated by calculate_transition_prob()), find all of the morphological
    antonyms. For the model similarities case, each word maps to the the similarity value of the
    morphological antonym. For the experimental data case, each word maps to the transition
    probability for its morphological antonym. In either case, if the word's morphological
    antonym does not occur, it is not included in the map (thus the length of the output is
    indicative of how many unique morphological antonyms occurred).

    Note that this function does not check if the morphological antonym is correct. For example,
    if the word is compassionate, incompassionate would be marked as a valid morphological antonym
    even though uncompassionate is the correct version (ie the function cares about intent over
    correctness). In the case where a correct and incorrect version (incompassionate AND
    uncompassionate) occur, the larger probability is chosen. This follows from the fact that the
    morphological antoynm with the correct prefix should always be the most common/similar.

    Keyword arguments:
    data -- either word similarity data (from convert_similar_words_to_map()) or
                experimental data (from calculate_transition_prob())
    all -- optional parameter, if True, look for ALL morphological negations present in the responses
                (ie inventive => uncreative); THIS FUNCTION IS NOT CURRENTLY IMPLEMENTED
    '''

    morph_ants = {}

    for (word, ants) in data.items():
        for ant, prob_or_sim in ants.items():
            if is_morphological_negation(word, ant):
                # if there are multiple morphological antonyms, sum them
                if prob_or_sim > morph_ants.get(word, 0):
                    morph_ants[word] = prob_or_sim

    return morph_ants


def find_word_freq(data, keep_all = False):
    '''
    Given some data (in the format returned by calculate_transition_prob()), append word frequency
    data. The returned dictionary is triply nested (the input is doubly nested). The transition
    probability from the original data is now stored with the 'trans-prob' key. Additionally,
    two more fields ('freq-absolute' and 'freq-relative') have also been added. Both metrics
    rely on the Zipf frequency as returned by the Python package "wordfreq" (see:
    https://pypi.org/project/wordfreq/). The absolute metric is the frequency of the specific
    antonym whereas the relative metric is the Zipf frequency of antonym minus Zipf frequency of
    the stimulus word. Thus, a positive relative frequency means the antonym is more common than
    the presented word and a negative relative frequency means less common.

    NOTE: Some extremely rare words do not show up in the "wordfreq" package (and thus assigned an
    absolute frequency of 0). This function removes all such words unless keep_all is True.

    Example:
    {'adventurous': {'boring': {'freq-absolute': 4.39,
                                'freq-relative': 0.9299999999999997,
                                'trans-prob': 0.16666666666666666},
                     'cautious': {'freq-absolute': 3.81,
                                  'freq-relative': 0.3500000000000001,
                                  'trans-prob': 0.1111111111111111},
                     'conventional': {'freq-absolute': 4.3,
                                      'freq-relative': 0.8399999999999999,
                                      'trans-prob': 0.05555555555555555},
                     'dull': {'freq-absolute': 3.91,
                              'freq-relative': 0.4500000000000002,
                              'trans-prob': 0.05555555555555555},
                     'home': {'freq-absolute': 5.74,
                              'freq-relative': 2.2800000000000002,
                              'trans-prob': 0.05555555555555555},
                     'homebody': {'freq-absolute': 1.98,
                                  'freq-relative': -1.48,
                                  'trans-prob': 0.05555555555555555},
                     'lame': {'freq-absolute': 3.96,
                              'freq-relative': 0.5,
                              'trans-prob': 0.05555555555555555},
                     **etc.},
 'affectionate': {'aloof': {'freq-absolute': 2.92,
                            'freq-relative': -0.3500000000000001,
                            'trans-prob': 0.05555555555555555},
                  'cold': {'freq-absolute': 5.03,
                           'freq-relative': 1.7600000000000002,
                           'trans-prob': 0.4444444444444444},
                  'distant': {'freq-absolute': 4.2,
                              'freq-relative': 0.9300000000000002,
                              'trans-prob': 0.05555555555555555},
                  'glum': {'freq-absolute': 2.52,
                           'freq-relative': -0.75,
                           'trans-prob': 0.05555555555555555},
                  'hateful': {'freq-absolute': 3.38,
                              'freq-relative': 0.10999999999999988,
                              'trans-prob': 0.05555555555555555},
                  'mean': {'freq-absolute': 5.53,
                           'freq-relative': 2.2600000000000002,
                           'trans-prob': 0.05555555555555555},
                  'unaffectionate': {'freq-absolute': 1.12,
                                     'freq-relative': -2.15,
                                     'trans-prob': 0.2777777777777778}}}

    Keyword arguments:
    data -- experimental data (from calculate_transition_prob())
    keep_all -- optional parameter, if True, all antonyms will be kept even if
                    their absolute frequency is 0 (untracked by "wordfreq")
    '''
    word_freq_data = {}
    for word, ants in data.items():
        wordf = zipf_frequency(word, "en")
        ants_data = {}

        for ant, prob in ants.items():
            antf = zipf_frequency(ant, "en")

            if antf == 0:
                if keep_all:
                    ants_data[ant] = {'freq-relative': antf-wordf,
                                       'freq-absolute': antf,
                                       'trans-prob': prob}
            else:
                ants_data[ant] = {'freq-relative': antf-wordf,
                                       'freq-absolute': antf,
                                       'trans-prob': prob}

        word_freq_data[word] = ants_data
    return word_freq_data


def make_dataframe(filename):
    # Read pandas from csv data
    init = pd.read_csv(filename)

    # Detect terminal width for dataframe printing
    pd.options.display.width = 0

    # Keep only these three fields... Be careful (names will change after experiment)
    filtered = init[['positive', 'antonym', 'response']]

    # Strip leading/trailing whitespace and lowercase all stimuli and responses
    filtered['response'] = filtered['response'].apply(lambda word: word.lower().strip())
    filtered['positive'] = filtered['positive'].apply(lambda word: word.lower().strip())

    # Sort by stimuli and then response
    sorted = filtered.sort_values(['positive', 'response'])

    # Add a column for response count
    sorted = sorted.assign(ant_count = sorted.groupby(['positive', 'response']).response.transform('count'))

    # Remove duplicates (because we have a count)
    sorted  = sorted.drop_duplicates()

    # Calculate transition probability: specific_antonym.count()/all_antonyms.count()
    sorted = sorted.assign(trans_prob = sorted.groupby('positive').transform(lambda x: x/x.sum()))
    sorted = sorted.reset_index(drop = True)# Add bool column for morphological antonyms

    # Add bool column for morphological antonyms
    sorted = sorted.assign(is_morph = sorted['response'] == sorted['antonym'])

    # Add word frequencies for response words
    sorted = sorted.assign(freq_absolute = sorted['response'].apply(lambda word: zipf_frequency(word, 'en')))

    # Add relative word frequencies: response_freq - stimuli_freq
    sorted = sorted.assign(freq_relative = sorted['freq_absolute'] - sorted['positive'].apply(lambda word: zipf_frequency(word, 'en')))

    return sorted
