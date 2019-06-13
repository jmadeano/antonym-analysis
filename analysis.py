from helpers import *
from pprint import pprint

# Analysis should always start with reading from a file. To update models or seed
#   words, go to update-models.py
with open('data-as-dict.json', 'r') as outfile:
    data_map = json.load(outfile)

# pprint(data_map)

# Occurences of morphological antonyms in word-embedding models
def find_morph_ants_in_similar_words(model):
    morph_ants_in_sim_words = {}

    for (word, ants) in model.items():
        for ant, sim in ants.items():
            if is_morphological_negation(word, ant):
                morph_ants_in_sim_words[word] = sim

    return morph_ants_in_sim_words

# ADD: ab-normal, a-typical


fasttext_morph_ants = find_morph_ants_in_similar_words(data_map['model-fasttext'])
glove_morph_ants = find_morph_ants_in_similar_words(data_map['model-glove'])

pprint(fasttext_morph_ants)
print('\n\n')
pprint(glove_morph_ants)

# Findings:
# morph_ants are very common in fastText model (23/30)
# much less common in glove (7/30)



# Occurences of morphological antonyms in data
# Note this does not check if it is the CORRECT morphological negation
# compassionate => incompassionate vs uncompassionate, still counts as morph ant
def find_morph_ants_in_data(data, all = False): # all means look for all morphological negations (not just of original word)
    morph_ants_in_sim_words = {}

    for (word, ants) in data.items():
        for ant, prob in ants.items():
            if is_morphological_negation(word, ant):
                morph_ants_in_sim_words[word] = prob

    return morph_ants_in_sim_words



print('\n\n')
data = find_morph_ants_in_data(data_map['data-full'])
pprint(data)
print(len(data))


# it is likely worth checking data for typos
# eg ID1: uncreaitive, unresonable, unfoorgiving
# look at ID4: Almost all responses were morphological antonyms

















# When I return, look at:
# http://propor2016.di.fc.ul.pt/wp-content/uploads/2016/07/BrunaThalenbergPROPORSRW2016.pdf
