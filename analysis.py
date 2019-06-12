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
        for ant in ants.keys():
            if is_morphological_negation(word, ant):
                morph_ants_in_sim_words[word] = True

    return morph_ants_in_sim_words




fasttext_morph_ants = find_morph_ants_in_similar_words(data_map['model-fasttext'])
glove_morph_ants = find_morph_ants_in_similar_words(data_map['model-glove'])

pprint(fasttext_morph_ants)
print('\n\n')
pprint(glove_morph_ants)


# Findings:
# morph_ants are very common in fastText model (23/30)
# much less common in glove (7/30)




# When I return, look at:
# http://propor2016.di.fc.ul.pt/wp-content/uploads/2016/07/BrunaThalenbergPROPORSRW2016.pdf
