from helpers import *
from pprint import pprint

# Analysis should always start with reading from a file. To update models or seed
#   words, go to update-models.py
with open('data-as-dict.json', 'r') as outfile:
    data_map = json.load(outfile)

# pprint(data_map)

# Occurences of morphological antonyms in word-embedding models
fasttext_morph_ants = find_morphological_antonyms(data_map['model-fasttext'])
glove_morph_ants = find_morphological_antonyms(data_map['model-glove'])

pprint(fasttext_morph_ants)
print('\n\n')
pprint(glove_morph_ants)
print('\n\n')

# Findings:
# morph_ants are very common in fastText model (23/30)
# much less common in glove (7/30)



# Occurences of morphological antonyms in data
data = find_morphological_antonyms(data_map['data-full'])
pprint(data)
print(len(data))


# it is likely worth checking data for typos
# eg ID1: uncreaitive, unresonable, unfoorgiving
# look at ID4: Almost all responses were morphological antonyms

















# When I return, look at:
# http://propor2016.di.fc.ul.pt/wp-content/uploads/2016/07/BrunaThalenbergPROPORSRW2016.pdf
