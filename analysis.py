from helpers import *
from pprint import pprint
from scipy import stats
import pandas as pd

# Analysis should always start with reading from a file. To update models or seed
#   words, go to update-models.py
with open('data-as-dict.json', 'r') as outfile:
    data_map = json.load(outfile)

# pprint(data_map)

# Occurences of morphological antonyms in word-embedding models
fasttext_morph_ants = find_morphological_antonyms(data_map['model-fasttext'])
glove_morph_ants = find_morphological_antonyms(data_map['model-glove'])

# pprint(fasttext_morph_ants)
# print('\n\n')
# pprint(glove_morph_ants)
# print('\n\n')

'''Findings:
    - morph_ants are very common in fastText model (23/30)
    - much less common in glove (7/30)
'''



# Occurences of morphological antonyms in data
data = find_morphological_antonyms(data_map['data-full'])
# pprint(data)
# print(len(data))



# Comparing word frequency in antonym choices
word_freq = find_word_freq(data_map['data-full'])

pprint(word_freq)


col = 'freq-absolute'
data = {}
for word, ants in word_freq.items():
    data[word] = {'x': [], 'y': []}

    for ant, props in ants.items():

        data[word]['x'].append(props[col])
        data[word]['y'].append(props['trans-prob'])


# pprint(data)



filename = '6_antonym-elicitation-trials.csv'
ant_list = generate_antonym_list(filename)
trans_map = calculate_transition_prob(ant_list)
# pprint(trans_map)

# Read pandas from csv data
test = pd.read_csv(filename)

# Detect terminal width for dataframe printing
pd.options.display.width = 0

# Keep only these three fields
filtered = test[['positive', 'antonym', 'response']]

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
sorted = sorted.reset_index(drop = True)

# Add word frequencies for response words
sorted = sorted.assign(freq_absolute = sorted['response'].apply(lambda word: zipf_frequency(word, 'en')))

# Add relative word frequencies: response_freq - stimuli_freq
sorted = sorted.assign(freq_relative = sorted['freq_absolute'] - sorted['positive'].apply(lambda word: zipf_frequency(word, 'en')))

print(sorted.tail(15))


'''
NOTES:
    - it is likely worth checking data for typos
        - eg ID1: uncreaitive, unresonable, unfoorgiving
        - maybe worth removing responses that weren't one word (eg not tolerant)
    - look at ID4: Almost all responses were morphological antonyms


    - Hypothesis: participants assume we do not want morphological antonym
        - Try to generate lexical antonym
            - If they cannot, only then do they respond with morph ant
        - Looking at availability/frequency may help to support/disprove

        - Also: look at how word frequencies compare to transition probabilites
            - for morphological antonyms in particular

    - Is there a significant difference in word frequency of positive and negative adjectives?
        - What is the best way to quantify


ASK MH:
    - Do you know of good data for word frequencies?
        - https://www.wordandphrase.info/frequencyList.asp
'''














# When I return, look at:
# http://propor2016.di.fc.ul.pt/wp-content/uploads/2016/07/BrunaThalenbergPROPORSRW2016.pdf
