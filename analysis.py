from helpers import *
from pprint import pprint

ant_list = generate_antonym_list('6_antonym-elicitation-trials.csv')
trans_map = calculate_transition_prob(ant_list)
likely_trans = find_most_likely_transition(trans_map)
# embedding_sim = find_similar_words('fasttext-wiki-news-subwords-300', ant_list.keys())


pprint(trans_map)

# ## UPDATE ALL JSON DATA
# with open('all_data.txt', 'r') as outfile:
#     data_map = json.load(outfile)
#
# data_map = {}
#
# data_map['data-likely'] = likely_trans
# data_map['data-full'] = trans_map
# data_map['model-glove'] = embedding_sim
# data_map['model-fasttext'] = embedding_sim
#
# with open('data.json', 'w') as outfile:
#     json.dump(data_map, outfile)


with open('data-as-dict.json', 'r') as outfile:
    data_map = json.load(outfile)

# # pprint(list_sim)
# data_map['model-fasttext'] = convert_similar_words_to_map(data_map['model-fasttext'])
# data_map['model-glove'] = convert_similar_words_to_map(data_map['model-glove'])

pprint(data_map)
