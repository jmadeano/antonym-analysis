from helpers import *
from pprint import pprint

# STANDARD UPDATE WORKFLOW
# Change the filename when we get new data
filename = '6_antonym-elicitation-trials.csv'
ant_list = generate_antonym_list(filename)
trans_map = calculate_transition_prob(ant_list)
likely_trans = find_most_likely_transition(trans_map)


# Uncomment the section below in order to try new models or change the list of seed words.
#   Review the documentation for find_similar_words in helpers.py if necessary.
# ------------START SECTION---------------------------------------------------------------

# embedding_sim = find_similar_words('fasttext-wiki-news-subwords-300', ant_list.keys())
#
# data_map = {}
# data_map['data-likely'] = likely_trans
# data_map['data-full'] = trans_map
# # NOTE: Make sure to change the model label matches your model
# data_map['model-fasttext'] = convert_similar_words_to_map(embedding_sim)
#
# pprint(data_map)
#
# # It is a good idea to check the print out before writing to a file (especially if you
# #   plan on overwriting previous data).

# ------------END SECTION-----------------------------------------------------------------



# Uncomment in order to update json data (This should be done when updating the
#     model since it takes ~10mins to calculate similarities)
# ------------START SECTION---------------------------------------------------------------

# # NOTE: It is a good idea to change the filename each time you uncomment this
# #       in order to avoid overwriting previous data!
# with open('new-data.json', 'w') as outfile:
#     json.dump(data_map, outfile)

# ------------END SECTION-----------------------------------------------------------------
