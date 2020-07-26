import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import logging

# local app imports
import preprocess_data
import store_and_load_functions


def vectorize_templates(templates):

    vect = TfidfVectorizer(lowercase=False, stop_words='english', ngram_range=(1, 3), min_df=1, max_features=None)
    vect_templates = vect.fit_transform(templates['text'])
    templates_columns = list(templates['type'])
    
    return vect_templates, vect, templates_columns
    
    
def main(templates):
    
    start = time.time()
    
    for name, group in templates.groupby(['year']):
        vect_templates, vect, templates_columns = vectorize_templates(group)
        store_and_load_functions.store_pickle(vect_templates, path='nDR-page-prediction/vectorized_templates/' + str(name) + '/', file_name='vect_templates')
        store_and_load_functions.store_pickle(vect, path='nDR-page-prediction/vectorized_templates/' + str(name) + '/', file_name='vect')
        store_and_load_functions.store_pickle(templates_columns, path='nDR-page-prediction/vectorized_templates/' + str(name) + '/', file_name='templates_columns')

    logging.info("{}s".format(round(time.time() - start, 4)))

if __name__ == '__main__':
    
    docs, templates = preprocess_data.main()
    main(templates)