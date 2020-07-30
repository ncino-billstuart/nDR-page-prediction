import pandas as pd
import os, inspect, sys
import time
import logging

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# local app imports
import app.common as common


def identify_duplicate_pages(similarities):
      
      multiple_pages_ok = common.define_multiple_pages_ok()
      
      # removes pages where it is okay to have multiple
      duplicate_pages = similarities[~similarities['y_pred_processed'].isin(multiple_pages_ok)]
      duplicate_pages = duplicate_pages.groupby(['file_name', 'y_pred_processed'])['y_pred_processed'].agg('count')
      duplicate_pages = duplicate_pages[duplicate_pages > 1]

      return duplicate_pages


def identify_missing_pages(similarities):
      
      # post processing to determine if there are missing pages
      rows = []
      
      for name, doc in similarities.groupby('file_name'):
          forms_in_doc = list(doc['y_pred_processed'].str.split('page').str[0].unique())
          pages_in_doc = set(doc['y_pred_processed'].unique())
          
          # map unique list of pages
          supported_pages = common.define_supported_pages()
          pages_needed = [supported_pages[k] for k in forms_in_doc if k in supported_pages]
          pages_needed = set([item for sublist in pages_needed for item in sublist])
      
          missing_pages = list(sorted(pages_needed - pages_in_doc))
          
          # 8825 can be missing page 2
          missing_pages = [i for i in missing_pages if i not in '8825page2']
          if max(doc['year']) < 2018:
              missing_pages = [i for i in missing_pages if i not in '1120page6']
                     
          if missing_pages:
              rows.append({
                  'name': name,
                  'forms_in_doc': [i for i in forms_in_doc if i != 'other'],
                  'pages_in_doc': [i for i in pages_in_doc if i != 'other'],
                  'missing_pages': missing_pages,
              })
      
      # want to preserve the dataframe column order
      if rows:
            columns = list(rows[0].keys())
            missing_pages = pd.DataFrame(rows)[columns]
      
            return missing_pages
      
      else:
            return pd.DataFrame()


def main(similarities):
      
      start = time.time()
      
      duplicate_pages = identify_duplicate_pages(similarities)
      missing_pages = identify_missing_pages(similarities)
      
      logging.info("{0}s, {1}s avg per doc, {2}s per page".format(
            round(time.time() - start, 4), round((time.time() - start)/len(similarities['file_name'].unique()), 4), round((time.time() - start)/len(similarities), 4)))
      
      return missing_pages, duplicate_pages
      
      
if __name__ == '__main__':
      similarities = pd.read_csv('nDR-page-prediction/results/similarities_saved_for_postprocessing.csv')
      
      missing_pages, duplicate_pages = main(similarities)
      
      missing_pages.to_csv('nDR-page-prediction/results/missing_pages.csv')
      duplicate_pages.to_csv('nDR-page-prediction/results/duplicate_pages.csv')