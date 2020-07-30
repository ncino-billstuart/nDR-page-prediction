import pandas as pd
import numpy as np
import time
import logging
import os, inspect, sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

# local app imports
import app.common as common
                  
                  
def prediction_logic(similarities):

      proba_threshold = .4
      rows = {}
      rows_all = {}
      y_pred_processed_all = pd.DataFrame()
      supported_pages = common.define_supported_pages()
      k1_forms = common.define_k1_forms()
      
      for name, doc in similarities.groupby('file_name'):
          
          rows = {}
          i = 0
          
          for index, row in doc.iterrows():
              i+=1
              
              # if similarity is higher than n
              if round(row['y_pred_similarity'], 2) > proba_threshold:
                  
                  # if page is predicted as "other" assign as "other"
                  if row['y_pred_raw'] == 'other':
                      y_pred = 'other'
                      
                  # specific case where 1065/1120s are very similar
                  elif row['y_pred_raw'] == '1065page4' and row['y_pred_raw_2nd'] == '1120spage3' and row['y_pred_similarity_2nd'] > (proba_threshold - .05) and '1120sp' in rows.get(index - 1, ""):
                      y_pred = row['y_pred_raw_2nd']
                      
                  elif row['y_pred_raw'] == '1120spage3' and row['y_pred_raw_2nd'] == '1065page4' and row['y_pred_similarity_2nd'] > (proba_threshold - .05) and '1065p' in rows.get(index - 1, ""):
                      y_pred = row['y_pred_raw_2nd']
                      
                  # specific case where 1120page5 is similar to state forms (e.g. MS Balance Sheet per Books)
                  elif row['y_pred_raw'] == '1120page5' and '1120p' not in rows.get(index - 1, ""):
                      y_pred = 'other'
                  
                  # specific case where 1065page5 is similar to state forms (e.g. LA Schedule L)
                  elif row['y_pred_raw'] == '1065page5' and '1065p' not in rows.get(index - 1, ""):
                      y_pred = 'other'
                      
                  # specific case where state K-1's are very similar to 1065 k-1
                  # elif row['y_pred_raw'] == 'k11065' and not any(x in k1_forms for x in list(rows.values())):
                      # y_pred = 'other'
                          
                  else: 
                      y_pred = row['y_pred_raw']
                      
              # if similarity is NOT higher than n
              else:
                  y_pred = 'other'
                  
              # append prediction to list
              rows.update({index: y_pred})
              rows_all.update(rows)
              
      # append all predictions for document to dataframe
      y_pred_processed = pd.DataFrame.from_dict(rows_all, orient='index', columns=['y_pred'])
      y_pred_processed['y_pred_processed'] = y_pred_processed['y_pred'].apply(lambda i: i if i in sum(supported_pages.values(), []) else 'other')
      
      return y_pred_processed
      
      
def merge_predictions_to_docs(y_pred_processed, similarities):
      
      similarities = pd.merge(similarities, y_pred_processed, how='inner', left_index=True, right_index=True)
      similarities['y_pred_match'] = np.where(similarities['y_pred_processed'] == similarities['target'], 1, 0)
      similarities['y_pred_processed_form'] = similarities['y_pred_processed'].str.split('page').str[0]

      return similarities
      

def main(similarities):
      
      start = time.time()
      y_pred_processed = prediction_logic(similarities)
      similarities = merge_predictions_to_docs(y_pred_processed, similarities)
      
      logging.info("{0}s, {1}s avg per doc, {2}s per page".format(
            round(time.time() - start, 4), round((time.time() - start)/len(similarities['file_name'].unique()), 4), round((time.time() - start)/len(similarities), 4)))
      return similarities
      
      
