import pandas as pd
import numpy as np
import time
import logging

def create_helper_objects():

      supported_pages = {
             'other': ['other'],  # can have multiple
             '1120s': ['1120spage1', '1120spage3', '1120spage4', '1120spage5'],
             '1040': ['1040page1', '1040page2'], 
             'schedulec': ['schedulecpage1', 'schedulecpage2'],  # can have multiple
             'schedule': ['scheduleepage1'],  # can have multiple  # can be missing page 2
             '1065': ['1065page1', '1065page4', '1065page5'],
             'k11065': ['k11065page1'],  # can have multiple and be uploaded with 1065 or with a 1040
             '8825': ['8825page1', '8825page2'],  # can have multiple  # can be missing page 2
             '1120': ['1120page1', '1120page5', '1120page6'],  # page 6 added in 2018
             'schedule1': ['schedule1page1'],
             'schedule2': ['schedule2page1'],
             'schedule3': ['schedule3page1'],
             'schedule4': ['schedule4page1'],
                        }
                        
      k1_forms = ['schedulecpage1', '1040page1']
      
      return supported_pages, k1_forms
                  
                  
def prediction_logic(similarities, supported_pages, k1_forms):

      proba_threshold = .4
      rows_all = {}
      y_pred_processed_all = pd.DataFrame()
      
      for name, doc in similarities.groupby('file_name'):
          
          rows = {}
          
          for index, row in doc.iterrows():
              
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
                      # print("{}: door 5".format(y_pred))
                      
              # if similarity is NOT higher than n
              else:
                  y_pred = 'other'
                  # print("{}: door 6".format(y_pred))
                  
              # append prediction to list
              rows.update({index: y_pred})
              rows_all.update(rows)
              
      # append all predicitons for document to dataframe
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
      supported_pages, k1_forms = create_helper_objects()
      y_pred_processed = prediction_logic(similarities, supported_pages, k1_forms)
      similarities = merge_predictions_to_docs(y_pred_processed, similarities)
      
      logging.info("{}s".format(round(time.time() - start, 4)))
      return similarities
      
      
