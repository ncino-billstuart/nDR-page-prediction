import pandas as pd
import time
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from tabulate import tabulate
import os, inspect, sys

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


def add_features(similarities):

      similarities['y_pred_match'] = np.where(similarities['y_pred_processed'] == similarities['target'], 1, 0)
      similarities['y_pred_processed_form'] = similarities['y_pred_processed'].str.split('page').str[0]
      similarities['file_num'] = similarities['file_name'].rank(method='dense')
      
      return similarities


def create_visuals(similarities):
      
      # precision: proportion of positive predictions actually correct
      # recall: proportion of actual positives was identified correctly

      report = classification_report(similarities['target'], similarities['y_pred_processed'], output_dict=True)
      report = pd.DataFrame(report).transpose().round(2).rename(columns={'support': 'pages'})
      
      columns = sorted(set(list(similarities['target'].unique()) + list(similarities['y_pred_processed'].unique())))
      
      cm = confusion_matrix(similarities['target'], similarities['y_pred_processed'])
      plt.figure(figsize=(15, 5))
      f = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=columns, yticklabels=columns)
      plt.savefig(parentdir + '/results/confusion_matrix.png')
      
      cm = confusion_matrix(similarities['target'], similarities['y_pred_processed'], normalize='true').round(2)
      plt.figure(figsize=(15, 5))
      f = sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=columns, yticklabels=columns)
      plt.savefig(parentdir + '/results/confusion_matrix_normalized.png')
      
      print(tabulate(report, headers='keys', tablefmt='psql'))
      
      return report


def calculate_page_performance(similarities):

      labeled = similarities[similarities['target'] != 'other']
      recall = round(accuracy_score(labeled['target'], labeled['y_pred_processed']), 4)
      
      labeled = similarities[similarities['y_pred_processed'] != 'other']
      precision = round(accuracy_score(labeled['target'], labeled['y_pred_processed']), 4)
      
      labeled = similarities[(similarities['target'] != 'other') | (similarities['y_pred_processed'] != 'other')]
      non_other_accuracy = round(accuracy_score(labeled['target'], labeled['y_pred_processed']), 4)
      
      overall_accuracy = round(accuracy_score(similarities['target'], similarities['y_pred_processed']), 4)
      
      return recall, precision, non_other_accuracy, overall_accuracy


def calculate_doc_performance(similarities):

      doc_accuracy = similarities.groupby(['file_name'])['y_pred_match'].mean()
      total_docs = len(doc_accuracy)
      no_error_docs = len(doc_accuracy[doc_accuracy == 1])
      error_docs = len(doc_accuracy[doc_accuracy != 1])
      
      return total_docs, error_docs


def determine_error_type(similarities):

      def error_type(row):
          if row['target'] == row['y_pred_processed']:
              return 'Correct'
          if row['target'] == 'other' and row['y_pred_processed'] != 'other':
              return 'False Positive'
          elif row['target'] != 'other' and row['y_pred_processed'] == 'other':
              return 'False Negative'
          elif row['target'] != 'other' and row['y_pred_processed'] != 'other':
              return 'Misclassification'
          else:
              return 'verify'

      similarities['error_type'] = similarities.apply(lambda row : error_type(row), axis = 1)
      
      return similarities
      
      
def main(similarities):
      start = time.time()
      
      similarities = add_features(similarities)
      create_visuals(similarities)
      recall, precision, non_other_accuracy, overall_accuracy = calculate_page_performance(similarities)
      total_docs, error_docs = calculate_doc_performance(similarities)
      similarities = determine_error_type(similarities)
      
      logging.info("{0}s, recall:{1}, precision: {2}, non-other acc: {3}, overall acc: {4}".format(
            round(time.time() - start, 4), recall, precision, non_other_accuracy, overall_accuracy))
      logging.info('{} docs of {} total have errors'.format(error_docs, total_docs))
      
      return similarities