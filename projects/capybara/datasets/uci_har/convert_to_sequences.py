import pandas as pd
import itertools
import os
import shutil
from collections import Counter

HELP_TEXT = """
  Individual time series files were stored with a two levels folder structure:

      %s/
          category_1_folder/
              file_1.txt
              file_2.txt
              ...
              file_42.txt
          category_2_folder/
              file_43.txt
              file_44.txt
              ...
"""



def convert_to_sequences(csv_path, output_dir):
  """
  Look for homogeneous chunk of time series with the same label.
  
  Individual time series files will be stored with a two levels folder
  structure such as the following:

      container_folder/
          category_1_folder/
              file_1.txt
              file_2.txt
              ...
              file_42.txt
          category_2_folder/
              file_43.txt
              file_44.txt
              ...
              

  :param csv_path: (str) path to the input data to convert.
  :param output_dir: (str) path to the output directory
  """

  df = pd.read_csv(csv_path)

  # Create output dir
  labels = list(df.label.unique())
  if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

  # Group time series by label chunks  
  metrics = list(df.columns.values)
  if 'label' in metrics:  metrics.remove('label')
  if 't' in metrics:   metrics.remove('t')

  for metric in metrics:
    grouping = itertools.groupby(zip(df.label, df[metric]), lambda x: x[0])

    for label in labels:
      os.makedirs(os.path.join(output_dir, metric, str(label)))

    counter = Counter()
    for label, group in grouping:
      values = [g[1] for g in group]
      df_out = pd.DataFrame(data={'value': values, 't': range(len(values))})

      output_file = 'series_%s.txt' % counter[label]
      output_path = os.path.join(output_dir, metric, str(label), output_file)
      df_out.to_csv(output_path, index=False)
      counter[label] += 1

  print HELP_TEXT % output_dir



def test_data_import():
  """Just testing that the conversion worked."""
  from sklearn.datasets import load_files
  train_dataset = load_files('inertial_signals_train/body_acc_x')
  test_dataset = load_files('inertial_signals_train/body_acc_x')

  X_train = train_dataset['data']
  y_train = train_dataset['target']
  assert len(X_train) == len(y_train)

  X_test = test_dataset['data']
  y_test = test_dataset['target']
  assert len(X_test) == len(y_test)



if __name__ == '__main__':
  csv_paths = ['debug_train.csv',
               'debug_test.csv',
               'inertial_signals_train.csv',
               'inertial_signals_test.csv']

  for csv_path in csv_paths:
    output_dir = os.path.basename(csv_path)[:-4]
    convert_to_sequences(csv_path, output_dir)

  test_data_import()
