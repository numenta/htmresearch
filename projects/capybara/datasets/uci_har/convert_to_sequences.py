import argparse
import itertools
import os
import shutil
import numpy as np
import pandas as pd



def convert_to_sequences(input_dir, base_name, phase, chunk_size=100):
  """
  Look for homogeneous chunk of time series with the same label.
  
  Individual time series files will be stored with a two levels folder
  structure such as the following:

      container_folder/
          metric_1/
            metric_1_TRAIN.txt
            metric_1_TEST.txt
            
            ...
          
          metric_N/
            metric_N_TRAIN.txt
            metric_N_TRAIN.txt
              
  Each .txt file has the following structure: 
      label, metric_1(t1), ..., metric_1(tN)
        ...
      label, metric_N(t1), ..., value(tM)

  Where N is the number of metrics, M is the number of timesteps, 
  and metric_N(tM) is the value of metric_N at time tM.
  """

  csv_path = os.path.join(input_dir, '%s_%s.csv' % (base_name, phase))

  df = pd.read_csv(csv_path)

  # Group time series by label chunks  
  metrics = list(df.columns.values)
  if 'label' in metrics:  metrics.remove('label')
  if 't' in metrics:   metrics.remove('t')

  for metric in metrics:

    # Create output dir
    output_dir = os.path.join(base_name, metric)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    txt_file = os.path.join(output_dir, '%s_%s' % (metric, phase.upper()))

    # Find sequences with chunks of homogeneous labels
    grouping = itertools.groupby(zip(df.label, df[metric]),
                                 lambda x: x[0])

    txt_rows = []
    for label, groups in grouping:

      ts_values = []
      for group in groups:
        ts_values.append(group[1])
        if len(ts_values) % chunk_size == 0:
          txt_rows.append([int(label)] + ts_values)
          ts_values = []

      # If txt_rows is empty (meaning that no sequence was long 
      # enough for this label), then use ts_values seen so far, if any.
      if len(ts_values) > 0 and len(txt_rows) == 0:
        txt_rows.append([int(label)] + ts_values)

    np.savetxt(txt_file, txt_rows, delimiter=',', fmt='%.10f')



if __name__ == '__main__':
  base_names = ['inertial_signals', 'debug']
  phases = ['train', 'test']

  # Parse input options.
  parser = argparse.ArgumentParser()
  parser.add_argument('--input_dir', '-i',
                      dest='input_dir',
                      default=os.getcwd(),
                      type=str)
  parser.add_argument('--chunk_size', '-c',
                      dest='chunk_size',
                      default=100,
                      type=int)

  options = parser.parse_args()
  input_dir = options.input_dir
  chunk_size = options.chunk_size

  for base_name in base_names:

    if os.path.exists(base_name):  # clean and create parent output dir
      shutil.rmtree(base_name)

    for phase in phases:
      convert_to_sequences(input_dir, base_name, phase, chunk_size)

    print 'Path to converted files: %s/' % base_name
