import argparse
import itertools
import os
import numpy as np
import pandas as pd



def find_phase_name(file_path):
  """
  Find the phase name ('test', 'train' or 'val') in the base name of a file.
  If not found then return 'all'.
  
  @param file_path: (str) path to file.
  @return phase: (str) phase name
  """
  base_name = os.path.basename(file_path)
  phases = ['train', 'test', 'val']
  for phase in phases:
    if phase in base_name.lower():
      return phase
  return 'all'



def convert_to_sequences(csv_path, parent_output_dir, phase, chunk_size):
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

  df = pd.read_csv(csv_path)

  # Group time series by label chunks  
  metrics = list(df.columns.values)
  if 'label' in metrics:  metrics.remove('label')
  if 't' in metrics:   metrics.remove('t')

  for metric in metrics:

    # Create output dir if it does not already exists
    output_dir = os.path.join(parent_output_dir, metric)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    txt_file = os.path.join(output_dir, '%s_%s' % (metric, phase.upper()))

    # Find sequences with chunks of homogeneous labels
    grouping = itertools.groupby(zip(df.label, df[metric]),
                                 lambda x: x[0])

    txt_rows = []
    for label, groups in grouping:

      try:
        label = int(label)
      except:
        print 'skip label:', label
        continue

      ts_values = []
      for group in groups:
        ts_values.append(float(group[1]))
        if len(ts_values) % chunk_size == 0:
          txt_rows.append([int(label)] + ts_values)
          ts_values = []

      # If txt_rows is empty (meaning that no sequence was long 
      # enough for this label), then use ts_values seen so far, if any.
      if len(ts_values) > 0 and len(txt_rows) == 0:
        txt_rows.append([int(label)] + ts_values)

    np.savetxt(txt_file, txt_rows, delimiter=',', fmt='%.10f')



if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--input_file', '-i',
                      dest='input_file',
                      type=str,
                      default=os.path.join(
                        os.getcwd(), 'uci_har', 'inertial_signals_train.csv'))

  parser.add_argument('--output_dir', '-o',
                      dest='output_dir',
                      default=os.path.join(
                        os.getcwd(), 'uci_sequences', 'inertial_signals'),
                      type=str)

  parser.add_argument('--chunk_size', '-c',
                      dest='chunk_size',
                      default=100,
                      type=int)

  options = parser.parse_args()
  csv_path = options.input_file
  output_dir = options.output_dir
  chunk_size = options.chunk_size

  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  phase = find_phase_name(csv_path)
  convert_to_sequences(csv_path, output_dir, phase, chunk_size)

  print 'Path to converted files: %s/' % output_dir
