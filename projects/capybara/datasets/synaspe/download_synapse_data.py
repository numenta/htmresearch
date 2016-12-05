#!/usr/bin/env python
# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2016, Numenta, Inc.  Unless you have an agreement
# with Numenta, Inc., for a separate license for this software code, the
# following terms and conditions apply:
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero Public License version 3 as
# published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU Affero Public License for more details.
#
# You should have received a copy of the GNU Affero Public License
# along with this program.  If not, see http://www.gnu.org/licenses.
#
# http://numenta.org/licenses/
# ----------------------------------------------------------------------

"""
Note: access to the data is restricted by Synaspe.org and cannot be added to 
this repo.
You can access to the data at 
https://www.synapse.org/#!Synapse:syn5511449/tables
"""

import csv
import json
import os
import synapseclient

LIMIT = 100
OUT_DIR = 'data'



def load_json_columns(df, columns, column_names):
  """
  Load columns containing the JSON files.
  
  :param df: (pd.DataFrame) data frame with columns containing JSON file ids.
  :param columns: (synapse Column object) Contains the file paths.
  :param column_names: (list of str) name of the columns.
  :return: (dict) loaded JSON column data.
  """
  data = {}
  for column_name in column_names:
    data[column_name] = {}
    file_ids = df[column_name]
    for file_id in file_ids:
      file_path = columns.get(str(file_id))
      if file_path:
        with open(file_path) as f:
          data[column_name][file_id] = json.load(f)
  return data



def connect_to_synapse():
  """
  Connect to synapse.org. You need to set the right environment variables:
  - SYN_EMAIL: Your synapse.org email.
  - SYN_PWD: Your synapse.org password.
  
  :return: (Synapse) synapse connection instance
  """
  syn_email = os.getenv('SYN_EMAIL')
  syn_pwd = os.getenv('SYN_PWD')
  assert syn_email, 'The SYN_EMAIL environment variable is not set.'
  assert syn_pwd, 'The SYN_PWD environment variable is not set.'
  return synapseclient.login(syn_email, syn_pwd)



def filter_controls(survey_df, walking_df):
  """
  Filter data between controls and people with PD.
  
  Note from researcher:
    We use the field 'professional-diagnosis'to determine if a person has PD.
    'professional-diagnosis' == None or False are counted  as controls.

  :return controls_df, has_pd_df: (pd.DataFrame tuple) controls and PD df.
  """

  has_pd_survey_df = survey_df[survey_df['professional-diagnosis'] == True]
  controls_survey_df = survey_df[survey_df['professional-diagnosis'] != True]
  assert len(has_pd_survey_df) + len(controls_survey_df) == len(survey_df)

  # Join walking data frames to filter by controls and patients with PD.
  has_pd_df = walking_df.merge(has_pd_survey_df,
                               on='healthCode',
                               how='inner')
  controls_df = walking_df.merge(controls_survey_df,
                                 on='healthCode',
                                 how='inner')

  return controls_df, has_pd_df



def write_csv(outdir, base_name, data, headers):
  """
  Write data dict to CSV.
  
  :param outdir: (str) path to output file.
  :param base_name: (str) base name of the csv file.
  :param data: (dict) data to write to file.
  :param headers: (list of str) CSV headers
  """
  if not os.path.exists(outdir):
    os.mkdir(outdir)

  for column_name, all_data in data.items():
    outfile = os.path.join(outdir, base_name + column_name[:-11] + '.csv')
    with open(outfile, 'w+') as f:
      writer = csv.writer(f)
      writer.writerow(['id'] + headers)
      for id, data in all_data.items():
        if data:
          for record in data:
            row = [id]
            for h in headers:
              row.append(record[h])
            writer.writerow(row)



def main():
  # Connect
  syn = connect_to_synapse()

  # Demographics Survey table.
  survey_query = ('SELECT * FROM syn5511429 ORDER BY "healthCode" ASC '
                  'LIMIT %s' % LIMIT)
  survey_table = syn.tableQuery(survey_query)
  survey_df = survey_table.asDataFrame()

  # Walking Activity table.
  walking_query = ('SELECT * FROM syn5511449 ORDER BY "healthCode" ASC '
                   'LIMIT %s' % LIMIT)
  walking_table = syn.tableQuery(walking_query)
  walking_df = walking_table.asDataFrame()

  # Filter data between controls and people with PD.
  controls_df, has_pd_df = filter_controls(survey_df, walking_df)

  # Download column JSON data.
  column_names = ['accel_walking_outbound.json.items',
                  'accel_walking_return.json.items',
                  'accel_walking_rest.json.items']

  columns = syn.downloadTableColumns(walking_table, column_names)
  controls_data = load_json_columns(controls_df, columns, column_names)
  has_pd_data = load_json_columns(has_pd_df, columns, column_names)

  # Write data to file
  headers = ['timestamp', 'x', 'y', 'z']
  write_csv(OUT_DIR, 'controls_', controls_data, headers)
  write_csv(OUT_DIR, 'has_pd_', has_pd_data, headers)



if __name__ == '__main__':
  main()
