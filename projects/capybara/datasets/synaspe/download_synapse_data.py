import synapseclient
import json
import os

LIMIT = 10
syn_email = os.getenv('SYN_EMAIL')  # Your synapse.org email
syn_pwd = os.getenv('SYN_PWD')  # Your synapse.org password
assert syn_email, 'The SYN_EMAIL environment variable is not set.'
assert syn_pwd, 'The SYN_PWD environment variable is not set.'

syn = synapseclient.login(syn_email, syn_pwd)

# Demographics survey table. Note from researcher:
# "We use the field 'professional-diagnosis' to determine if a person has PD. 
#  Blanks are counted  as controls."
demographics_query = ('SELECT * FROM syn5511429 ORDER BY "healthCode" ASC '
                      'LIMIT %s' % LIMIT)

demographics_table = syn.tableQuery(demographics_query)

demo_df = demographics_table.asDataFrame()

# Walking Activity table
walking_query = ('SELECT * FROM syn5511449 ORDER BY "healthCode" ASC '
                 'LIMIT %s' % LIMIT)
walking_table = syn.tableQuery(walking_query)
walking_df = walking_table.asDataFrame()

has_pd_df = demo_df[demo_df['professional-diagnosis'] != None]
controls_df = demo_df[demo_df['professional-diagnosis'] == None]
assert len(has_pd_df) + len(controls_df) == len(demo_df)

walking_has_pd_df = walking_df.merge(has_pd_df, on='healthCode', how='inner')
walking_controls_df = walking_df.merge(controls_df, on='healthCode', 
                                       how='inner')

column_names = ['accel_walking_outbound.json.items',
                'accel_walking_return.json.items',
                'accel_walking_rest.json.items']



columns = syn.downloadTableColumns(walking_table, column_names)
walking_data = {}
for columnId, file_path in columns.iteritems():
  with open(file_path) as f:
    walking_data[handle] = json.load(f)

# Walking activity table
# walking_query = "SELECT * FROM syn5511449"
# walking_columns = [t for t in syn.chunkedQuery(walking_query)]

# query the mPower project (syn4993293) for all the tables
query = 'SELECT id, name FROM table WHERE parentId=="syn4993293"'
tables = [t for t in syn.chunkedQuery(query)]

# Analyze the actual data
walking_table = [t for t in tables
                 if t['table.name'] == 'Walking Activity'][0]
dem_table = [t for t in tables
             if t['table.name'] == 'Demographics Survey'][0]

tables_to_analyze = [walking_table, dem_table]
data = {
  t['table.name']: syn.tableQuery('SELECT * FROM %s LIMIT 20' % t['table.id'])
  for t in tables_to_analyze}

# Extract the sample walking activity as a data frame
df = data['Walking Activity'].asDataFrame()

column_names = ['accel_walking_outbound.json.items',
                'accel_walking_return.json.items',
                'accel_walking_rest.json.items']
columns = syn.downloadTableColumns(data['Walking Activity'], column_names)

results = {handle: json.load(open(f)) for handle, f in columns.iteritems()}
