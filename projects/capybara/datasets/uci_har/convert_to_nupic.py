import csv
import os

cwd = os.getcwd()
input_file = 'inertial_signals_train.csv'
output_dir = os.path.join(cwd, os.pardir, os.pardir, 
                           'classification', 'data', 'uci')

if not os.path.exists(output_dir):
  os.makedirs(output_dir)
  
# how many times a category is allowed to repeat
max_category_reps = {1:20000, 2:20000}
category_reps = {c:0 for c in max_category_reps.keys()}

output_file =  os.path.join(output_dir, 'body_acc_x.csv')
with open(input_file, 'r') as fr:
  reader = csv.reader(fr)
  headers = reader.next()
  with open(output_file, 'w+') as fw:
    writer = csv.writer(fw)
    writer.writerow(['x', 'y', 'label'])
    writer.writerow(['float', 'float', 'int'])
    writer.writerow([None, None, 'C'])
    t = 0
    for row in reader:
      row = dict(zip(headers, row))
      category = int(row['label']) 
      if category in max_category_reps:
        category_reps[category] += 1
        if category_reps[category] < max_category_reps[category]:
          writer.writerow([t, row['body_acc_x'], row['label']])
          t += 1
    print 'file saved to: %s' % output_file
    print 'number of rows: %s' % t