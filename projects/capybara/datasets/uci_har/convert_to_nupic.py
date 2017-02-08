import csv
import os

cwd = os.getcwd()
metric = 'body_acc_x'

output_dir = os.path.join(cwd, os.pardir, os.pardir,
                          'classification', 'data', 'uci')

if not os.path.exists(output_dir):
  os.makedirs(output_dir)

# How many times a category is allowed to repeat
max_category_reps = {1: 20000, 2: 20000}  # or "None" if you want everything

for train_or_test in ['train', 'test']:
  input_file = 'inertial_signals_%s.csv' % train_or_test
  if max_category_reps:
    category_reps = {c: 0 for c in max_category_reps.keys()}
  else:
    category_reps = None

  output_file = os.path.join(output_dir, '%s_%s' % (metric, input_file))
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
        if max_category_reps:
          if category in max_category_reps:
            category_reps[category] += 1
            if category_reps[category] < max_category_reps[category]:
              writer.writerow([t, row[metric], category])
              t += 1
        else:
          writer.writerow([t, row[metric], category])
          t += 1

      print 'file saved to: %s' % output_file
      print 'number of rows: %s' % t
