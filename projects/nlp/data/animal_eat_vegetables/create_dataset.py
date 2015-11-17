# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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
Generate simple test datasets for sequence classification.

Each record in the dataset either has the format of
"animal eats vegetables" or "vegetables eats animals"

The goal is to classify the dataset into two groups
"""
import csv
import random
animals = []
vegetables = []

animal_reader = csv.reader(open('animals.txt', 'r'))
for row in animal_reader:
  animals.append(row[0])

vegetable_reader = csv.reader(open('vegetables.txt', 'r'))
for row in vegetable_reader:
  vegetables.append(row[0])

animals = animals[:5]
vegetables = vegetables[:5]

nSample = 100
sentences = []
classification_label_list = ['animalEatsVegetable', 'vegetableEatsAnimal']
classification_labels = []
for ID in xrange(nSample):
  animal = animals[random.randint(0, len(animals)-1)]
  vegetable = vegetables[random.randint(0, len(vegetables)-1)]

  sentence = animal + ' eats ' + vegetable
  classification = 0

  sentences.append(sentence)
  classification_labels.append(classification)
  print str(ID), ': ', classification_labels[-1], sentences[-1]

  sentence = vegetable + ' eats ' + animal
  classification = 1

  sentences.append(sentence)
  classification_labels.append(classification)
  print str(ID), ': ', classification_labels[-1], sentences[-1]


data_writer = csv.writer(open('animal_eat_vegetable.csv', 'w'))
data_writer.writerow(['ID', 'Question', 'Sample', 'Classification'])
for ID in xrange(nSample):
  data_writer.writerow([ID, '', sentences[ID], classification_label_list[classification_labels[ID]]])


data_writer = csv.writer(open('animal_eat_vegetable_network.csv', 'w'))
data_writer.writerow(['_token', '_category', '_sequenceID', '_reset', 'ID'])
data_writer.writerow(['string', 'list', 'int', 'int', 'string'])
data_writer.writerow(['', 'C', 'S', 'R', ''])
for ID in xrange(nSample):
  sentence = sentences[ID]
  classification = classification_labels[ID]
  words = sentence.split(' ')
  for i in xrange(len(words)):
    if i == 0:
      reset = 1
    else:
      reset = 0
    data_writer.writerow([words[i], classification_labels[ID], ID, reset, ID])


