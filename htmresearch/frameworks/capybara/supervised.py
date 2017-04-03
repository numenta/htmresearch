# ----------------------------------------------------------------------
# Numenta Platform for Intelligent Computing (NuPIC)
# Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
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

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def train_and_test(X_train, y_train, X_test, y_test, distance, n_neighbors):
  knn = KNeighborsClassifier(n_neighbors=n_neighbors,
                             algorithm='auto', metric=distance)

  knn.fit(X_train, y_train)

  y_pred = knn.predict(X_train)
  train_acc = '%.2f' % (accuracy_score(y_train, y_pred) * 100)

  y_pred = knn.predict(X_test)
  test_acc = '%.2f' % (accuracy_score(y_test, y_pred) * 100)

  return train_acc, test_acc
