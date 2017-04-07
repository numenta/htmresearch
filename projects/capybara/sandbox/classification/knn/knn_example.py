"""Helper plot functions"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, accuracy_score
import itertools



def plot_loss(loss):
  """
  Plot model loss per epoch.
  :param loss: (array) loss of the model 
  """
  plt.figure(figsize=(5, 5), dpi=100)
  plt.plot(range(len(loss)), loss)
  plt.title('Loss')
  plt.xlabel('Epoch')
  plt.ylabel('Loss')



def plot_confusion_matrix(y_true, y_pred, class_names=None,
                          compute_accuracy=True,
                          normalize=False,
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  :param y_true: (array) target classes
  :param y_pred: (array) predicted classes
  :param class_names: (list) pretty names for the classes. Integers from 0 to 
    len(y_true) will be used if set to None.
  :param compute_accuracy: (bool) whether to compute the accuracy and
    display it in the title.
  :param normalize: (bool) whether to normalize the confusion matrix
  :param cmap: (mpl.Colormap) color map
  :return cm: (np.array) confusion matrix
  """

  cm = confusion_matrix(y_true, y_pred)

  if compute_accuracy:
    accuracy = accuracy_score(y_true, y_pred)
    title = 'Confusion matrix (accuracy=%.2f)' % (accuracy * 100)
  else:
    title = 'Confusion matrix'
  plt.figure(figsize=(6, 6), dpi=100)
  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()

  if class_names is None:
    class_names = range(len(set(list(y_true))))
  tick_marks = np.arange(len(class_names))
  plt.xticks(tick_marks, class_names, rotation=45)
  plt.yticks(tick_marks, class_names)

  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')

  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, cm[i, j],
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.grid(False)
  return cm



def plot_decision_boundary(model, X, y, h=.01):
  """
  Plot the decision boundary. For that, we will assign a color to each
  point in the mesh [x_min, x_max]*[y_min, y_max].
  
  :param model: (Object) model supporting predict(X) where X is 2D. 
  :param X: (array) 2D input.
  :param y: (array) target classes for X.
  :param h: (float) step size in the mesh .
  """
  if type(y) != np.array:
    y = np.array(y)
  if type(X) != np.array:
    X = np.array(X)
  assert X.shape[1] == 2

  # Create color maps
  num_classes = len(set(list(y)))
  cmap_light = ListedColormap(sns.color_palette('colorblind', num_classes))
  cmap_bold = ListedColormap(sns.color_palette('colorblind', num_classes))

  x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
  y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                       np.arange(y_min, y_max, h))
  Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

  # Put the result into a color plot
  Z = Z.reshape(xx.shape)
  plt.figure()
  plt.contourf(xx, yy, Z, cmap=cmap_light)

  # Plot also the training points
  if y.ndim > 1:
    c = [cmap_bold(y[i:, 0]) for i in range(len(y[:, ]))]
  else:
    c = [cmap_bold(y[i]) for i in range(len(y))]

  plt.scatter(X[:, 0], X[:, 1], c=c, cmap=cmap_bold, edgecolor='black')
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())
  plt.title("Decision boundary")



def example():
  from sklearn import neighbors, datasets
  n_neighbors = 15

  # import some data to play with
  iris = datasets.load_iris()
  X = iris.data[:, :2]  # we only take the first two features. We could
  # avoid this ugly slicing by using a two-dim dataset
  y = iris.target
  for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)
    plot_decision_boundary(clf, X, y)

    y_pred = clf.predict(X)
    y_true = y
    cm = plot_confusion_matrix(y_true, y_pred)
    print cm

  plt.show()



if __name__ == '__main__':
  example()
