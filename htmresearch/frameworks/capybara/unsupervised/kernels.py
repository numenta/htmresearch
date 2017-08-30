import math
import scipy



def kernel_normalize(k):
  return lambda x, y: k(x, y) / math.sqrt(k(x, x) + k(y, y))



def kernel_linear(x, y):
  return scipy.dot(x, y)



def kernel_poly(x, y, a=1.0, b=1.0, p=2.0):
  return (a * scipy.dot(x, y) + b) ** p



def kernel_gauss(x, y, sigma=0.00001):
  v = x - y
  l = math.sqrt(scipy.square(v).sum())
  return math.exp(-sigma * (l ** 2))



normalized_gaussian_kernel = kernel_normalize(kernel_gauss)

normalized_linear_kernel = kernel_normalize(kernel_linear)

normalized_poly_kernel = kernel_normalize(kernel_poly)
