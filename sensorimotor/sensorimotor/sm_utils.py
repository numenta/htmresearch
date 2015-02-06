

# TODO: from temporal_pooler.py
def formatRow(x, formatString="%d", spacing=10, rowSize=100):
  """
  Utility routine for pretty printing large vectors
  """
  s = ""
  for c, v in enumerate(x):
    if c > 0 and c % spacing == 0:
      s += " "
    if c > 0 and c % rowSize == 0:
      s += "\n"
    s += formatString % v
  s += " "
  return s


# TODO: from sensorimotor_experiment_runner.py
#   def formatRow(self, x, formatString = "%d", rowSize = 700):
#     """
#     Utility routine for pretty printing large vectors
#     """
#     s = ''
#     for c,v in enumerate(x):
#       if c > 0 and c % 7 == 0:
#         s += ' '
#       if c > 0 and c % rowSize == 0:
#         s += '\n'
#       s += formatString % v
#     s += ' '
#     return s


# TODO: from sm_test_with_pooling.py parameteric changes with previous methods
# def formatRow(x, formatString="%d", rowSize=700):
#   """
#   Utility routine for pretty printing large vectors
#   """
#   s = ''
#   for c, v in enumerate(x):
#     if c > 0 and c % 7 == 0:
#       s += ' '
#     if c > 0 and c % rowSize == 0:
#       s += '\n'
#     s += formatString % v
#   s += ' '
#   return s


# TODO: from TM.py
# def formatRow(var, i):
#   s = ''
#   for c in range(self.numberOfCols):
#     if c > 0 and c % 10 == 0:
#       s += ' '
#     s += str(var[c, i])
#   s += ' '
#   return s

# TODO from TM_SM.py
# def formatRow(var, i):
#       s = ''
#       for c in range(self.numberOfCols):
#         if c > 0 and c % 10 == 0:
#           s += ' '
#         s += str(var[c,i])
#       s += ' '
#       return s
