import csv
from optparse import OptionParser
from os import listdir
from os.path import isfile, join
import sys



SIG_FIGS = 5



def main(dir):
  fileList = sorted([f for f in listdir(dir) if isfile(join(dir, f))])
  for f in fileList:
    with open(join(dir, f), "rU") as outputFile:
      csvReader = csv.reader(outputFile)
      next(csvReader)
      print "{1:.{0}f}".format(SIG_FIGS, float(next(csvReader)[0]))



def _getArgs():
  parser = OptionParser(usage="%prog OUTPUT_FILES_DIR "
                              "\n\nSpecify dir containing output files.")
  (options, args) = parser.parse_args(sys.argv[1:])
  if len(args) < 1:
    parser.print_help(sys.stderr)
    sys.exit()

  return args



if __name__ == "__main__":
  _args = _getArgs()
  main(_args[0])
