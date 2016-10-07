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
This file is used to debug specific Thing experiments.
"""

import pprint
from htmresearch.frameworks.layers.l2_l4_inference import L4L2Experiment

thingObjects = {
  "Capsule":[
    { 0: [[34,43,92,102,129,197,333,334,344,412,456,468,475,482,591,616,671,703,728,778,954],
          [836,837,838,839,840,841,842,843,844,845,846,847,848,849,850,851,852,853,854,855,856]]},
    { 0: [[9,56,62,102,276,334,407,412,433,447,468,475,496,616,655,703,778,799,907,914,993],
          [502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522]]},
    { 0: [[34,43,92,102,129,150,197,239,305,333,344,398,420,456,482,511,591,671,738,778,829],
          [502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522]]},
    { 0: [[43,107,130,287,291,367,451,468,482,509,510,591,615,648,716,838,876,917,923,947,954],
          [836,837,838,839,840,841,842,843,844,845,846,847,848,849,850,851,852,853,854,855,856]]},
    { 0: [[24,47,92,129,197,222,344,399,407,409,414,475,593,599,616,652,703,728,787,857,1019],
          [836,837,838,839,840,841,842,843,844,845,846,847,848,849,850,851,852,853,854,855,856]]},
    { 0: [[156,171,410,447,475,512,537,579,616,681,702,753,778,796,799,825,854,894,907,960,1021],
          [702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722]]},
    { 0: [[69,129,145,197,291,394,398,418,420,507,511,700,722,811,827,829,851,907,922,932],
          [702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722]]}
  ],
  "Sphere":[
    { 0: [[24,47,156,171,222,287,447,468,475,579,616,652,681,702,703,753,778,799,907,993,1019],
          [702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722]]},
    { 0: [[24,47,67,107,222,269,287,291,407,451,476,652,703,744,931,947,961,974,979,1019],
          [702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722]]},
    { 0: [[34,43,92,102,129,197,333,334,344,412,456,468,475,482,591,616,671,703,728,778,954],
          [702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722]]},
    { 0: [[24,47,92,129,197,222,344,399,407,409,414,475,593,599,616,652,703,728,787,857,1019],
          [702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722]]},
    { 0: [[43,107,130,287,291,367,451,468,482,509,510,591,615,648,716,838,876,917,923,947,954],
          [702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722]]}
  ],
  "Cube":[
    { 0: [[19,24,67,156,171,222,237,254,269,283,476,508,579,651,652,703,890,913,974,1007,1019],
          [33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53]]},
    { 0: [[67,110,245,287,292,342,378,447,476,510,512,579,591,675,681,702,753,787,861,961,979],
          [33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53]]},
    { 0: [[9,56,62,95,113,156,182,292,307,334,342,433,447,468,509,713,716,779,838,876,907],
          [33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53]]},
    { 0: [[102,233,276,308,334,377,409,412,433,475,593,612,616,635,703,768,778,799,833,857,914],
          [33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53]]},
    { 0: [[24,67,156,171,222,269,283,287,447,476,579,652,675,702,703,753,890,974,979,1007,1019],
          [167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187]]},
    { 0: [[9,56,62,102,276,334,407,412,433,447,468,475,496,616,655,703,778,799,907,914,993],
          [167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187]]},
    { 0: [[9,95,110,287,292,307,342,378,447,468,509,510,579,681,702,713,716,753,838,876,907],
          [167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187]]},
    { 0: [[19,24,47,156,169,171,222,254,409,475,579,593,616,635,651,652,703,778,799,857,1019],
          [167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187]]},
    { 0: [[24,47,156,171,222,287,447,468,475,579,616,652,681,702,703,753,778,799,907,993,1019],
          [836,837,838,839,840,841,842,843,844,845,846,847,848,849,850,851,852,853,854,855,856]]},
    { 0: [[19,24,47,156,169,171,222,254,409,475,579,593,616,635,651,652,703,778,799,857,1019],
          [502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522]]},
    { 0: [[9,95,110,287,292,307,342,378,447,468,509,510,579,681,702,713,716,753,838,876,907],
          [502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522]]},
    { 0: [[24,67,156,171,222,269,283,287,447,476,579,652,675,702,703,753,890,974,979,1007,1019],
          [502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522]]},
    { 0: [[9,56,62,102,276,334,407,412,433,447,468,475,496,616,655,703,778,799,907,914,993],
          [502,503,504,505,506,507,508,509,510,511,512,513,514,515,516,517,518,519,520,521,522]]}
  ]
}

def getObjectPair(objectName, pointNumber):
  """
  Returns the location/feature pair for point pointNumber in object objName

  """
  return thingObjects[objectName][pointNumber][0]


def runExperiment():
  """
  Runs a simple experiment where three objects share a number of location,
  feature pairs.

  Parameters:
  ----------------------------
  @param    noiseLevel (float)
            Noise level to add to the locations and features during inference

  @param    profile (bool)
            If True, the network will be profiled after learning and inference

  """
  exp = L4L2Experiment(
    "shared_features",
  )

  exp.learnObjects(thingObjects)

  L2Representations = exp.objectL2Representations
  print "Learned object representations:"
  pprint.pprint(L2Representations, width=400)
  print "=========================="

  print "\nRun inference with a point on the capsule"
  sensationList = [
    {0: getObjectPair("Capsule", 0)},
  ]
  exp.infer(sensationList, objectName="Capsule", reset= False)
  print "Output for capsule:", exp.getL2Representations()
  print "Intersection with sphere:", len(
    exp.getL2Representations()[0] & L2Representations["Sphere"][0])
  print "Intersection with capsule:", len(
    exp.getL2Representations()[0] & L2Representations["Capsule"][0])
  print "Intersection with cube:", len(
    exp.getL2Representations()[0] & L2Representations["Cube"][0])
  exp.sendReset()

  print "\nRun inference with a point on the sphere"
  sensationList = [
    {0: getObjectPair("Sphere", 0)},
  ]
  exp.infer(sensationList, objectName="Sphere", reset= False)
  print "Output for sphere:", exp.getL2Representations()
  print "Intersection with sphere:", len(
    exp.getL2Representations()[0] & L2Representations["Sphere"][0])
  print "Intersection with Capsule:", len(
    exp.getL2Representations()[0] & L2Representations["Capsule"][0])
  print "Intersection with cube:", len(
    exp.getL2Representations()[0] & L2Representations["Cube"][0])
  exp.sendReset()

  print "\nRun inference with two points on the sphere"
  sensationList = [
    {0: getObjectPair("Sphere", 0)},
    {0: getObjectPair("Sphere", 2)},
  ]
  exp.infer(sensationList, objectName="Sphere", reset= False)
  print "Output for sphere:", exp.getL2Representations()
  print "Intersection with sphere:", len(
    exp.getL2Representations()[0] & L2Representations["Sphere"][0])
  print "Intersection with Capsule:", len(
    exp.getL2Representations()[0] & L2Representations["Capsule"][0])
  print "Intersection with cube:", len(
    exp.getL2Representations()[0] & L2Representations["Cube"][0])
  exp.sendReset()




if __name__ == "__main__":
  runExperiment()