"""
Run an HTM agent in the Mario testbed.
1. Generate (i) sensory and (ii) motor sequences
  i. sequences of x values and sequences of y values
  ii. sequences of binary lists of length 6 -- e.g. [0,1,1,0,0,0]
2. Encode sequences
3. Mario runner object... train and test by calling tm.compute()
  note: need novel sequences for test phase, which returns stats
"""
import argparse
import csv
import numpy
import subprocess
import time


from sensorimotor.experiments.mario_encoders import MarioEncoders
from sensorimotor.experiments.mario_sensorimotor_runner import MarioSensorimotorRunner
from python_scripts import htm_client


#java_cmd = ['java', '-cp', 'bin/MarioAI/:lib/asm-all-3.3.jar:lib/jdom.jar',
#    'ch.idsia.scenarios.Main']
#python_cmd = ['python', 'python_scripts/htm_client.py']
#no_runs = 2  # first param in Main.sensorimotorTask.doEpisodes(2,true,1);
#subprocess.call(java_cmd, shell=False)



def generateSequences(no_runs):
  print "Generating N sensory and N motor sequences from Mario API."
  motor_sequences = []
  sensory_sequences = []
  for _ in xrange(no_runs):
    m, s = htm_client.runner()
    motor_sequences.append(m)
    sensory_sequences.append(s)
    time.sleep(0.1)
  print "=============================================="
  print "motor sequences: ", motor_sequences
  print "______________________________________________"
  print "sensory sequences (x and y): ", sensory_sequences
  print "______________________________________________"
  
  import pdb; pdb.set_trace()
  
  return sensory_sequences, motor_sequences


def encodeSequences(sensory_sequences, motor_sequences, write=False):
  print "Encoding raw sequences as SDRs."
#  nSensor, wSensor = 1024, 20
#  nMotor, wMotor = 1024, 20
  encoder = MarioEncoders(1024, 20)
  sensorySDRs = encoder.encodeXYPositionSequences(sensory_sequences)
  motorSDRs = encoder.encodeMotorSequences(motor_sequences)
  import pdb; pdb.set_trace()

  if write==True:
    with open("sensorySDRs.csv", "wb") as f:
      writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_ALL)
      for s in sensorySDRs:
        writer.writerow(s)
    print "sensory SDR sequences written to \'sensorySDRs.csv\'"
    with open("motorSDRs.csv", "wb") as f:
      writer = csv.writer(f, delimiter=",", quoting=csv.QUOTE_ALL)
      for m in motorSDRs:
        writer.writerow(m)
    print "motor SDR sequences written to \'motorSDRs.csv\'"
  return sensorySDRs, motorSDRs

def main(args):
  start = time.time()
  print "Setting up experiment..."
  s, m = generateSequences(args.no_runs)
  sSDR, mSDR = encodeSequences(s, m)
#  sSDR, mSDR = encodeSequences(generateSequences(1000))
  print "Running sensorimotor experiment..."
  import pdb; pdb.set_trace()
  mario = MarioSensorimotorRunner()
  # Train and test
  print "Training TM on sequences"
  mario.feed([s, m], verbosity=2, showProgressInterval=50)
  print "Testing TM on sequences"
  s, m = generateSequences(args.no_runs)
  sSDR, mSDR = encodeSequences(s, m)
#  sSDR, mSDR = encodeSequences(generateSequences(100))
  mario.feed([s, m], tmLearn=False, verbosity=2, showProgressInterval=50)


  elapsed = int(time.time() - start)
  print "Total time: {0:2} seconds".format(elapsed)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--no_runs",
                    help="Number of runs through level.",
                    default=100)
                    # no_runs = 2  # first param in Main.sensorimotorTask.doEpisodes(2,true,1);
  parser.add_argument("--save",
                    help="Save sequences to an output csv.",
                    default=True)
  args = parser.parse_args()
  
  if args.runs != 2: print "Warning: number of runs x must match parameter "\
    "passed to java server in Main.java at sensorimotorTask.doEpisodes(x,_,_)."

  main(args)
