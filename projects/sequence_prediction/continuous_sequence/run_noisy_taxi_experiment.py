import sys
import subprocess



# noiseList = [0, 0.02, 0.04, 0.06, 0.08, 0.1]
# noiseList = [0.04, 0.06, 0.08]
noiseList = [0.005, 0.01]
for noise in noiseList:
  process = subprocess.Popen(
    [sys.executable,
          "run_tm_model_sdr_classifier.py",
          "-d", "nyc_taxi",
          "--noise={:.2f}".format(noise)])
