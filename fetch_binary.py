import sys
import os
import requests
from boto.s3.connection import S3Connection

REGION = "us-west-2"
BUCKET = "artifacts.numenta.org"
REPO = "numenta/nupic"
SHA_FILE = "nupic_sha.txt"
LOCAL_BINARY_ARCHIVE = "nupic-archive.tar.gz"
AWS_KEY = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET = os.environ["AWS_SECRET_ACCESS_KEY"]

def fetchNupicEggFor(sha):
  conn = S3Connection(AWS_KEY, AWS_SECRET)
  artifactsBucket = conn.get_bucket(BUCKET)
  s3Url = None

  for key in artifactsBucket.list(prefix="%s/%s" % (REPO, sha)):
    if key.name[-2:] == "gz":
      s3Url = "https://s3-%s.amazonaws.com/%s/%s" % (REGION, BUCKET, key.name)

  if s3Url is None:
    raise ValueError("Couldn't find binary archive!")

  print "Fetching archive from %s..." % s3Url
  blockCount = 0
  with open(LOCAL_BINARY_ARCHIVE, "wb") as handle:
    response = requests.get(s3Url, stream=True)
    if not response.ok:
      raise Exception("Cannot fetch NuPIC egg from %s" % s3Url)
    for block in response.iter_content(1024):
      if blockCount % 100 == 0:
        sys.stdout.write('.')
        sys.stdout.flush()
      blockCount += 1
      if not block:
        break
      handle.write(block)
    print "\nDone."



def getSha():
  with open(SHA_FILE, "r") as shaFile:
    return shaFile.read().strip()



if __name__ == "__main__":
  sha = getSha()
  fetchNupicEggFor(sha)
