import os
import socket
import subprocess

#from nupic.research import *


def getAction(i):
  return "this would be python agent action #%i\n" % i


def processDataIn(data_string):
  """
  Expected format is string of Mario's action and location.
  """
  data_strings = data_string.split("+")
  x = int(data_strings[0])
  y = int(data_strings[1])
  
  buttons = data_strings[2][1:-2].split(",")
  actions = []
#  import pdb; pdb.set_trace()
  for c in buttons:
    if "t" in c: actions.append(1)
    if "f" in c: actions.append(0)

  return actions, x, y

def runner(N):
  HOST = "localhost"
  PORT = 1123

  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
  sock.connect((HOST, PORT))

  sock.send("entering the python while loop...\n")
  motor_sequence = []
  sensory_sequence = []
  i = 1
  while 1<2:
    data_in = sock.recv(1024)
    print "step ", i
    print "raw data string received: ", data_in

    # Stoppage criteria
    if (data_in == "Kill\n"):
      # java server initializing the shutdown
      print "Stoppage criteria reached, shutting down socket..."
      sock.close()
      print "Socket closed"
      break
    elif i>1000:
      # python client (i.e. this) initializing the shutdown
      print "Stoppage criteria reached, shutting down socket..."
      sock.send("Shut it down")
      sock.close()
      print "Socket closed"
      break
    
  #  import pdb; pdb.set_trace()
    mario_action, mario_x, mario_y = processDataIn(data_in)
    print "data parsed into x, y, and action:"
    print mario_x
    print mario_y
    print mario_action
      # NOTE: this action is implemented at this position; i.e. gets agent to the next x,y
      # this data represents the environment preceding the action for this step
    motor_sequence.append(mario_action)
    sensory_sequence.append(mario_x)  # ignoring y locations for now
    data_out = getAction(i)
    sock.send(data_out)  # random for sensorimotor
    print "data sent: ", data_out
    i += 1

  return motor_sequence, sensory_sequence


if __name__ == "__main__":
  runner(N)
