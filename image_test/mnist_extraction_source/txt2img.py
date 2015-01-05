import numpy as np
from PIL import Image

# utility to convert "txt" files extracted from MNIST to image (.png) format
a=np.genfromtxt("a.txt", skip_header=1)
im = Image.fromarray(a, mode="L")
im.save("a.jpg")
