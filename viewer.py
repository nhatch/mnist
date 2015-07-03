#!/usr/bin/python
import io
import struct
import sys

dimensions = None
images = None

def initfile(filename):
  global dimensions, images
  f = open(filename, "rb")
  b1,b2,b3,b4 = struct.unpack(">BBBB", f.read(4))

  if b1 != 0 or b2 != 0:
    raise IOError, "cannot read non-IDX files"

  if b3 == 8:
    data_type = "B"
  else:
    raise IOError, "unsupported data type"

  if b4 != 3:
    raise IOError, "unsupported number of dimensions"

  dimensions = struct.unpack(">" + "I"*b4, f.read(4*b4))

  images = []
  for i in range(dimensions[0]):
    images.append(
      struct.unpack("B"*dimensions[1]*dimensions[2], f.read(dimensions[1]*dimensions[2]))
    )
  f.close()

initfile("train-images-idx3-ubyte")
print dimensions

def console_print(offset):
  image = images[offset]
  for i in range(dimensions[1]):
    for j in range(dimensions[2]):
      pixel = image[i*dimensions[2] + j]
      print "#" if pixel > 127 else " ",
    print

def csv_print(offset, f=sys.stdout):
  image = images[offset]
  for i in range(dimensions[1]):
    for j in range(dimensions[2]):
      pixel = image[i*dimensions[2] + j]
      pixel = 255 - pixel # invert
      f.write("%02x," % pixel)
    f.write("\b\n")

f = open('image.csv','w')
csv_print(int(sys.argv[1]), f)
f.close()
