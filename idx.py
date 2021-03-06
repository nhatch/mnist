import struct
import numpy

def read(filename):
  f = open(filename, "rb")
  b1,b2,b3,b4 = struct.unpack("BBBB", f.read(4))
  if b1 != 0 or b2 != 0:
    raise IOError, "malformed IDX file"
  if b3 != 8:
    raise IOError, "unexpected data type"
  input_dimensions = struct.unpack(">" + "I"*b4, f.read(4*b4))
  data = read_list(f, input_dimensions)
  f.close()
  return data

def read_list(f, input_dimensions):
  head, tail = input_dimensions[0], input_dimensions[1:]
  if len(tail) == 0:
    return list(struct.unpack("B"*head, f.read(head)))
  else:
    return map(lambda(_): read_list(f, tail), range(head))

# We're just going to trust that this data is all the same data
# type (ubyte) and is stored in a valid n-dimensional matrix.
def write(filename, data):
  dimensions = []
  current = data
  while hasattr(current, '__iter__'):
    dimensions.append(len(current))
    current = current[0]
  f = open(filename, "wb")
  f.write(bytearray([0,0,8,len(dimensions)]))
  f.write(struct.pack('>'+'I'*len(dimensions), *dimensions))
  write_list(f, data)
  f.close()

def write_list(f, data):
  if hasattr(data[0], '__iter__'):
    for subdata in data:
      write_list(f, subdata)
  else:
    f.write(struct.pack("B"*len(data), *data))

def crop_borders(data, border_size):
  crop_list = lambda(l): l[border_size:-border_size]
  crop_datum = lambda(datum): map(crop_list, crop_list(datum))
  return map(crop_datum, data)

def crop(filename, border_size):
  data = read(filename)
  cropped_data = crop_borders(data, border_size)
  write("cropped-" + filename, cropped_data)

def half(filename):
  data = read(filename)
  halved_data = map(half_size, data)
  write("halved-" + filename, halved_data)

def half_size(array):
  return map(half_list, half_array(array))

def half_array(array):
  new_array = []
  for i in range(len(array)/2):
    avg = avg_lists(array[2*i], array[2*i+1])
    new_array.append(avg)
  return new_array

def avg_lists(l1, l2):
  new_list = numpy.add(l1, l2)
  return map(lambda(x): x/2, new_list)

def half_list(l):
  new_list = []
  for i in range(len(l)/2):
    avg = (l[2*i] + l[2*i+1]) / 2
    new_list.append(avg)
  return new_list
