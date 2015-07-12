import struct
import numpy
import math
import sys

NUM_CLASSES = 10
MINIBATCH_SIZE = 50
REGULARIZATION_PENALTY = 0.01
INITIAL_LEARNING_RATE = 1.0
LEARNING_RATE_DECREASE_FACTOR = 2.0
CONVERGENCE_THRESHOLD = 0.2

def gradient_for_minibatch(parameter_matrix, minibatch):
  return numpy.add(
      probability_gradient_for_minibatch(parameter_matrix, minibatch),
      regularization_gradient(parameter_matrix)
      )

def probability_gradient_for_minibatch(parameter_matrix, minibatch):
  example_gradients = map(
      lambda(example): probability_gradient_for_example(parameter_matrix, example),
      minibatch
      )
  total = reduce(numpy.add, example_gradients)
  return numpy.dot(total, 1.0/len(minibatch))

def regularization_gradient(parameter_matrix):
  # We don't want to penalize the constant term for each parameter vector
  zeroer = numpy.identity(parameter_matrix.shape[1])
  zeroer[0,0] = 0
  zeroed_parameter_matrix = numpy.dot(parameter_matrix, zeroer)
  return numpy.dot(-2 * REGULARIZATION_PENALTY, zeroed_parameter_matrix)

def probability_gradient_for_example(parameter_matrix, example):
  datum = example[0]
  label = example[1]
  probabilities = map(lambda(lbl): probability_of_example(parameter_matrix, datum, lbl), range(NUM_CLASSES))
  scalars = map(lambda(lbl): [delta(lbl, label) - probabilities[lbl]], range(NUM_CLASSES))
  return numpy.dot(scalars, datum.reshape((1, len(datum))))

def probability_of_example(parameter_matrix, datum, lbl):
  dot_products = numpy.dot(parameter_matrix, datum)
  exponentiated_dot_products = map(lambda(x): math.exp(x), dot_products)
  return exponentiated_dot_products[lbl] / sum(exponentiated_dot_products)

def delta(x, y):
  return 1 if x == y else 0

def run(training_examples, testing_examples):
  global params_over_time
  parameter_matrix, params_over_time = train(training_examples)
  if len(testing_examples[0][0]) != len(training_examples[0][0]):
    raise IOError, "training and testing data dimensions do not match"
  incorrect_predictions = test(parameter_matrix, testing_examples)
  print "error rate (%):", 100 * float(len(incorrect_predictions)) / len(testing_examples)
  save_incorrect_predictions(incorrect_predictions)
  save_parameter_matrix(parameter_matrix)

def train(training_examples):
  parameter_matrix = numpy.zeros((NUM_CLASSES, len(training_examples[0][0])))
  params_over_time = []
  learning_rate = INITIAL_LEARNING_RATE
  epoch = 0
  while True:
    params_over_time.append(parameter_matrix)
    epoch += 1
    parameter_matrix = run_epoch(parameter_matrix, training_examples, learning_rate)
    norm = distance_between(params_over_time[-1], parameter_matrix)
    print "change from epoch %d: %f" % (epoch, norm)
    if norm < CONVERGENCE_THRESHOLD or epoch >= 10:
      break
    learning_rate = learning_rate / LEARNING_RATE_DECREASE_FACTOR
  return parameter_matrix, params_over_time

def run_epoch(parameter_matrix, training_examples, learning_rate):
  for minibatch in slice_array(training_examples, MINIBATCH_SIZE):
    gradient = gradient_for_minibatch(parameter_matrix, minibatch)
    parameter_matrix = numpy.add(parameter_matrix, numpy.dot(gradient, learning_rate))
  return parameter_matrix

def distance_between(a,b):
  difference = numpy.subtract(a, b)
  return numpy.linalg.norm(difference)

def slice_array(array, slice_length):
  counter = 0
  while counter < len(array):
    yield array[counter:counter+slice_length]
    counter += slice_length
    if counter % 1000 < slice_length:
      print ".",
      sys.stdout.flush()

def test(parameter_matrix, testing_examples):
  incorrect_predictions = []
  for index, example in enumerate(testing_examples):
    prediction = predict(parameter_matrix, example[0])
    if example[1] != prediction:
      incorrect_predictions.append([index, example[1], prediction])
  return incorrect_predictions

def predict(parameter_matrix, data):
  exponents = numpy.dot(parameter_matrix, data)
  return numpy.argmax(exponents)

def save_incorrect_predictions(incorrect_predictions):
  f = open("incorrect_predictions", "w")
  f.write(str(incorrect_predictions))
  f.close()

def training_examples():
  return zip_examples("train-images", "train-labels")

def testing_examples():
  return zip_examples("test-images", "test-labels")

def zip_examples(data_filename, label_filename):
  training_data = read_data_file(data_filename)
  training_labels = read_label_file(label_filename)
  if len(training_data) != len(training_labels):
    raise IOError, "number of data and number of labels do not match"
  return zip(training_data, training_labels)

def read_data_file(filename):
  f = open(filename, "rb")
  b1,b2,b3,b4 = struct.unpack("BBBB", f.read(4))
  if b1 != 0 or b2 != 0:
    raise IOError, "malformed IDX file"
  if b3 == 8:
    data_type = "B"
  else:
    raise IOError, "unexpected data type"
  if b4 != 3:
    raise IOError, "unexpected number of dimensions"
  input_dimensions = struct.unpack(">" + "I"*b4, f.read(4*b4))
  bytes_per_datum = input_dimensions[1]*input_dimensions[2]
  data = []
  for i in range(input_dimensions[0]):
    raw_datum = struct.unpack(data_type*bytes_per_datum, f.read(bytes_per_datum))
    transformed_datum = numpy.array([1] + map(lambda(x): x/256.0, raw_datum))
    data.append(transformed_datum)
  f.close()
  return data

def read_label_file(filename):
  f = open(filename, "rb")
  b1,b2,b3,b4 = struct.unpack("BBBB", f.read(4))
  if b1 != 0 or b2 != 0:
    raise IOError, "malformed IDX file"
  if b3 == 8:
    data_type = "B"
  else:
    raise IOError, "unexpected data type"
  if b4 != 1:
    raise IOError, "unexpected number of dimensions"
  label_dimensions = struct.unpack(">" + "I"*b4, f.read(4*b4))
  labels = struct.unpack(data_type*label_dimensions[0], f.read(label_dimensions[0]))
  f.close()
  return labels

def save_parameter_matrix(parameter_matrix):
  import idx
  idx.write("parameter_matrix", square_parameters(parameter_matrix))

def square_parameters(parameter_matrix):
  side_length = int(math.sqrt(len(parameter_matrix[0]) - 1))
  return map(lambda(v): vector_to_ubyte_square(v, side_length), parameter_matrix)

def vector_to_ubyte_square(parameter_vector, side_length):
  parameter_vector = parameter_vector[1:] # ignore the constant term
  factor = 255 / (parameter_vector.max() - parameter_vector.min())
  shifted_vector = numpy.add(parameter_vector, -parameter_vector.min())
  normalized_vector = numpy.dot(shifted_vector, factor).astype(int)
  return normalized_vector.reshape(side_length, side_length)

