import struct
import numpy
import math
import sys

MINIBATCH_SIZE = 50
INITIAL_LEARNING_RATE = 1.0
LEARNING_RATE_DECREASE_FACTOR = 2.0
CONVERGENCE_THRESHOLD = 0.2

def use_algorithm(module):
  global alg
  alg = module

def run(training_examples, testing_examples):
  global params_over_time
  parameters, params_over_time = train(training_examples)
  if len(testing_examples[0][0]) != len(training_examples[0][0]):
    raise IOError, "training and testing data dimensions do not match"
  incorrect_predictions = test(parameters, testing_examples)
  print "error rate (%):", 100 * float(len(incorrect_predictions)) / len(testing_examples)
  save_incorrect_predictions(incorrect_predictions)
  alg.save_parameters(parameters)

def train(training_examples):
  parameters = alg.initial_parameters(len(training_examples[0][0]))
  params_over_time = []
  learning_rate = INITIAL_LEARNING_RATE
  epoch = 0
  while True:
    params_over_time.append(parameters)
    epoch += 1
    parameters = run_epoch(parameters, training_examples, learning_rate)
    norm = alg.distance_between(params_over_time[-1], parameters)
    print "change from epoch %d: %f" % (epoch, norm)
    if norm < CONVERGENCE_THRESHOLD or epoch >= 10:
      break
    learning_rate = learning_rate / LEARNING_RATE_DECREASE_FACTOR
  return parameters, params_over_time

def run_epoch(parameters, training_examples, learning_rate):
  for minibatch in slice_array(training_examples, MINIBATCH_SIZE):
    gradient = gradient_for_minibatch(parameters, minibatch)
    parameters = numpy.subtract(parameters, numpy.dot(gradient, learning_rate))
  return parameters

def gradient_for_minibatch(parameters, minibatch):
  return numpy.add(
      probability_gradient_for_minibatch(parameters, minibatch),
      alg.regularization_gradient(parameters)
      )

def probability_gradient_for_minibatch(parameters, minibatch):
  example_gradients = map(
      lambda(example): alg.probability_gradient_for_example(parameters, example),
      minibatch
      )
  total = reduce(numpy.add, example_gradients)
  return numpy.dot(total, 1.0/len(minibatch))

def slice_array(array, slice_length):
  counter = 0
  while counter < len(array):
    yield array[counter:counter+slice_length]
    counter += slice_length
    if counter % 1000 < slice_length:
      print ".",
      sys.stdout.flush()

def test(parameters, testing_examples):
  incorrect_predictions = []
  for index, example in enumerate(testing_examples):
    prediction = alg.predict(parameters, example[0])
    if example[1] != prediction:
      incorrect_predictions.append([index, example[1], prediction])
  return incorrect_predictions

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
    transformed_datum = numpy.array(map(lambda(x): x/256.0, raw_datum))
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

