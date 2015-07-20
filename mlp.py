import numpy
import math
import utils

NUM_CLASSES = 2
NUM_HIDDEN_UNITS = 10
UNITS_LAYERS = [NUM_HIDDEN_UNITS, NUM_CLASSES]
ACTIVATION_FUNCTION = math.tanh
DERIVATIVE_OF_ACTIVATION_FUNCTION = lambda(x): 1 - math.tanh(x)**2

def initial_parameters(data_dimension):
  # These sizes do not include the constant-1 output on each layer
  lower_layer_sizes = [data_dimension] + UNITS_LAYERS[:-1]
  upper_layer_sizes = UNITS_LAYERS
  dimension_pairs = zip(upper_layer_sizes, lower_layer_sizes)
  parameter_layers = map(lambda(x): _random_weights(*x), dimension_pairs)
  # For convenience, even though these layers have different dimensions,
  # we wrap them in a numpy.array. This lets us add, subtract, and multiply
  # by scalars without explicitly iterating through the array.
  return numpy.array(parameter_layers)

def _random_weights(num_upper_units, num_lower_units):
  return numpy.dot(0.02, numpy.random.rand(num_upper_units, num_lower_units+1))

def regularization_gradient(parameters):
  _verify_type(parameters)
  r = map(utils.regularization_gradient_ignoring_constant, parameters)
  return r

def probability_gradient_for_example(parameters, example):
  _verify_type(parameters)
  datum = example[0]
  label = example[1]
  unit_outputs, derivative_unit_outputs = _calculate_outputs(parameters, datum)
  output_delta_partials = unit_outputs[-1]
  output_delta_partials[label] -= 1
  delta_partials = _calculate_delta_partials(parameters, derivative_unit_outputs, output_delta_partials)
  res = _calculate_gradient(delta_partials, unit_outputs)
  return res

def _verify_type(parameters):
  if any(map(lambda(parameter_layer): type(parameter_layer) != numpy.ndarray, parameters)):
    raise StandardError

def _calculate_outputs(parameters, datum, calculate_derivatives=True):
  unit_outputs = []
  if calculate_derivatives:
    derivative_unit_outputs = []
  previous_layer_outputs = datum
  for index, parameter_layer in enumerate(parameters):
    previous_layer_outputs = numpy.insert(previous_layer_outputs, 0, 1)
    unit_outputs.append(previous_layer_outputs)
    weighted_sums = numpy.dot(parameter_layer, previous_layer_outputs)
    if index < len(parameters) - 1:
      previous_layer_outputs = map(ACTIVATION_FUNCTION, weighted_sums)
      if calculate_derivatives:
        derivatives = map(DERIVATIVE_OF_ACTIVATION_FUNCTION, weighted_sums)
        derivative_unit_outputs.append(derivatives)
    else:
      previous_layer_outputs = weighted_sums
  unit_outputs.append(previous_layer_outputs)
  if calculate_derivatives:
    return unit_outputs, derivative_unit_outputs
  else:
    return unit_outputs

def _calculate_delta_partials(parameters, derivative_unit_outputs, output_delta_partials):
  delta_partials = []
  previous_delta_partials_layer = output_delta_partials
  delta_partials.insert(0, previous_delta_partials_layer)
  params_with_upper_deltas = zip(parameters[1:], derivative_unit_outputs)
  for parameter_layer, derivative_layer in reversed(params_with_upper_deltas):
    sums = numpy.dot(previous_delta_partials_layer, parameter_layer)
    sums = sums[1:] # ignore the constant
    previous_delta_partials_layer = numpy.multiply(sums, derivative_layer)
    delta_partials.insert(0, previous_delta_partials_layer)
  return delta_partials

def _calculate_gradient(delta_partials, unit_outputs):
  layer_gradients = map(lambda(x): _gradient_for_layer(*x), zip(delta_partials, unit_outputs[:-1]))
  return numpy.array(layer_gradients)

def _gradient_for_layer(next_delta_partials, previous_outputs):
  po_reshape = previous_outputs.reshape(1, len(previous_outputs))
  ndp_reshape = next_delta_partials.reshape(len(next_delta_partials), 1)
  return numpy.dot(ndp_reshape, po_reshape)

def norm(parameter_type_array):
  norm_squareds = map(lambda(x): numpy.linalg.norm(x)**2, parameter_type_array)
  return math.sqrt(sum(norm_squareds))

def loss(parameters, examples):
  _verify_type(parameters)
  return sum(map(lambda(e): _loss_for_example(parameters, e), examples))

def _loss_for_example(parameters, example):
  datum = example[0]
  label = example[1]
  expected_outputs = numpy.zeros(NUM_CLASSES); expected_outputs[label] = 1
  actual_outputs = _calculate_outputs(parameters, datum, False)[-1]
  return 0.5 * numpy.linalg.norm(numpy.subtract(expected_outputs, actual_outputs))**2

def predict(parameters, datum):
  unit_outputs = _calculate_outputs(parameters, datum, False)
  return numpy.argmax(unit_outputs[-1])

def save_parameters(parameters):
  # TODO
  global mlp_parameters
  mlp_parameters = parameters

