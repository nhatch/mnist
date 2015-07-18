import numpy
import math
import utils

NUM_CLASSES = 2
NUM_HIDDEN_UNITS = 2
UNITS_LAYERS = [NUM_HIDDEN_UNITS, NUM_CLASSES]
ACTIVATION_FUNCTION = math.tanh
DERIVATIVE_OF_ACTIVATION_FUNCTION = lambda(x): 1 - math.tanh(x)**2

def initial_parameters(data_dimension):
  # These sizes do not include the constant-1 output on each layer
  lower_layer_sizes = [data_dimension] + UNITS_LAYERS[:-1]
  upper_layer_sizes = UNITS_LAYERS
  dimension_pairs = zip(upper_layer_sizes, lower_layer_sizes)
  return map(lambda(upper, lower): numpy.zeros((upper, lower+1)), dimension_pairs)

def regularization_gradient(parameters):
  return map(utils.regularization_gradient_ignoring_constant, parameters)

def probability_gradient_for_example(parameters, example):
  datum = example[0]
  label = example[1]
  unit_outputs, derivative_unit_outputs = _calculate_outputs(parameters, datum)
  output_delta_partials = unit_outputs[-1]
  output_delta_partials[label] -= 1
  delta_partials = _calculate_delta_partials(parameters, derivative_unit_outputs, output_delta_partials)
  return _calculate_gradient(delta_partials, unit_outputs)

def _calculate_outputs(parameters, datum, calculate_derivatives=True):
  unit_outputs = []
  if calculate_derivatives:
    derivative_unit_outputs = []
  previous_layer_outputs = datum
  for parameter_layer in parameters:
    previous_layer_outputs = numpy.insert(previous_layer_outputs, 0, 1)
    unit_outputs.append(previous_layer_outputs)
    weighted_sums = numpy.dot(parameter_layer, previous_layer_outputs)
    if parameter_layer is not parameters[-1]:
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
  return map(lambda(x): _gradient_for_layer(*x), zip(delta_partials, unit_outputs[:-1]))

def _gradient_for_layer(next_delta_partials, previous_outputs):
  po_reshape = previous_outputs.reshape(1, len(previous_outputs))
  ndp_reshape = next_delta_partials.reshape(len(next_delta_partials), 1)
  return numpy.dot(ndp_reshape, po_reshape)

def predict(parameters, datum):
  unit_outputs = _calculate_outputs(parameters, datum, False)
  return numpy.argmax(unit_outputs[-1])

def distance_between(a, b):
  difference = numpy.subtract(a, b)
  return sum(map(numpy.linalg.norm, difference))

def save_parameters(parameters):
  # TODO
  global mlp_parameters
  mlp_parameters = parameters

