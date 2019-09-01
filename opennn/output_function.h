#ifndef __OUTPUTFUNCTION_H__
#define __OUTPUTFUNCTION_H__

// System includes

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>

// Macro definitions (needed by isoglib)

#define USE_MPI

// OpenNN includes

#include "vector.h"
#include "matrix.h"
#include "isoglib_interface.h"
#include "numerical_differentiation.h"
#include "neural_network.h"
#include "perceptron_layer.h"
#include "multilayer_perceptron.h"
#include "normalized_squared_error.h"
#include "SUPGdataBase.h"


namespace OpenNN
{

/// This class contains tools to manipulate a single output of a neural networks
/// Using an external function, obtaining multiple outputs and compare them
/// with multiple targets, through the loss function
/// In particular, we will construct a neural networks that predicts
/// the SUPG stabilzation parameter and we will use this class to calculate
/// the pde solution (using IsoGlib library) and compare it with the exact one,
/// using numerical integration to compute a L2-norm.

class OutputFunction : public NormalizedSquaredError
{

public:

  // Default constructor
  OutputFunction() = delete;

  OutputFunction(string meshload_directory_name, SUPGdataBase* data_ptr)
  {
    isoglib_interface_pointer = new IsoglibInterface(meshload_directory_name, data_ptr);
  }

  // Destructor
  ~OutputFunction()
  {
    delete isoglib_interface_pointer;
  }

private:
  /// Calculates the derivative of the PDE solution wrt the stabilization parameter
  Matrix<double> calculate_PDE_solution_derivative(const Matrix<double>& tau_values, const Vector<size_t> & batch_indices) const;

  /// Calculates the solution of the PDE given the stabilization parameter predicted by the neural network
  Matrix<double> calculate_PDE_solution(const Matrix<double>& tau_values, const Vector<size_t> & batch_indices) const;

public:
  /// Calculates the derivative of the loss function wrt the stabilization parameter
  /// (does the same job as NormalizedSquaredError::calculate_output_gradient)
  Matrix<double> calculate_loss_derivative(const Matrix<double>& tau_values, const Vector<size_t> & batch_indices) const;

  IsoglibInterface * get_isoglib_interface_pointer() { return isoglib_interface_pointer; };

  void set_unscaled_inputs(const Matrix<double>& values) { unscaled_inputs = values; };

  // --------------------------------------------------------------------------
  // Overridden methods from NormalizedSquaredError
  // --------------------------------------------------------------------------

  // called only in epoch 0 directly in gradient descent
  double calculate_training_error() const override;
  // called from epoch 1 while calculating directional points
  double calculate_training_error(const Vector<double>& parameters) const override;

  double calculate_selection_error() const override;
  Vector<double> calculate_training_error_gradient() const override;

private:

  IsoglibInterface *isoglib_interface_pointer;

  // original inputs of the dataset (we can't use dimensionless parameters 
  // to solve the PDE)
  Matrix<double> unscaled_inputs;

  };

}

#endif // __OUTPUTFUNCTION_H__
