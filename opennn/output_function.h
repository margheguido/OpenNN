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


namespace OpenNN
{

/// This class contains tools to manipulate a single output of a neural networks
/// Using an external function, obtaining multiple outputs and compare them
/// with multiple targets, through the loss function
/// In particular, we will construct a neural networks that predicts
/// the SUPG stabilzation parameter and we will use this class to calculate
/// the pde solution and compare it with the exact one

class OutputFunction : public NormalizedSquaredError
{

public:

  // Default constructor
  OutputFunction()
  {
    isoglib_interface_pointer = new IsoglibInterface();
  }

  // Destructor
  ~OutputFunction()
  {
    delete isoglib_interface_pointer;
  }

  //it calculates the derivative of the solution outputs wrt neural network one (tau)
  //#columns: number of instances
  //#rows: number of outputs
  Matrix<double> gradient_outputs(const Matrix<double>& single_output, const Matrix<double>& solution_outputs) const;


  //  this function calculate the solution outputs given the neural network one (tau)
  // #rows: number of instances (tipacally batch size)
  // #columns :number of outputs (nodes)
  Matrix<double> calculate_solution_outputs(const Matrix<double>& single_output) const;

  // //same function but for only one instance
  // //size of the vector: number of outputs (nodes)
  // Vector<double> calculate_solution_outputs(double tau) const;

  // void print_solution() const;
  void set_file_names( string sol_name);

  void set_nDof(size_t n);

  // --------------------------------------------------------------------------
  // Overridden methods from NormalizedSquaredError
  // --------------------------------------------------------------------------

  // OLD:
  // double calculate_training_error(const Vector<double>& parameters);
  // double calculate_batch_error(const Vector<size_t>& batch_indices);

  double calculate_training_error() const override;
  double calculate_selection_error() const override;
  Matrix<double> calculate_output_gradient(const Matrix<double>& outputs, const Matrix<double>& targets) const override;

private:

  IsoglibInterface *isoglib_interface_pointer;

  };

}

#endif // __OUTPUTFUNCTION_H__
