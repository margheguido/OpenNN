
#ifndef __OUTPUTFUNCTION_H__
#define __OUTPUTFUNCTION_H__

// System includes

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>

// OpenNN includes

#include "vector.h"
#include "matrix.h"



namespace OpenNN
{

/// This class contains tools to manipulate a single output of a neural networks
/// Using an external function, obtaining multiple outputs and compare them
/// with multiple targets, through the loss function
/// In particular, we will construct a neural networks that predicts
/// the SUPG stabilzation parameter and we will use this class to calculate
/// the pde solution and compare it with the exact one

class OutputFunction
{

public:

  // DEFAULT CONSTRUCTOR
  OutputFunction (const char * dir_name, string sol_name);

  //it calculates the derivative of the solution outputs wrt neural network one (tau)
  //#columns: number of instances
  //#rows: number of outputs
  Matrix<double> gradient_outputs (const Matrix<double>& single_output,const Matrix<double>& solution_outputs) const;


  //  this function calculate the solution outputs given the neural network one (tau)
  // #rows: number of instances (tipacally batch size)
  // #columns :number of outputs (nodes)
  Matrix<double> calculate_solution_outputs (const Matrix<double>& single_output) const

  //same function but for only one instance
  //size of the vector: number of outputs (nodes)
  Vector<double> calculate_solution_outputs (double tau) const;

  void print_solution() const;

private:

  IsoglibInterface isoglib_interface;

 };

 }

 #endif

 // OpenNN: Open Neural Networks Library.
