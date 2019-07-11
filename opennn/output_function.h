
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

   

   //it calculates the derivative of the multiple outputs wrt neural network one
   //#columns: number of outputs
   //#rows: number of instances
   Matrix<double> gradient_outputs (const Matrix<double>& single_output,const Matrix<double>& multiple_outputs) const;

   //  this function calculate multiple outputs given the neural network one
   // #rows: number of instances (tipacally batch size)
   // #columns :number of outputs
   Matrix<double> calculate_multiple_outputs (const Matrix<double>& single_output) const;

   //same function but for only one instance
   //size of the vector: number of outputs
   Vector<double> calculate_multiple_outputs (double single_output) const;



 private:

    // MEMBERS

    ///qui ci sarà l oggetto che chiamerà isoglib

 };

 }

 #endif

 // OpenNN: Open Neural Networks Library.
