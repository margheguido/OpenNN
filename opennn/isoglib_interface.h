#ifndef __ISOGLIBINTERFACE_H__
#define __ISOGLIBINTERFACE_H__

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
#include "Problem.hpp"
#include "TimeAdvancing.hpp"
#include "DefSolutionState.hpp"
#include "TestCase.hpp"
#include "local_matrix.hpp"
#include "SUPGLocalMatrix.hpp"
#include "epetra_communicator.hpp"
#include "data.hpp"

namespace OpenNN
{
  /* This class represents the interface between the libraries Isoglib and OpenNN.
  The main member of this class take as input the stabilization parameter TAU
  computed by the Neural Network and calculate the corresponding Pde Solution
  using Isoglib code (specialized by us in SUPG stabilzation).
  It creates the solution as a binary file in the data folder of openNN
  Another member can read it and give back the solution in matrix form, that can
  be used by the class output_function to calculate loss function or gradient
  of our customized function */


class IsoglibInterface
{

public:

  // Default constructor (TODO: remove)
  IsoglibInterface() = delete;
  // {
  //   localMatrix_pointer = new supg_local_matrix;
  //   set_problem_resolution();
  // }

  IsoglibInterface(string meshload_directory_name, data_class_interface* data_ptr):
  data_pointer(data_ptr)
  {
    localMatrix_pointer = new supg_local_matrix;
    directory_name = meshload_directory_name.c_str();
    set_problem_resolution();
  }

  ~IsoglibInterface()
  {
    delete localMatrix_pointer;
  }

  void set_problem_resolution();

  Vector<double> calculate_solution(double tau, double mu);

  unsigned get_nDof() const { return nDof; };
  unsigned get_nElems() const { return nElems; };
  unsigned get_nGaussPoints() const { return nGaussPoints; };

  void set_nDof(unsigned n){ nDof = n; };
  void set_nElems(unsigned n){ nElems = n; };
  void set_nGaussPoints(unsigned n){ nGaussPoints = n; };

private:

  //name of the test directories in isoglib
   const char *directory_name;

  //number of points in which isoglib calculate the solution
  unsigned nDof; // ref3 -> 81

  //number of elements
  unsigned nElems; // ref3 -> 64

  //number of gauss points per element
  unsigned nGaussPoints; // 4 by default in isoglib, set from main

  // data
  data_class_interface* data_pointer;

  Problem pde_prob;

  TimeAdvancing timeAdvancing;

  supg_local_matrix * localMatrix_pointer;

  //fill pde_prob and timeAdvancing with the data from the mesh
  void setProblem( data_class_interface *data, TestCase::ProblemFunc setupProblem );

  //assembly the local matrix and
  //it creates a binary file with the solution
  void solveSteady(double tau,double mu);


  };

 }

 #endif // __ISOGLIBINTERFACE_H__
