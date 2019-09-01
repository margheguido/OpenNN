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
#include "SUPGdataBase.h"

namespace OpenNN
{
  /* This class represents the interface between the libraries Isoglib and OpenNN.
  The main member of this class take as input the stabilization parameter TAU
  computed by the Neural Network and calculate the corresponding Pde Solution
  using Isoglib code (specialized by us in SUPG stabilzation).
  The user obtain the solution computed in the guass nodes, in matrix form, that can
  be used by the class output_function to calculate loss function or
  its gradient */


class IsoglibInterface
{

public:

  //Constructors and destructor
  IsoglibInterface() = delete;
  IsoglibInterface(string meshload_directory_name, SUPGdataBase* data_ptr);
  ~IsoglibInterface()
  {
    delete localMatrix_pointer;
  };

  void set_problem_resolution();
  Vector<double> calculate_solution(double tau, double mu);

  unsigned get_nDof() const { return nDof; };
  unsigned get_nElems() const { return nElems; };
  unsigned get_nGaussPoints() const { return nGaussPoints; };
  double get_tau_scaling() const { return tau_scaling; };
  void set_nDof(unsigned n)
  {
    nDof = n;
    h = 1/ (sqrt(nDof) - 1);
    compute_tau_scaling();
  };
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

  //scaling value for the stabilization parameter computed by the ANN
  double tau_scaling;

  //mesh refinement
  double h;

  // data
  SUPGdataBase* data_pointer;

  Problem pde_prob;

  TimeAdvancing timeAdvancing;

  supg_local_matrix * localMatrix_pointer;

  //fill pde_prob and timeAdvancing with the data from the mesh
  void setProblem( data_class_interface *data, TestCase::ProblemFunc setupProblem );

  //assembly the local matrix and solve the pde problem, saving the solution
  //in pde_prob
  void solveSteady(double tau,double mu);

  //compute the scaling coefficent
  void compute_tau_scaling();
  };

 }

 #endif // __ISOGLIBINTERFACE_H__
