
#ifndef __ISOGLIBINTERFACE_H__
#define __ISOGLIBINTERFACE_H__

// System includes

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>

// OpenNN includes

#include "vector.h"
#include "matrix.h"
#include "Problem.hpp"
#include "TimeAdvancing.hpp"
#include "TestCase.hpp"
#include "local_matrix.hpp"
#include "SUPGLocalMatrix.hpp"
#include "epetra_communicator.hpp"
#include "GuidoVidulisTest.h"

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


  void set_file_names(char* dir_name , string sol_name);
  void set_problem_resolution();
  Vector <double> calculate_solution(double tau);
  size_t get_nDof() const { return nDof; };

private:
  //name of the test directories in isoglib
   char *directory_name;
  //Data_GuidoVidulisSUPGExactSol_p1_ref2

  //name of the binary file
  string solution_file_name;

  //number of points in which isoglib calculate the solution
  size_t nDof; //ref3 81, ref2 25

  // data
  GuidoVidulisTest data;

  Problem pde_prob;

  TimeAdvancing timeAdvancing;

  supg_local_matrix localMatrix;

  //fill pde_prob and timeAdvancing with the data from the mesh
  void setProblem( const char *dirName, LocalMatrixBase *localMatrix, data_class_interface *data, TestCase::ProblemFunc setupProblem );

  //assembly the local matrix and
  //it creates a binary file with the solution
  void solveSteady(double tau);

  //reads from a binary file the pde solution and give it back in matrix form
  Vector <double> load_solution_binary();


 };

 }

 #endif

 // OpenNN: Open Neural Networks Library.
