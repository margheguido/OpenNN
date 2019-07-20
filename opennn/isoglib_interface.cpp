#include "isoglib_interface.h"

namespace OpenNN
{
  Matrix<double> IsoglibInterface::calculate_solution(tau)
  {
    call_isoglib(tau);
    return load_solution_binary();
  }


  void IsoglibInterface::set_solution_data()
  {
    // function to setup the problem
    auto setupProblem = [&]( Problem *problem ) {
        problem->getSolverParams().solverType = DIRECT;
    };


    g_meshFlags = FLAGS_DEFAULT | DO_NOT_USE_BASIS_CACHES;

    //qui chiamare la funzione che legge la mesh e crea il problema
  }


  void IsoglibInterface::call_isoglib(tau)
  {
    //creare la matrice con questo tau e risolvere
    // local matrix
    supg_local_matrix localMatrix;

  //  TestCase::solveSteadyAndComputeErrors( dir_name, ARRAY_SIZE( dir_name ), &localMatrix, &data, setupProblem );

    //creare due nuove funzioni in TestCase che si dividono ciò che fa solveSteadyAndComputeErrors
    //la prima crea il problema e la mesh (e andrebbe chiamata da set solution data)
    //la seconda chiamata da call_isoglib assembla la matrice e risolve
    //inoltre eliminiamo la parte in cui si calcolano gli errori e il salvataggio vtk
  }


  //reads from a binary file the pde solution and give it back in matrix form
  Matrix<double> IsoglibInterface::load_solution_binary()
  {
    ifstream file;
    file.open(solution_file_name.c_str(), ios::binary);

    if(!file.is_open())
    {
        ostringstream buffer;

        buffer << "OpenNN Exception: IsoglibInterface class.\n"
               << "void load_solution_binary() method.\n"
               << "Cannot open data file: " << data_file_name << "\n";

        throw logic_error(buffer.str());
    }

    streamsize size = sizeof(size_t);
    size = sizeof(double);
    double value;
    Matrix<double> solution_stab;
    solution_stab.set(nDof, 1); //vector of the solution

    for(size_t i = 0; i < nDof; i++)
    {
        file.read(reinterpret_cast<char*>(&value), size);

        solution_stab[i] = value;
    }

    file.close();
    return solution_stab;
  }


}
