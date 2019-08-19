#include "isoglib_interface.h"

namespace OpenNN
{


  void IsoglibInterface::set_problem_resolution()
  {

    // function to setup the problem
    auto setupProblem = [&]( Problem *problem ) {
        problem->getSolverParams().solverType = DIRECT;
    };

    g_meshFlags = FLAGS_DEFAULT | DO_NOT_USE_BASIS_CACHES;

    setProblem( directory_name, localMatrix_pointer, &data, setupProblem );
  }


  Vector <double> IsoglibInterface::calculate_solution(double tau)
  {
    solveSteady(tau);

    // Load solution from isoglib
    SolutionState * solution_state_pointer = timeAdvancing.getCurState();
    matr_Real * current_solution_pointer = solution_state_pointer->getSolArray(0);
    unsigned n_Dof = current_solution_pointer->getNumColumns();
    Vector<double> solution( n_Dof );
    for( unsigned i = 0; i < n_Dof; i++ )
    {
      solution[i] = current_solution_pointer->getData()[i];
    }
    return solution;
  }


  void IsoglibInterface::setProblem(const char *dirNames, LocalMatrixBase *localMatrix, data_class_interface *data, TestCase::ProblemFunc setupProblem)
  {
    // start global clock
    g_timer.reset();

    // create communicator
    EpetraCommunicator::create();
    pout << "=============================================================\n";
    pout << "=============================================================\n";
    Communicator::instance2()->printInfo();

    // number of components
    const int numComps = data->getNumComponents();

    // problem

    // set data
    pde_prob.setData( data, false );

    // load mesh
    if ( pde_prob.loadMesh( directory_name, directory_name, new DofMapperBase( numComps ),
                                g_meshFlags, 0, g_numLagrangeMultipliers ) < 0 )
        exit( 1 );


    // set local matrix
    pde_prob.setLocalMatrix( localMatrix_pointer );
    // time advancing
    timeAdvancing.setup( &pde_prob, 1, 0 );
    pde_prob.setTimeAdvancingScheme( &timeAdvancing );
    /*    // call callback
    if ( setupProblem )
        setupProblem( &pde_prob );*/
  }


  void IsoglibInterface::solveSteady(double tau)
  {
    auto setupProblem = [&]( Problem *problem ) {
        problem->getSolverParams().solverType = DIRECT;
    };

    // set stabilization parameter
    localMatrix_pointer->set_tau(tau);
    // pde_prob.setLocalMatrix(localMatrix_pointer);
    // timeAdvancing.setup( &pde_prob, 1, 0 );
    // pde_prob.setTimeAdvancingScheme( &timeAdvancing );

        setupProblem( &pde_prob );
    pde_prob.computeTimestep(false);
  }


  void IsoglibInterface::set_file_names( string sol_name)
  {
    solution_file_name = sol_name;
  }


  // //reads from a binary file the pde solution and give it back in matrix form
  // Vector<double> IsoglibInterface::load_solution_binary()
  // {
  //   ifstream file;
  //   file.open(solution_file_name.c_str(), ios::binary);
  //
  //   if(!file.is_open())
  //   {
  //       ostringstream buffer;
  //
  //       buffer << "OpenNN Exception: IsoglibInterface class.\n"
  //              << "void load_solution_binary() method.\n"
  //              << "Cannot open data file: " << solution_file_name << "\n";
  //
  //       throw logic_error(buffer.str());
  //   }
  //
  //   streamsize size = sizeof(size_t);
  //   size = sizeof(double);
  //   double value;
  //   Vector<double> solution_stab;
  //   solution_stab.set(nDof); //vector of the solution
  //
  //   for(size_t i = 0; i < nDof; i++)
  //   {
  //       file.read(reinterpret_cast<char*>(&value), size);
  //
  //       solution_stab[i] = value;
  //   }
  //
  //   file.close();
  //   return solution_stab;
  // }


}
