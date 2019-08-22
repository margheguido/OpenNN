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


  Vector <double> IsoglibInterface::calculate_solution(double tau, double mu)
  {
    solveSteady(tau, mu);
    // solveSteady(0,0.0005);

    // Load solution from isoglib
    SolutionState * solution_state_pointer = timeAdvancing.getCurState();
    matr_Real * current_solution_pointer = solution_state_pointer->getSolArray(0);
    unsigned n_Dof = current_solution_pointer->getNumColumns();
    Vector<double> solution( n_Dof );
    for( unsigned i = 0; i < n_Dof; i++ )
    {
      solution[i] = current_solution_pointer->getData()[i];
    }

    std::cout << '\n' << '\n';
    std::cout << "tau: " << tau << '\n';
    std::cout << "solution:" << '\n';
    unsigned sqrt_n_Dof = 9;
    for( size_t i = sqrt_n_Dof-1; i >= 0 && i < sqrt_n_Dof; i-- )
    {
      for( unsigned j = 0; j < sqrt_n_Dof; j++ )
      {
        std::cout << std::setw(10) << solution[i*sqrt_n_Dof+j] << "  ";
      }
      std::cout << '\n';
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
  /*  timeAdvancing.setup( &pde_prob, 1, 0 );
    pde_prob.setTimeAdvancingScheme( &timeAdvancing );
       // call callback
    if ( setupProblem )
        setupProblem( &pde_prob );*/
  }


  void IsoglibInterface::solveSteady(double tau, double mu)
  {


    // set stabilization parameter
    localMatrix_pointer->set_tau(tau);
    localMatrix_pointer->set_mu(mu);
     pde_prob.setLocalMatrix(localMatrix_pointer);
     timeAdvancing.setup( &pde_prob, 1, 0 );
  pde_prob.setTimeAdvancingScheme( &timeAdvancing );

    pde_prob.computeTimestep(false);
  }

}
