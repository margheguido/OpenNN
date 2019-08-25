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
    double tau_for_EDP = tau * 0.009;
    solveSteady(tau_for_EDP, mu);
    // solveSteady(0.005,0.0005);

    // Load solution from isoglib
    vector<double> isoglib_solution = pde_prob.getSubProblem()->getSolution()->sol_r;
    Vector<double> solution( nDof );
    for( unsigned i = 0; i < nDof; i++ )
    {
      solution[i] = isoglib_solution[i];
    }

    std::cout << '\n' << '\n';
    std::cout << "tau: " << tau << '\t' << "tau_for_EDP: " << tau_for_EDP << '\n';
    std::cout << "solution:" << '\n';
    unsigned sqrt_nDof = sqrt(nDof);
    for( size_t i = sqrt_nDof-1; i >= 0 && i < sqrt_nDof; i-- )
    {
      for( unsigned j = 0; j < sqrt_nDof; j++ )
      {
        std::cout << std::setw(10) << solution[i*sqrt_nDof+j] << "  ";
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
