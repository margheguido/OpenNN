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

    setProblem( data_pointer, setupProblem );
  }


  Vector <double> IsoglibInterface::calculate_solution(double tau, double mu)
  {
      double tau_for_EDP = tau * 0.009;
      solveSteady(tau_for_EDP, mu);

      // // Load solution from isoglib
      // vector<double> isoglib_solution = pde_prob.getSubProblem()->getSolution()->sol_r;
      // Vector<double> solution( nDof );
      // for( unsigned i = 0; i < nDof; i++ )
      // {
      //   solution[i] = isoglib_solution[i];
      // }
      //
       std::cout << '\n' << '\n';
       std::cout << "tau: " << tau << '\n';
       std::cout << "tau_for_EDP: " << tau_for_EDP << '\n';
      // // std::cout << "mu: " << mu << '\n';
      // std::cout << "solution:" << '\n';
      // unsigned sqrt_nDof = sqrt(nDof);
      // for( size_t i = sqrt_nDof-1; i >= 0 && i < sqrt_nDof; i-- )
      // {
      //   for( unsigned j = 0; j < sqrt_nDof; j++ )
      //   {
      //     std::cout << std::setw(10) << solution[i*sqrt_nDof+j] << "  ";
      //   }
      //   std::cout << '\n';
      // }

      // used to acces sol_r
      solution_class * solution_pointer = pde_prob.getSubProblem()->getSolution();

      // get elements of this process
      const int numMyElements = pde_prob.getSubProblem()->getMesh()->getNumMyElements();

      // solution
      Vector<double> solution_on_gauss_points( numMyElements * nGaussPoints, 0 );

      // compute value of the solution on gauss points for each element
      for ( int locE = 0; locE < numMyElements; locE++ )
      {
          // element
          const Element &element = pde_prob.getSubProblem()->getMesh()->getGeometryMapParam().getLocalElement( locE );
          const vect_int &funcs = element.functions;

          // for each Gauss points
          int numGauss = pde_prob.getSubProblem()->getMesh()->getGeometryMapParam().getNumGaussPoints( locE );
          for ( int locG = 0; locG < numGauss; ++locG )
          {
              // pointer to basis function values
              const GaussPoint *point;
              const ShapeValues *basisValues;
              pde_prob.getSubProblem()->getMesh()->getGeometryMap().getBasisValues( locE, locG, &point, nullptr, &basisValues, nullptr );

              // Gauss point
              const Real xx_gp = point->physCoords[ 0 ];
              const Real yy_gp = point->physCoords[ 1 ];
              const Real zz_gp = point->physCoords[ 2 ];

              const Real gWt = point->physWeight;

              const int lfuncs = (int) funcs.size();
              for ( int k = 0; k < lfuncs; k++ )
              {
                  const int dof = pde_prob.getSubProblem()->getDofManager()->getDofMapper().getDof2( funcs[ k ], 0 );
                  const Real sol = solution_pointer->sol_r[ dof ];
                  solution_on_gauss_points[ nGaussPoints * locE + locG ] += basisValues->R[ k ] * sol;
              }
          }  // end iterate gauss point
      } // end iterate elements

      return solution_on_gauss_points;
  }


  void IsoglibInterface::setProblem(data_class_interface *data, TestCase::ProblemFunc setupProblem)
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
      // setup
      localMatrix_pointer->set_tau(tau);
      data_pointer->set_diffusion_coefficient(mu);
      pde_prob.setLocalMatrix(localMatrix_pointer);
      timeAdvancing.setup( &pde_prob, 1, 0 );
      pde_prob.setTimeAdvancingScheme( &timeAdvancing );

      // solve
      pde_prob.computeTimestep(false);
  }







}
