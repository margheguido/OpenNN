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
      double tau_for_EDP = tau * tau_scaling;
      solveSteady(tau_for_EDP, mu);

       std::cout << '\n' << '\n';
       std::cout << "tau: " << tau << '\n';
       std::cout << "tau_for_EDP: " << tau_for_EDP << '\n';

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


  void IsoglibInterface::compute_tau_scaling()
  {
    Real beta[3];
    Real x = 0.0;
    data_pointer->beta_coeff(beta,x,x,x,x);
    double norm_b = sqrt (square(beta[0]) + square(beta[1]) + square(beta[2]));
    tau_scaling = h/norm_b;
  }



}
