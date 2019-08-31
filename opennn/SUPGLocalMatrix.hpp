#ifndef H_SUPGLOCALMATRIXFAST
#define H_SUPGLOCALMATRIXFAST

#include "local_matrix_fast.hpp"
#include <cmath>
#include <math.h>
// generic SUPG stabilized problem
class supg_local_matrix: public LocalMatrixFast<1>
{
protected:
    static const unsigned int numComps = 1;

    typedef typename LocalMatrixFast<1>::ValueVector ValueVector;
    typedef typename LocalMatrixFast<1>::ValueMatrix ValueMatrix;

    // parameters
    Real  mu, beta[ 3 ], gamma, fff;
    double tau;

public:

    void set_tau(double t)
    {
        tau = t;
    }

    supg_local_matrix():
        LocalMatrixFast<1>()
    {
        beta[ 2 ] = 0.0;
    }

    virtual void evaluate_parameters_on_gauss_point( Real x, Real y, Real z, const ShapeValues &basisValues ) override
    {
        const data_class_interface &data = this->m_solution->getData();
        data.diff_coeff( &mu, x, y, z, 0.0 );
        data.beta_coeff( beta, x, y, z, 0.0 );
        data.gamma_coeff( &gamma, x, y, z, 0.0 );
        data.source_term( &fff, x, y, z, 0.0 );
    }

    // void source_term_mu( Real *outValues, Real xx, Real yy, Real zz, Real t,double mu) const
    // {
  //   /*  outValues[ 0 ] = mu*((2*square(2*xx - 1)*(square(xx - 1/2) + square(yy - 1/2) - 1/16))/
  //     (mu*sqrt(mu)*square(square(square(xx - 1/2)+ square(yy - 1/2)- 1/16)/mu + 1))
  //   - 4/(sqrt(mu)*(square(square(xx - 1/2) + square(yy - 1/2)- 1/16)/mu + 1)) +
  //   (2*square(2*yy- 1)*(square(xx - 1/2) +square(yy - 1/2) - 1/16)) /
  //   (mu*sqrt(mu)*square(square(square(xx - 1/2) +
  // square(yy - 1/2) - 1/16)/mu + 1))) +
  // (-(2*xx - 1)/(sqrt(mu)*(square(square(xx - 1/2) +
  //  square(yy - 1/2) - 1/16)/mu + 1)))
  //  + (-(2*yy - 1)/(sqrt(mu)*(square(square(xx - 1/2) +
  //   square(yy - 1/2) - 1/16)/mu + 1)));*/
  //   outValues[ 0 ] = exp(-yy) * ( -mu *( -square(xx)+  xx -2) + square(xx) - 3*xx +1);
  //      // outValues[ 0 ] = mu* sin(2.0 * mPi * xx) * (2.0 + 4.0 * mPi * mPi * (yy - square(yy))) + 2.0 * mPi * cos(2.0 * mPi * xx) * (yy - square(yy)) + sin(2.0 * mPi * xx) * (1.0 - 2.0 * yy);
  //   }

    virtual void integrate_on_gauss_point( ValueMatrix &outValues, Real x, Real y, Real z, int k, int m, const ShapeValues &basisValues ) const override
    {
      //  Real tau = 0.5 * 0.25 * 1.4; // delta * h_K / norm(b)

        Real ret = ( mu * ( basisValues.Rgrad[ 0 ][ k ] * basisValues.Rgrad[ 0 ][ m ]
                            + basisValues.Rgrad[ 1 ][ k ] * basisValues.Rgrad[ 1 ][ m ]
                            + basisValues.Rgrad[ 2 ][ k ] * basisValues.Rgrad[ 2 ][ m ] ) // on surface
                       + basisValues.R[ k ] * ( beta[ 0 ] * basisValues.Rgrad[ 0 ][ m ]
                                              + beta[ 1 ] * basisValues.Rgrad[ 1 ][ m ]
                                              + beta[ 2 ] * basisValues.Rgrad[ 2 ][ m ] ) // on surface
                       + gamma * basisValues.R[ k ] * basisValues.R[ m ] )
                       + tau * ( beta[ 0 ] * basisValues.Rgrad[ 0 ][ k ] // SUPG
                               + beta[ 1 ] * basisValues.Rgrad[ 1 ][ k ]
                               + beta[ 2 ] * basisValues.Rgrad[ 2 ][ k ] )
                             * ( - mu * basisValues.Rlapl[ m ]
                                 + beta[ 0 ] * basisValues.Rgrad[ 0 ][ m ]
                                 + beta[ 1 ] * basisValues.Rgrad[ 1 ][ m ]
                                 + beta[ 2 ] * basisValues.Rgrad[ 2 ][ m ] );
        outValues[ 0 ][ 0 ] = ret;
    }

    virtual void rhs_on_gauss_point( ValueVector &outValues, Real x, Real y, Real z, int k, const ShapeValues &basisValues ) const override
    {
        //Real tau = 0.5 * 0.25 * 1.4;
        outValues[ 0 ] = fff * ( basisValues.R[ k ]
                               + tau * ( beta[ 0 ] * basisValues.Rgrad[ 0 ][ k ] // SUPG
                                       + beta[ 1 ] * basisValues.Rgrad[ 1 ][ k ]
                                       + beta[ 2 ] * basisValues.Rgrad[ 2 ][ k ] ) );
    }
};

#endif // H_SUPGLOCALMATRIXFAST
