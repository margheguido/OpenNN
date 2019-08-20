#ifndef H_SUPGLOCALMATRIXFAST
#define H_SUPGLOCALMATRIXFAST

#include "local_matrix_fast.hpp"

// generic SUPG stabilized problem
class supg_local_matrix: public LocalMatrixFast<1>
{
protected:
    static const unsigned int numComps = 1;

    typedef typename LocalMatrixFast<1>::ValueVector ValueVector;
    typedef typename LocalMatrixFast<1>::ValueMatrix ValueMatrix;

    // parameters
    Real  beta[ 3 ], gamma, fff;
    double tau, mu;

public:

    void set_tau(double t)
    {
      tau=t;
    }

    void set_mu(double m)
    {
      mu=m;
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
