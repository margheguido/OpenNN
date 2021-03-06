#ifndef __SUPGDATA1_H__
#define __SUPGDATA1_H__

// OpenNN includes
#include "SUPGdataBase.h"

// data for an advection diffusion reaction problem with known exact solution
class SUPGdata1: public SUPGdataBase
{
protected:
public:
    SUPGdata1(): SUPGdataBase() { }

    int updateData( meshload_class &mesh ) override { m_isUpdated = true; return 0; }

    void diff_coeff( Real *outValues, Real xx, Real yy, Real zz, Real t ) const override
    {
        outValues[ 0 ] = current_diffusion_coefficient;
    }

    void beta_coeff( Real *outValues, Real xx, Real yy, Real zz, Real t ) const override
    {
        outValues[ 0 ] = 1.0;
        outValues[ 1 ] = 1.0;
        outValues[ 2 ] = 0.0;
    }

    void gamma_coeff( Real *outValues, Real xx, Real yy, Real zz, Real t ) const override
    {
        outValues[ 0 ] = 0.0;
    }

    void source_term( Real *outValues, Real xx, Real yy, Real zz, Real t ) const override
    {
        Real local_mu;
        diff_coeff( &local_mu, xx, yy, zz, t );
        outValues[ 0 ] = local_mu * sin(2.0 * mPi * xx) * (2.0 + 4.0 * mPi * mPi * (yy - square(yy))) + 2.0 * mPi * cos(2.0 * mPi * xx) * (yy - square(yy)) + sin(2.0 * mPi * xx) * (1.0 - 2.0 * yy);
    }

    void sol_ex( Real *outValues, Real xx, Real yy, Real zz, Real tt ) const override
    {
        outValues[ 0 ] =  sin(2.0 * mPi * xx) * (yy - square(yy));
    }

    void grad_sol_ex( Real *outValues, Real xx, Real yy, Real zz, Real tt ) const override
    {
        outValues[ 0 ] = 2.0 * mPi * cos(2.0 * mPi * xx) * (yy - square(yy));
        outValues[ 1 ] = sin(2.0 * mPi * xx) * (1.0 - 2.0 * yy);
        outValues[ 2 ] = 0.0;
    }

    void lapl_sol_ex( Real *outValues, Real xx, Real yy, Real zz, Real tt ) const override
    {
        // const Real Pi2 = mPi * mPi;
        // outValues[ 0 ] = Pi2 * (cos(2.0 * mPi * xx) - cos(2.0 * mPi * (xx - yy)) + cos(2.0 * mPi * yy) - cos(2.0 * mPi * (xx + yy)));
    }
};

#endif //__SUPGDATA1_H__
