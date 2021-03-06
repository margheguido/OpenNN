#ifndef __SUPGDATA2_H__
#define __SUPGDATA2_H__

// OpenNN includes
#include "SUPGdataBase.h"

// data for an advection diffusion reaction problem with known exact solution
class SUPGdata2: public SUPGdataBase
{
protected:
public:
    SUPGdata2(): SUPGdataBase() { }

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
        outValues[ 0 ] = 10*xx*yy*exp(4*yy)*(xx - 1)*(xx - 0.5)*(yy - 1) - local_mu*(20*xx*yy*exp(4*yy)*(xx - 1)*(xx - 0.5) + 20*xx*yy*exp(4*yy)*(yy - 1)*(yy - 0.75) + 20*xx*exp(4*yy)*(xx - 1)*(xx - 0.5)*(yy - 1) + 20*xx*exp(4*yy)*(xx - 1)*(xx - 0.5)*(yy - 0.75) + 20*yy*exp(4*yy)*(xx - 1)*(yy - 1)*(yy - 0.75) + 20*yy*exp(4*yy)*(xx - 0.5)*(yy - 1)*(yy - 0.75) + 80*xx*yy*exp(4*yy)*(xx - 1)*(xx - 0.5)*(yy - 1) + 80*xx*yy*exp(4*yy)*(xx - 1)*(xx - 0.5)*(yy - 0.75) + 80*xx*exp(4*yy)*(xx - 1)*(xx - 0.5)*(yy - 1)*(yy - 0.75) + 160*xx*yy*exp(4*yy)*(xx - 1)*(xx - 0.5)*(yy - 1)*(yy - 0.75)) + 10*xx*yy*exp(4*yy)*(xx - 1)*(xx - 0.5)*(yy - 0.75) + 10*xx*yy*exp(4*yy)*(xx - 1)*(yy - 1)*(yy - 0.75) + 10*xx*yy*exp(4*yy)*(xx - 0.5)*(yy - 1)*(yy - 0.75) + 10*xx*exp(4*yy)*(xx - 1)*(xx - 0.5)*(yy - 1)*(yy - 0.75) + 10*yy*exp(4*yy)*(xx - 1)*(xx - 0.5)*(yy - 1)*(yy - 0.75) + 40*xx*yy*exp(4*yy)*(xx - 1)*(xx - 0.5)*(yy - 1)*(yy - 0.75);
    }

    void sol_ex( Real *outValues, Real xx, Real yy, Real zz, Real tt ) const override
    {
        // outValues[ 0 ] =  sin(2.0 * mPi * xx) * (yy - square(yy));
    }

    void grad_sol_ex( Real *outValues, Real xx, Real yy, Real zz, Real tt ) const override
    {
        // outValues[ 0 ] = 2.0 * mPi * cos(2.0 * mPi * xx) * (yy - square(yy));
        // outValues[ 1 ] = sin(2.0 * mPi * xx) * (1.0 - 2.0 * yy);
        // outValues[ 2 ] = 0.0;
    }

    void lapl_sol_ex( Real *outValues, Real xx, Real yy, Real zz, Real tt ) const override
    {
        // const Real Pi2 = mPi * mPi;
        // outValues[ 0 ] = Pi2 * (cos(2.0 * mPi * xx) - cos(2.0 * mPi * (xx - yy)) + cos(2.0 * mPi * yy) - cos(2.0 * mPi * (xx + yy)));
    }
};

#endif //__SUPGDATA2_H__
