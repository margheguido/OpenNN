#ifndef __SUPGDATABASE_H__
#define __SUPGDATABASE_H__

// System includes
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <cmath>
#include <math.h>

// OpenNN includes
#include "vector.h"
#include "matrix.h"
#include "data.hpp"

// data for an advection diffusion reaction problem with known exact solution
class SUPGdataBase: public data_class_interface
{
protected:

    double current_diffusion_coefficient;

public:

    SUPGdataBase(): data_class_interface( 1 ) { }

    void set_diffusion_coefficient( double mu ) { current_diffusion_coefficient = mu; }

};

#endif //__SUPGDATABASE_H__
