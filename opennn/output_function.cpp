#include "output_function.h"


namespace OpenNN
{

  OutputFunction::OutputFunction(const char * dir_name, string sol_name):
    isoglib_interface(dir_name,sol_name)
    {};

  Matrix<double> OutputFunction::gradient_outputs (const Matrix<double>& single_output,const Matrix<double>& solution_outputs) const
  {
    size_t nIstances = single_output.get_rows_number();
    size_t nOutputs = isoglib_interface.get_nDof()

   Matrix<double> grad_outputs(nOutputs,nIstances);
   //#columns: number of instances
   //#rows: number of outputs

   for (int i=0; i< nIstances; i++)
   {
     double out_value=single_output(i,0);
     double h=0.0001;
     Vector<double> y = calculate_solution_outputs(out_value);
     double x_forward = out_value + h;
     Vector<double> y_forward = calculate_solution_outputs(x_forward);
     Vector<double> d = (y_forward - y)/h;

     grad_outputs.set_column(i,d);
    }

    return grad_outputs;
 }



 Vector<double> OutputFunction::calculate_solution_outputs (double tau) const
  {
   Vector<double> temp_solution = isoglib_interface.calculate_solution(tau);

   return temp_solution;
  }

  //  this function calculate the solution outputs given the neural network one (tau)
  // #rows: number of instances (tipacally batch size)
  // #columns :number of outputs (nodes)
  Matrix<double> OutputFunction::calculate_solution_outputs (const Matrix<double>& single_output) const
  {
    size_t nIstances = single_output.get_rows_number();
    size_t nOutputs = isoglib_interface.get_nDof()
    Matrix<double> sol_outputs(nIstances, nOutputs);

    for (size_t i=0; i < nIstances; i++)
    {
      Real tau = single_output(i,0);
      Vector<double> temp_solution = isoglib_interface.calculate_solution(tau);
      sol_outputs.set_row(i,temp_solution);
    }

    return sol_outputs;
  }

  void OutputFunction::print_solution() const
  {
    solution_stab.print();
  }
}
