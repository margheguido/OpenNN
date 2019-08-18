#include "output_function.h"


namespace OpenNN
{

  Matrix<double> OutputFunction::gradient_outputs(const Matrix<double>& single_output, const Matrix<double>& solution_outputs)
  {
    size_t nIstances = single_output.get_rows_number();
    size_t nOutputs = isoglib_interface_pointer->get_nDof();

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



 Vector<double> OutputFunction::calculate_solution_outputs(double tau) const
  {
   Vector<double> temp_solution = isoglib_interface_pointer->calculate_solution(tau);

   return temp_solution;
  }

  //  this function calculate the solution outputs given the neural network one (tau)
  // #rows: number of instances (tipacally batch size)
  // #columns :number of outputs (nodes)
  Matrix<double> OutputFunction::calculate_solution_outputs(const Matrix<double>& single_output) const
  {
    size_t nIstances = single_output.get_rows_number();
    size_t nOutputs = isoglib_interface_pointer->get_nDof();
    Matrix<double> sol_outputs(nIstances, nOutputs);

    for (size_t i=0; i < nIstances; i++)
    {
      Real tau = single_output(i,0);
      Vector<double> temp_solution = isoglib_interface_pointer->calculate_solution(tau);
      sol_outputs.set_row(i,temp_solution);
    }

    return sol_outputs;
  }

  void OutputFunction::set_file_names(const char *dir_name, string sol_name)
  {
    isoglib_interface_pointer->set_file_names(dir_name, sol_name);
  }

  // void OutputFunction::print_solution() const
  // {
  //   solution_stab.print();
  // }


// --------------------------------------------------------------------------
// Overridden methods from NormalizedquaredError
// --------------------------------------------------------------------------

  double OutputFunction::calculate_training_error() const
  {
  #ifdef __OPENNN_DEBUG__

  check();

  #endif

      // Multilayer perceptron

      const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

      // Data set

      const Vector< Vector<size_t> > training_batches = data_set_pointer->get_instances_pointer()->get_training_batches(batch_size);

      const size_t batches_number = training_batches.size();

      double training_error = 0.0;

      #pragma omp parallel for reduction(+ : training_error)

      for(int i = 0; i < static_cast<int>(batches_number); i++)
      {
          const Matrix<double> inputs = data_set_pointer->get_inputs(training_batches[static_cast<unsigned>(i)]);
          const Matrix<double> targets = data_set_pointer->get_targets(training_batches[static_cast<unsigned>(i)]);

          const Matrix<double> stabilization_parameters = multilayer_perceptron_pointer->calculate_outputs(inputs);

          // IMPORTANT: here the solution of the PDE is computed using tau (the outputs of the network)
          Matrix<double> PDE_solutions = calculate_solution_outputs(stabilization_parameters);

          const double batch_error = PDE_solutions.calculate_sum_squared_error(targets);

          training_error += batch_error;
      }

      return training_error/normalization_coefficient;
  }

}
