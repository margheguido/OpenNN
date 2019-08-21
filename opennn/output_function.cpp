#include "output_function.h"


namespace OpenNN
{

  /// \param      tau_values          Stabilization parameters [batch_size x 1]
  /// \return                         Derivative of the PDE solution wrt the stabilization parameter [batch_size x n_Dof]
  Matrix<double> OutputFunction::calculate_PDE_solutions_derivative(const Matrix<double>& tau_values) const
  {
    // size_t nOutputs = isoglib_interface_pointer->get_nDof();
    //
    // const Matrix<double> unscaled_inputs = compute_unscaled_inputs();
    //
    // Matrix<double> grad_outputs(nOutputs, batch_size);
    // //#columns: number of instances
    // //#rows: number of outputs
    //
    // for (int i=0; i< batch_size; i++)
    // {
    //   double in_value = unscaled_train_inputs(i,0);
    //   double out_value = tau_values(i,0);
    //
    //   double h = 1;
    //   Vector<double> y = isoglib_interface_pointer->calculate_solution(out_value,in_value);
    //   double x_forward = out_value + h;
    //   Vector<double> y_forward = isoglib_interface_pointer->calculate_solution(x_forward,in_value);
    //   Vector<double> d = (y_forward - y)/h;
    //
    //   grad_outputs.set_column(i,d);
    // }
    //
    // return grad_outputs;
 }


  /// \param      tau_values        Stabilization parameters [batch_size x 1]
  /// \param      batch_indices     Indeces of the current batch, used to select the inputs [batch_size]
  /// \return                       PDE solutions [batch_size x n_Dof]
  Matrix<double> OutputFunction::calculate_PDE_solutions(const Matrix<double>& tau_values, const Vector<size_t> & batch_indices) const
  {
    size_t dofs_number = isoglib_interface_pointer->get_nDof();
    Matrix<double> solutions(batch_size, dofs_number);

    for (size_t i = 0; i < batch_size; i++)
    {
      Real mu = unscaled_inputs(batch_indices[i], 0);
      Real tau = tau_values(batch_indices[i], 0);
      Vector<double> temp_solution = isoglib_interface_pointer->calculate_solution(tau, mu);
      solutions.set_row(i, temp_solution);
    }

    return solutions;
  }

// --------------------------------------------------------------------------
// Overridden methods from NormalizedquaredError
// --------------------------------------------------------------------------

  double OutputFunction::calculate_training_error() const
  {
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

          const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

          // IMPORTANT: here the solution of the PDE is computed using tau (the outputs of the network)
          Matrix<double> PDE_solutions = calculate_PDE_solutions(outputs, training_batches[static_cast<unsigned>(i)]);

          const double batch_error = PDE_solutions.calculate_sum_squared_error(targets);

          training_error += batch_error;
      }

      return training_error/normalization_coefficient;
  }


  double OutputFunction::calculate_training_error(const Vector<double>& parameters) const
  {
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

          const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

          // IMPORTANT: here the solution of the PDE is computed using tau (the outputs of the network)
          Matrix<double> PDE_solutions = calculate_PDE_solutions(outputs, training_batches[static_cast<unsigned>(i)]);

          const double batch_error = PDE_solutions.calculate_sum_squared_error(targets);

          training_error += batch_error;
      }

      return training_error/normalization_coefficient;
  }


  double OutputFunction::calculate_selection_error() const
  {
      // Multilayer perceptron

      const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();

      // Data set

      const Vector< Vector<size_t> > selection_batches = data_set_pointer->get_instances_pointer()->get_selection_batches(batch_size);

      const size_t batches_number = selection_batches.size();

      double selection_error = 0.0;

      #pragma omp parallel for reduction(+ : selection_error)

      for(int i = 0; i < static_cast<int>(batches_number); i++)
      {
          const Matrix<double> inputs = data_set_pointer->get_inputs(selection_batches[static_cast<unsigned>(i)]);
          const Matrix<double> targets = data_set_pointer->get_targets(selection_batches[static_cast<unsigned>(i)]);

          const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

          // IMPORTANT: here the solution of the PDE is computed using tau (the outputs of the network)
          Matrix<double> PDE_solutions = calculate_PDE_solutions(outputs, selection_batches[static_cast<unsigned>(i)]);

          const double batch_error = PDE_solutions.calculate_sum_squared_error(targets);

          selection_error += batch_error;
      }

      return selection_error/selection_normalization_coefficient;
  }


  Matrix<double> OutputFunction::calculate_output_gradient(const Matrix<double>& outputs, const Matrix<double>& targets) const
  {
      // const Matrix<double> train_inputs = data_set_pointer->get_training_inputs();
      // // Matrix<double> our_outputs = calculate_PDE_solutions(outputs,train_inputs);
      //
      // Matrix<double> gradient_our_outputs = calculate_PDE_solutions_derivative(outputs);
      //
      // Matrix<double> deriv_loss = our_outputs-targets;
      //
      // Matrix<double> result;
      //
      // result.set(deriv_loss.get_rows_number(),1);
      //
      // for (int i=0; i< deriv_loss.get_rows_number(); i++)
      // {
      //     double temp_res= deriv_loss.get_row(i).dot(gradient_our_outputs.get_column(i));
      //
      //     result(i,0)=temp_res;
      // }
      //
      // return result;
  }

}
