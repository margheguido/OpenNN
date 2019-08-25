#include "output_function.h"


namespace OpenNN
{

  /// \param      tau_values        Stabilization parameters [current_batch_size x 1]
  /// \param      batch_indices     Indeces of the current batch, used to select the inputs [current_batch_size]
  /// \return                       PDE solutions [current_batch_size x n_Dof]
  Matrix<double> OutputFunction::calculate_PDE_solution(const Matrix<double>& tau_values, const Vector<size_t> & batch_indices) const
  {
      size_t n_Dof = isoglib_interface_pointer->get_nDof();
      unsigned current_batch_size = batch_indices.size();
      Matrix<double> solutions(current_batch_size, n_Dof);

      for (size_t i = 0; i < current_batch_size; i++)
      {
          Real mu = unscaled_inputs(batch_indices[i], 0);
          Real tau_scaled = tau_values(i, 0);
          Real tau = tau_scaled *0.009;
          Vector<double> temp_solution = isoglib_interface_pointer->calculate_solution(tau, mu);

          solutions.set_row(i, temp_solution);
      }

      return solutions;
  }

  /// \param      tau_values          Stabilization parameters [current_batch_size x 1]
  /// \return                         Derivative of the PDE solution wrt the stabilization parameter [current_batch_size x n_Dof]
  Matrix<double> OutputFunction::calculate_PDE_solution_derivative(const Matrix<double>& tau_values, const Vector<size_t> & batch_indices) const
  {
      unsigned n_Dof = isoglib_interface_pointer->get_nDof();
      unsigned current_batch_size = batch_indices.size();
      Matrix<double> solutions_derivatives(current_batch_size, n_Dof);

      for (int i = 0; i < current_batch_size; i++)
      {
          double mu = unscaled_inputs(batch_indices[i], 0);
          double tau = tau_values(i, 0);

          double h = 0.001;
          Vector<double> y = isoglib_interface_pointer->calculate_solution(tau, mu);
          Vector<double> y_forward = isoglib_interface_pointer->calculate_solution(tau + h, mu);
          Vector<double> derivative = ( y_forward - y ) / h;

          solutions_derivatives.set_row(i, derivative);
      }

      return solutions_derivatives;
  }

  /// \param      tau_values        Stabilization parameters [current_batch_size x 1]
  /// \param      batch_indices     Indeces of the current batch, used to select the inputs [current_batch_size]
  /// \return                       Derivative of the loss function wrt the stabilization parameter [current_batch_size x 1]
  Matrix<double> OutputFunction::calculate_loss_derivative(const Matrix<double>& tau_values, const Vector<size_t> & batch_indices) const
  {
      unsigned current_batch_size = batch_indices.size();

      Matrix<double> PDE_solutions = calculate_PDE_solution(tau_values, batch_indices);
      const Matrix<double> targets = data_set_pointer->get_targets(batch_indices);
      Matrix<double> simple_loss_derivatives = PDE_solutions - targets;

      Matrix<double> PDE_solutions_derivatives = calculate_PDE_solution_derivative(tau_values, batch_indices);

      Matrix<double> composed_loss_derivatives(current_batch_size, 1);

      for (int i = 0; i < current_batch_size; i++)
      {
          composed_loss_derivatives(i, 0) = simple_loss_derivatives.get_row(i).dot(PDE_solutions_derivatives.get_row(i));
      }

      return composed_loss_derivatives;
  }

// --------------------------------------------------------------------------
// Overridden methods from NormalizedquaredError
// --------------------------------------------------------------------------

  double OutputFunction::calculate_training_error() const
  {
      // Multilayer perceptron

      const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();
      // const BoundingLayer* bounding_layer_pointer = neural_network_pointer->get_bounding_layer_pointer(); // BOUNDING_LAYER
      // Data set

      const Vector< Vector<size_t> > training_batches = data_set_pointer->get_instances_pointer()->get_training_batches(batch_size);

      const size_t batches_number = training_batches.size();

      double training_error = 0.0;

      #pragma omp parallel for reduction(+ : training_error)

      for(int i = 0; i < static_cast<int>(batches_number); i++)
      {
          const Matrix<double> inputs = data_set_pointer->get_inputs(training_batches[static_cast<unsigned>(i)]);
          const Matrix<double> targets = data_set_pointer->get_targets(training_batches[static_cast<unsigned>(i)]);

          // const Matrix<double> outputs_nobound = multilayer_perceptron_pointer->calculate_outputs(inputs); // BOUNDING_LAYER
          // const Matrix<double> outputs = bounding_layer_pointer->calculate_outputs(outputs_nobound); // BOUNDING_LAYER

          const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

          // IMPORTANT: here the solution of the PDE is computed using tau (the outputs of the network)
          Matrix<double> PDE_solutions = calculate_PDE_solution(outputs, training_batches[static_cast<unsigned>(i)]);

          const double batch_error = PDE_solutions.calculate_sum_squared_error(targets);

          training_error += batch_error;
      }

      return training_error/normalization_coefficient;
  }


  double OutputFunction::calculate_training_error(const Vector<double>& parameters) const
  {
      // Multilayer perceptron

      const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();
      // const BoundingLayer* bounding_layer_pointer = neural_network_pointer->get_bounding_layer_pointer(); // BOUNDING_LAYER
      // Data set

      const Vector< Vector<size_t> > training_batches = data_set_pointer->get_instances_pointer()->get_training_batches(batch_size);

      const size_t batches_number = training_batches.size();

      double training_error = 0.0;

      #pragma omp parallel for reduction(+ : training_error)

      for(int i = 0; i < static_cast<int>(batches_number); i++)
      {
          const Matrix<double> inputs = data_set_pointer->get_inputs(training_batches[static_cast<unsigned>(i)]);
          const Matrix<double> targets = data_set_pointer->get_targets(training_batches[static_cast<unsigned>(i)]);

          // const Matrix<double> outputs_nobound = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters); // BOUNDING_LAYER
          // const Matrix<double> outputs = bounding_layer_pointer->calculate_outputs(outputs_nobound); // BOUNDING_LAYER

          const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs, parameters);

          // IMPORTANT: here the solution of the PDE is computed using tau (the outputs of the network)
          Matrix<double> PDE_solutions = calculate_PDE_solution(outputs, training_batches[static_cast<unsigned>(i)]);

          const double batch_error = PDE_solutions.calculate_sum_squared_error(targets);

          training_error += batch_error;
      }

      return training_error/normalization_coefficient;
  }


  double OutputFunction::calculate_selection_error() const
  {
      // Multilayer perceptron

      const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();
      // const BoundingLayer* bounding_layer_pointer = neural_network_pointer->get_bounding_layer_pointer(); // BOUNDING_LAYER
      // Data set

      const Vector< Vector<size_t> > selection_batches = data_set_pointer->get_instances_pointer()->get_selection_batches(batch_size);

      const size_t batches_number = selection_batches.size();

      double selection_error = 0.0;

      #pragma omp parallel for reduction(+ : selection_error)

      for(int i = 0; i < static_cast<int>(batches_number); i++)
      {
          const Matrix<double> inputs = data_set_pointer->get_inputs(selection_batches[static_cast<unsigned>(i)]);
          const Matrix<double> targets = data_set_pointer->get_targets(selection_batches[static_cast<unsigned>(i)]);

          // const Matrix<double> outputs_nobound = multilayer_perceptron_pointer->calculate_outputs(inputs); // BOUNDING_LAYER
          // const Matrix<double> outputs = bounding_layer_pointer->calculate_outputs(outputs_nobound); // BOUNDING_LAYER

          const Matrix<double> outputs = multilayer_perceptron_pointer->calculate_outputs(inputs);

          // IMPORTANT: here the solution of the PDE is computed using tau (the outputs of the network)
          Matrix<double> PDE_solutions = calculate_PDE_solution(outputs, selection_batches[static_cast<unsigned>(i)]);

          const double batch_error = PDE_solutions.calculate_sum_squared_error(targets);

          selection_error += batch_error;
      }

      return selection_error/selection_normalization_coefficient;
  }


  Vector<double> OutputFunction::calculate_training_error_gradient() const
  {
      // Neural network

      const MultilayerPerceptron* multilayer_perceptron_pointer = neural_network_pointer->get_multilayer_perceptron_pointer();
      // const BoundingLayer* bounding_layer_pointer = neural_network_pointer->get_bounding_layer_pointer(); // BOUNDING_LAYER

      const size_t layers_number = multilayer_perceptron_pointer->get_layers_number();

      const size_t parameters_number = multilayer_perceptron_pointer->get_parameters_number();

      // Data set

      const Vector< Vector<size_t> > training_batches = data_set_pointer->get_instances_pointer()->get_training_batches(batch_size);

      const size_t batches_number = training_batches.size();

      // Loss index

      Vector<double> training_error_gradient(parameters_number, 0.0);

      for(int i = 0; i < static_cast<int>(batches_number); i++)
      {
          const Matrix<double> inputs = data_set_pointer->get_inputs(training_batches[static_cast<unsigned>(i)]);
          const Matrix<double> targets = data_set_pointer->get_targets(training_batches[static_cast<unsigned>(i)]);

          const MultilayerPerceptron::FirstOrderForwardPropagation first_order_forward_propagation
                  = multilayer_perceptron_pointer->calculate_first_order_forward_propagation(inputs);

          // IMPORTANT: here derivative of the loss wrt tau is computed
          const Matrix<double> output_gradient
                  = calculate_loss_derivative(first_order_forward_propagation.layers_activations[layers_number-1], training_batches[static_cast<unsigned>(i)]);

          const Vector< Matrix<double> > layers_delta
                  = calculate_layers_delta(first_order_forward_propagation.layers_activation_derivatives, output_gradient);

          const Vector<double> batch_gradient
                  = calculate_error_gradient(inputs, first_order_forward_propagation.layers_activations, layers_delta);

  //        #pragma omp critical

          training_error_gradient += batch_gradient;
      }

      return training_error_gradient*2.0/normalization_coefficient;
  }

}
