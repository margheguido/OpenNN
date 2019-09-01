// System includes

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <time.h>

// OpenNN includes

#include "opennn.h"
#include "isoglib_interface.h"
#include "output_function.h"
#include "SUPGdataBase.h"
#include "data/Test1/SUPGdata1.h"
#include "data/Test2/SUPGdata2.h"


using namespace OpenNN;

int main()
{
  try
    {
        cout << "OpenNN. New loss function for EDP solutions" << endl;

        srand(1);

        //======================================================================
        // EXTERNAL FILES (meshload.dat and dataset.txt)
        //======================================================================

        unsigned test_number = 1;
        unsigned ref_number = 3;
        string dataset_name = "data/Test" + to_string(test_number) + "/ref" + to_string(ref_number)+ "/SUPG.txt";
        string meshload_folder = "data/Test" + to_string(test_number) + "/ref" + to_string(ref_number);

        //======================================================================
        // DATASET
        //======================================================================

        DataSet data_set;
        data_set.set_header_line(true); // Our data set has a name for every column
        data_set.set_data_file_name(dataset_name);
        data_set.set_file_type("txt");
        data_set.set_separator("Tab");
        data_set.load_data();
        data_set.print_data();

        //======================================================================
        // VARIABLES (COLUMNS OF THE DATASET)
        //======================================================================

        Variables* variables_pointer = data_set.get_variables_pointer();

        // Number of inputs of the ANN (only mu in this case)
        unsigned inputs_number = 1;
        // Number of gauss points for every element (it's an internal parameter of isoglib, by default set to 4)
        unsigned gauss_points_number = 4;
        // Number of elements
        unsigned elements_number = ( variables_pointer->get_variables_number() - inputs_number ) / gauss_points_number; // nRef3 -> 64
        // Number of degrees of freedom
        unsigned dofs_number = square( sqrt( elements_number ) + 1 ); // nRef -> 81

        // Vector of items, each one corresponding to the columns of the dataset
        Vector< Variables::Item > variables_items( inputs_number + elements_number*gauss_points_number );
        // Set type of item (input or target)
        for( unsigned i = 0; i < inputs_number; i++ )
        {
          variables_items[i].use = Variables::Input;
        }
        for( unsigned i = inputs_number; i < inputs_number + elements_number*gauss_points_number; i++ )
        {
          variables_items[i].use = Variables::Target;
        }
        variables_pointer->set_items(variables_items);

        // Get inputs and outputs informations
        const Matrix<string> inputs_information = variables_pointer->get_inputs_information();
        const Matrix<string> targets_information = variables_pointer->get_targets_information();

        //======================================================================
        // INSTANCES (ROWS OF THE DATASET)
        //======================================================================

        Instances* instances_pointer = data_set.get_instances_pointer();
        // Assign every row to a task: training, validation or testing (respectively: 60%, 20%, 20%)
        instances_pointer->split_random_indices();

        //======================================================================
        // NEURAL NETWORK
        //======================================================================

        // const size_t inputs_number = ... [see above, already chosen]
        const size_t hidden_perceptrons_number = 12;
        const size_t outputs_number = 1;
        NeuralNetwork neural_network(inputs_number, hidden_perceptrons_number, outputs_number);

        // Set activation funcion of the last layer so that output is always positive
        MultilayerPerceptron* multilayer_perceptron_pointer = neural_network.get_multilayer_perceptron_pointer();
        multilayer_perceptron_pointer->set_layer_activation_function(1, PerceptronLayer::SoftPlus);

        // Set inputs and outputs informations
        Inputs* inputs = neural_network.get_inputs_pointer();
        inputs->set_information(inputs_information);
        Outputs* outputs = neural_network.get_outputs_pointer();
        outputs->set_information(targets_information);

        //======================================================================
        // TRAINING
        //======================================================================

        TrainingStrategy training_strategy(&neural_network, &data_set);

        // Choice of the optimization algorithm and setup
        training_strategy.set_training_method( "GRADIENT_DESCENT" );
        GradientDescent* gradient_descent_method_pointer = training_strategy.get_gradient_descent_pointer();
        gradient_descent_method_pointer->set_maximum_epochs_number(2);
        gradient_descent_method_pointer->set_display_period(1);
        gradient_descent_method_pointer->set_minimum_loss_decrease(1.0e-6);
        gradient_descent_method_pointer->set_maximum_time(2000);
        gradient_descent_method_pointer->set_reserve_error_history(true);
        gradient_descent_method_pointer->set_reserve_selection_error_history(true);

        // Definition of the PDE: diffusion and advection coefficients, forcing term
        SUPGdata1 data1;
        SUPGdata2 data2;
        SUPGdataBase* data_pointer;
        if( test_number == 1 )
          data_pointer = &data1;
        else if( test_number == 2 )
          data_pointer = &data2;

        // Iniziatialization of the new loss function depending on the PDE chosen
        OutputFunction *output_function_pointer = new OutputFunction(meshload_folder, data_pointer);
        IsoglibInterface* isoglibinterface_pointer = output_function_pointer->get_isoglib_interface_pointer();
        isoglibinterface_pointer->set_nDof(dofs_number);
        isoglibinterface_pointer->set_nElems(elements_number);
        isoglibinterface_pointer->set_nGaussPoints(gauss_points_number);
        // Save the original inputs inside OutputFunction before scaling them for the neural network
        output_function_pointer->set_unscaled_inputs(data_set.get_inputs());
        // Normalize the value of training and selection error so that they are comparable (its absolute value is meaningless)
        unsigned training_instances_number = instances_pointer->get_training_instances_number();
        unsigned selection_instances_number = instances_pointer->get_selection_instances_number();
        output_function_pointer->set_normalization_coefficient(training_instances_number);
        output_function_pointer->set_selection_normalization_coefficient(selection_instances_number);
        // Link the loss to data and ANN
        output_function_pointer->set_data_set_pointer(&data_set);
        output_function_pointer->set_neural_network_pointer(&neural_network);

        // Set the new loss inside the training object
        training_strategy.set_loss_index_pointer(output_function_pointer);

        // The ANN works well between with inputs of order zero: scale the inputs (targets are not scaled since represent the exact solution)
        const Vector< Statistics<double> > inputs_statistics = data_set.scale_inputs_minimum_maximum();

        // Start training
        const TrainingStrategy::Results training_strategy_results = training_strategy.perform_training();

        //======================================================================
        // TRAINING AND SELECTION ERRORS
        //======================================================================

        // Get learning history
        Vector<double> loss_history = training_strategy_results.gradient_descent_results_pointer->loss_history;
        Vector<double> selection_history = training_strategy_results.gradient_descent_results_pointer->selection_error_history;
        size_t loss_size = loss_history.size();
        size_t selection_size = selection_history.size();

        // Print training and selection errors (on screen)
        std::cout << '\n';
        std::cout << "================================================" << '\n';
        std::cout << "               Learning history" << '\n';
        std::cout << "================================================" << '\n';
        std::cout << '\n';
        std::cout << "Training error: " << '\n';
        for( unsigned i = 0; i < loss_size; i++ )
        {
          std::cout << loss_history[i] << '\n';
        }
        std::cout << '\n';
        std::cout << "Validation error: " << '\n';
        for( unsigned i = 0; i < selection_size; i++ )
        {
          std::cout << selection_history[i] << '\n';
        }

        //======================================================================
        // TESTING
        //======================================================================

        TestingAnalysis testing_analysis(&neural_network, &data_set);
        // dim( results ) = (#output) x (#testing_instances) x (2 = target_i + output_i)
        Vector< Matrix<double> > results = testing_analysis.calculate_target_outputs();

        // Print inputs and predicted outputs (on screen)
        Matrix<double> testing_inputs = data_set.get_testing_inputs();
        unsigned testing_instances_number = testing_inputs.get_rows_number();
        testing_inputs.unscale_columns_minimum_maximum( inputs_statistics, {0} );
        std::cout << '\n';
        std::cout << "================================================" << '\n';
        std::cout << "               Testing analysis" << '\n';
        std::cout << "================================================" << '\n';
        std::cout << '\n';
        std::cout << "[Input] Mu: " << '\n';
        for( unsigned i = 0; i < testing_instances_number; i++ )
        {
          std::cout << testing_inputs(i,0) << '\n';
        }
        std::cout << '\n';
        std::cout << "[Output] Unscaled tau: "<< '\n';
        for( unsigned i = 0; i < testing_instances_number; i++ )
        {
          std::cout << results[0](i,1) << '\n';
        }
        std::cout << '\n';
        double tau_scaling = isoglibinterface_pointer->get_tau_scaling();
        std::cout << "Predicted tau for EDP: "<< '\n';
        for( unsigned i = 0; i < testing_instances_number; i++ )
        {
          std::cout << results[0](i,1) * tau_scaling<< '\n';
        }
        std::cout << '\n';

        return 0 ;
    }
    catch(exception& e)
    {
        cerr << e.what() << endl;

        return(1);
    }

}


// OpenNN: Open Neural Networks Library.
// Copyright (C) Artificial Intelligence Techniques SL.
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
