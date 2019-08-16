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


using namespace OpenNN;

int main()
{
  try
    {
        cout << "OpenNN. New loss function for EDP solutions" << endl;

      //   srand(static_cast<unsigned>(time(nullptr)));
        srand(100);

        // Data set

        DataSet data_set;

        data_set.set_data_file_name("data/SUPG.txt");
        data_set.set_file_type("txt");
        data_set.set_separator("Tab");

        data_set.load_data();
        data_set.print_data();

        // Variables
        Variables* variables_pointer = data_set.get_variables_pointer();

        unsigned nDof = 3; // number of points in which the computed solution is sampled
        unsigned number_inputs = 1; // mu
        Vector< Variables::Item > variables_items( nDof + number_inputs );

        for( unsigned i = 0; i < number_inputs; i++ )
        {
          // variables_items[i].name = "mu";
          // variables_items[i].units = "";
          variables_items[i].use = Variables::Input;
        }
        for( unsigned i = number_inputs; i < number_inputs + nDof; i++ )
        {
          // variables_items[i].name = "dof_" + i;
          // variables_items[i].units = "";
          variables_items[i].use = Variables::Target;
        }

        variables_pointer->set_items(variables_items);

//==============================================================================
// DA QUI IN POI ANCORA DA MODIFICARE (TODO)
//==============================================================================

        const Matrix<string> inputs_information = variables_pointer->get_inputs_information();
        const Matrix<string> targets_information = variables_pointer->get_targets_information();
        /* Matrix<string> my_out_information(1,3);
        my_out_information(0,0)= "scaled_sound_pressure_level";
        my_out_information(0,1)="decibels";
        my_out_information(0,2)="pressure";*/

        // Instances

        Instances* instances_pointer = data_set.get_instances_pointer();

        instances_pointer->split_random_indices();
        // // See which indexes have been selected for testing (random selection)
        // Vector<size_t> testing_indices = instances_pointer->get_testing_indices();
        // std::cout << "testing_indeces: " << std::endl;
        // for( size_t i = 0; i < testing_indices.size(); i++ )
        //   std::cout << testing_indices[i] << " " << std::endl;


        // calls Vector< Statistics<double> > DataSet::scale_inputs_minimum_maximum()
        // dim(inputs_statistics) = #inputs, la funzione scala tutte le colonne del DataSet
        // che corrispondono ad input e per farlo ne calcola le statistiche (min, max, mean...)
        // che poi restituisce in modo da averle a disposizione senza ricalcolarle
        const Vector< Statistics<double> > inputs_statistics = data_set.scale_inputs_minimum_maximum();
        const Vector< Statistics<double> > targets_statistics = data_set.scale_targets_minimum_maximum();
        // Matrix<double> targ=data_set.get_targets();
        // cout <<"scaled targets"<<endl;
        // targ.print();


        // Neural network

        const size_t inputs_number = variables_pointer->get_inputs_number();
        const size_t hidden_perceptrons_number = 12;
        const size_t outputs_number = variables_pointer->get_inputs_number();

        NeuralNetwork neural_network(inputs_number, hidden_perceptrons_number, outputs_number);

        // neural_network.set_folder_file_names("name of the mesh directory", "name of the binary file ");

        Inputs* inputs = neural_network.get_inputs_pointer();

        inputs->set_information(inputs_information);

        Outputs* outputs = neural_network.get_outputs_pointer();

        outputs->set_information(targets_information);

        neural_network.construct_scaling_layer();

        ScalingLayer* scaling_layer_pointer = neural_network.get_scaling_layer_pointer();
        // std::cout << "Number of inputs in scaling: " << scaling_layer_pointer->get_scaling_neurons_number() << std::endl;
        scaling_layer_pointer->set_statistics(inputs_statistics);

        scaling_layer_pointer->set_scaling_methods(ScalingLayer::NoScaling);
        // scaling_layer_pointer->set_scaling_methods(ScalingLayer::MeanStandardDeviation);
        neural_network.construct_unscaling_layer();

        UnscalingLayer* unscaling_layer_pointer = neural_network.get_unscaling_layer_pointer();

        unscaling_layer_pointer->set_statistics(targets_statistics);

        unscaling_layer_pointer->set_unscaling_method(UnscalingLayer::NoUnscaling);
        // unscaling_layer_pointer->set_unscaling_method(UnscalingLayer::MeanStandardDeviation);


        // Training strategy object
        TrainingStrategy training_strategy(&neural_network, &data_set);

        OutputFunction * output_function_pointer = new OutputFunction();
        const char * dir_name="data";
        string sol_file="binary_solution";
        output_function_pointer->set_names(dir_name,sol_file);

        output_function_pointer->set_data_set_pointer(&data_set);

        output_function_pointer->set_neural_network_pointer(&neural_network);

        training_strategy.set_loss_index_pointer(output_function_pointer);

        if(1)
        {
          // QUASI_NEWTON built as default, no need of set_training_method
          QuasiNewtonMethod* quasi_Newton_method_pointer = training_strategy.get_quasi_Newton_method_pointer();

          quasi_Newton_method_pointer->set_maximum_epochs_number(100);
          quasi_Newton_method_pointer->set_display_period(10);

          quasi_Newton_method_pointer->set_minimum_loss_decrease(1.0e-6);
        }

        if(0)
        {
          training_strategy.set_training_method( "GRADIENT_DESCENT" ); // STOCHASTIC_GRADIENT_DESCENT, LEVENBERG_MARQUARDT_ALGORITHM, ADAPTIVE_MOMENT_ESTIMATION
          GradientDescent* gradient_descent_method_pointer = training_strategy.get_gradient_descent_pointer();

          gradient_descent_method_pointer->set_maximum_epochs_number(100);
          gradient_descent_method_pointer->set_display_period(10);

          gradient_descent_method_pointer->set_minimum_loss_decrease(1.0e-6);
        }

//        quasi_Newton_method_pointer->set_reserve_loss_history(true);

        const TrainingStrategy::Results training_strategy_results = training_strategy.perform_training();
        std::cout << "Training completed" << '\n';

        // Vector<double> loss_history =training_strategy_results.gradient_descent_results_pointer->loss_history;
        // Vector<double> selection_history=training_strategy_results.gradient_descent_results_pointer->selection_error_history;


        // Testing analysis

        TestingAnalysis testing_analysis(&neural_network, &data_set);
        // calls Vector< Matrix<double> > TestingAnalysis::calculate_target_outputs() const
        // dim( results ) = (#output) x (#testing_instances) x (2 = output_i + target_i da confrontare)
        Vector< Matrix<double> > results = testing_analysis.calculate_target_outputs();
        Vector<size_t> columns_to_be_scaled{ 0, 1 }; // scala sia target che output (results[0][:,0] e results[0][:,1])
        Vector< Statistics<double> > statistics1_to_scale_target_and_output{ targets_statistics[0], targets_statistics[0] };
        Vector< Statistics<double> > statistics2_to_scale_target_and_output{ targets_statistics[1], targets_statistics[1] };
        results[1].unscale_columns_minimum_maximum( statistics1_to_scale_target_and_output, columns_to_be_scaled );
        results[0].unscale_columns_minimum_maximum( statistics2_to_scale_target_and_output, columns_to_be_scaled );

        std::cout << "targets, outputs (scaled):" << '\n';
        std::cout << "first " << '\n';
        results[0].print();
        std::cout << "second" << '\n';
        results[1].print();

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
