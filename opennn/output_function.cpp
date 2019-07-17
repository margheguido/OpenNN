#include "output_function.h"


namespace OpenNN
{


Matrix<double> OutputFunction::gradient_outputs (const Matrix<double>& single_output,const Matrix<double>& multiple_outputs) const
{
   Matrix<double> grad_outputs(multiple_outputs.get_columns_number(),single_output.get_rows_number());
   //#columns: number of outputs
   //#rows: number of instances

   for (int i=0; i< single_output.get_rows_number(); i++)
   {
     double out_value=single_output(i,0);
     double h=0.0001;
     Vector<double> y = calculate_multiple_outputs(out_value);
     double x_forward = out_value + h;
     Vector<double> y_forward = calculate_multiple_outputs(x_forward);
     Vector<double> d = (y_forward - y)/h;

     grad_outputs.set_column(i,d);
    }

    return grad_outputs;
 }

 Matrix<double> OutputFunction::calculate_multiple_outputs (const Matrix<double>& single_output) const
  {
   Matrix<double> multiple_outputs(single_output.get_rows_number(),2);
   multiple_outputs.set_column(0,single_output,"old_out");
   multiple_outputs.set_column(1,single_output+single_output,"new_out");

   return multiple_outputs;
  }

 Vector<double> OutputFunction::calculate_multiple_outputs (double single_output) const
  {
   Vector<double> multiple_outputs(2);
   multiple_outputs[0]=single_output;
   multiple_outputs[1]=single_output*2;

   return multiple_outputs;
  }

  void OutputFunction::load_solution_binary(string data_file_name)
  {
      ifstream file;
      file.open(data_file_name.c_str(), ios::binary);

      if(!file.is_open())
      {
          ostringstream buffer;

          buffer << "OpenNN Exception: OutputFunction class.\n"
                 << "void load_solution_binary() method.\n"
                 << "Cannot open data file: " << data_file_name << "\n";

          throw logic_error(buffer.str());
      }

      streamsize size = sizeof(size_t);
      size = sizeof(double);

      double value;

      size_t nDof= 25; //numero di punti in cui c'è la soluzione
      solution_stab.set(nDof, 1); //vector of the solution

      for(size_t i = 0; i < nDof; i++)
      {
          file.read(reinterpret_cast<char*>(&value), size);

          solution_stab[i] = value;
      }

      file.close();
  }

  void OutputFunction::print_solution() const
  {
    solution_stab.print();
  }
}
