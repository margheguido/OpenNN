clear all 
close all 
clc
fileID = fopen('SUPGOutput/Test1/output_250_4_2.txt', 'r');
%file of type output_n_m_k 
% where n is the number of instances corresponding to the input
% mu= linspace( 1e-m, 1e-k, n )
 
 error_size = 6; %epoch+1
 test_size = 50; % 0.2*n
  %50 se dataset è 250, 30 se dataset è 150 %40 se è 200
  linea = fgetl(fileID);
  

 
 train_error = fscanf(fileID,'%e ',error_size);
  linea = fgetl(fileID);
 validation_error =  fscanf(fileID,'%e ',error_size);
  linea = fgetl(fileID);
 
 test_mu =   fscanf(fileID,'%e  ',test_size);
 linea = fgetl(fileID);
 test_tau = fscanf(fileID,'%e ',test_size);
 linea = fgetl(fileID);
 test_tau_edp = fscanf(fileID,'%e ',test_size);
 fclose(fileID);
 
 
 
 plot([1:error_size], train_error(1:error_size)./3, '-ob')
 hold on 
 plot([1:error_size], validation_error(1:error_size), '-or')
 xlabel('epoch')
 ylabel('errors')
 
 legend('train error','validation error')
 
 b = [ 1, 1 ];
 h = 1/8;
 Pe = @(m) ( h * norm(b) ) ./ ( 2 * m );
 xi = @(theta) coth(theta) - 1 ./ theta;
 tau_2 = @(m) h ./ ( 2 * norm(b) ) .* xi(Pe(m));
 
 figure
 plot(test_mu, test_tau_edp,'-om')
 hold on 
% plot(test_mu, tau_2(test_mu),'-oc') %theoretical mu
 xlabel('mu')
 ylabel('tau')
 legend('predicted tau')
 