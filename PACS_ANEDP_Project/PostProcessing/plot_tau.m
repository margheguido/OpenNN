
clear all
close all
clc
mu= [ 1e-5, 5e-5, 1e-4, 5e-4, 1e-3];
% tau_th= [4.42, 4.42,4.41,4.39,4.37,4.17].*1e-2; %theoretical tau 
tau_opt=[9.56, 9.52, 9.46, 9, 8.43,].*1e-3; %optimal tau computed using IsoGlib
%tau_pred=[9.96, 9.48, 9.44, 9.12, 8.42, 3.35].*1e-3; %tau predicted by ANN
%tau_pred2=[ 0.003089490000000, 0.003022120000000, 0.002924120000000, 0.002306330000000, 0.001665060000000, 0.001108800000000];
%tau_pred= [0.009935750000000,0.009355780000000,0.009091330000000,0.008725990000000, 0.008617390000000,0.004374510000000];
tau_opt_2 = [ 0.0134, 0.014, 0.015, 0.0157, 0.0159 ];
mu_2 = [  0.001, 0.0008, 0.0005, 0.0001, 0.00005 ];


plot (mu, tau_opt, '-or')
hold on
plot (mu_2, tau_opt_2, '-*b')
xlabel('mu')
ylabel('tau')
legend(  'L2-optimal tau Test1', 'L2-optimal tau Test2')