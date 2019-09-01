clear all
close all

%% Test1

test = 'Test3';
mu = 1 * 1e-3;
nRef = 3;
precision = 1;

ii = 1;
if precision == 0
    stabilization_amount = 10.^[ -2.5, -2, -1.5, -1, -0.75, -0.5, -0.3, -0.1, 0, 0.5, 1 ];
elseif precision == 1
    stabilization_amount = linspace( 0.85e-1, 0.95e-1, 8 );
%     stabilization_amount = [ 0.05 : 0.01 : 0.15 ];
end

for stab = stabilization_amount
    [ errors, solutions, femregion, Dati, Peclet, tau ] = C_main2D( test, nRef, stab, mu );
    err_L2( ii ) = errors.Error_L2;
    err_H1( ii ) = errors.Error_H1;
    ii = ii + 1;
end

tau_values = tau * stabilization_amount;

exact_tau_index = find( stabilization_amount == 1 );

[ ~, min_error_index ] = min( err_L2 );
best_tau = tau_values( min_error_index )

%%
figure

subplot( 2, 1, 1 )
semilogx( tau_values, err_L2, '.-' )
hold on
scatter( tau_values(exact_tau_index), err_L2(exact_tau_index), 'r' )
xlabel( 'Tau' )
ylabel( 'L2 error' )

subplot( 2, 1, 2 )
% semilogx( tau_values, err_H1, '.-' )
% hold on
% scatter( tau_values(exact_tau_index), err_H1(exact_tau_index), 'r' )
% xlabel( 'Tau' )
% ylabel( 'H1 error' )
semilogx( stabilization_amount, err_L2, '.-' )
xlabel( 'stabilization_amount' )
ylabel( 'L2 error' )
