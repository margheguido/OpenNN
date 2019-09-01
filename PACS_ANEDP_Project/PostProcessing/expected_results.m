clear all
close all

b = [ 1, 1 ];
h = 1/8;
delta = 1/2;
mu = linspace( 1e-7, 1e-2, 100 );
Pe = @(m) ( h * norm(b) ) ./ ( 2 * m );
xi = @(theta) coth(theta) - 1 ./ theta;
tau_1 = @(m) 0.*m + delta * h ./ norm(b);
tau_2 = @(m) h ./ ( 2 * norm(b) ) .* xi(Pe(m));
tau_3 = @(m) 1 ./ ( ( 2 * norm(b) ) ./ ( h ) + ( 4 * m ) ./ ( h^2 ) );
% tau_previsti = linspace( 0.01, 0.007, 100 );

figure
plot( mu, tau_1(mu) )
hold on
plot( mu, tau_2(mu) )
plot( mu, tau_3(mu) )
plot( mu, tau_previsti )
% ylim( [ 0, 0.05 ] )
