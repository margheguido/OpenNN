clear all
close all


test_number = 1;
polynomial_degree = 1;

% Read tau
FileID = fopen( ['IsoGlib_Errors/Test',num2str(test_number),'/tau_Test',num2str(test_number),'.dat'], 'r' );
[ tau, tau_count ] = fscanf( FileID, '%f' );
fclose( FileID );

% Read errors
FileID = fopen( ['IsoGlib_Errors/Test',num2str(test_number),'/errors_Test',num2str(test_number),'_p',num2str(polynomial_degree),'.dat'], 'r' );
[ to_be_formatted, total_count ] = fscanf( FileID, '%f' );
fclose( FileID );
for i = 0 : ( length( to_be_formatted ) / 5 - 1 )
    L2_error( i + 1 ) = to_be_formatted( 5*i + 2 );
end


plot( tau, L2_error, '.-', 'linewidth', 1.2 )
title( 'L^2 error for \mu = 5e-3' )
xlabel( 'Tau' )
ylabel( '||u_h-u||_{L^2}' )

