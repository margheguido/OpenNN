clear all
close all

%%
test = 'Test3';
nRef = 3;
number_stab_refinements = 10; % quante volte viene aumentata la precisione della stima
print_figure = 1;            % ad ogni stab_refinement (ciclo 2) stampa l'andamento dell'errore



count1 = 1;

for mu = 1e-5 %logspace( -6, -5, 2 )%linspace( 1e-5, 1e-4, 10 ) %logspace( -3, -5, 6 )
    
    stab_amounts = logspace( log10(0.5e-1), log10(5e-1), 10 ); %logspace( -2, 1, 10 );
    
    for count2 = 1 : number_stab_refinements
        
        count3 = 1;
        err_L2 = [];
        
        for stab = stab_amounts
            
            [ errors, solutions, femregion, Dati, Peclet, tau ] = C_main2D( test, nRef, stab, mu );
            err_L2( count3 ) = errors.Error_L2;
            count3 = count3 + 1;
            
        end
        
        tau_values = tau * stab_amounts;
        
        if print_figure == 1
            figure
            subplot( 2, 1, 1 )
            semilogx( tau_values, err_L2, '.-' )
            xlabel( 'Tau' )
            ylabel( 'L2 error' )
            subplot( 2, 1, 2 )
            semilogx( stab_amounts, err_L2, '.-' )
            xlabel( 'stab amount' )
            ylabel( 'L2 error' )
        end
        
        [ ~, min_error_index ] = min( err_L2 );
        best_tau( count1, count2 ) = tau * stab_amounts( min_error_index );
        optimal_stab_amount = stab_amounts( min_error_index );
        if count2 == number_stab_refinements
            break;
        else
            stab_amounts = logspace( log10( stab_amounts( min_error_index - 1 ) ), ...
                                     log10( stab_amounts( min_error_index + 1 ) ), 8 );
        end
        
    end
    
    best_tau(count1) = tau * stab_amounts( min_error_index )
    
    count1 = count1 + 1;
    
end