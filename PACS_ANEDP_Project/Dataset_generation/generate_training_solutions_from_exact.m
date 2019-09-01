clear all
close all

SUPG_ANN_dataset = fopen( 'SUPG.txt', 'w' );

compute_on_gauss_points = 1; % 0 -> sample on dofs, 1 -> sample on n_gauss_points gauss points per element

% Print headers
fprintf( SUPG_ANN_dataset, 'mu          \t' );
n_dofs = 81;         % n_dofs = 9 ---> nRef = 1, n_dofs = 25 ---> nRef = 2, n_dofs = 81 ---> nRef = 3, ...
n_elems = 64;        % n_elems = 4 --> nRef = 1, n_elems = 16 --> nRef = 2, n_elems = 64 --> nRef = 3, ...
n_gauss_points = 4;  % gauss points to integrate over a songle element
for i = 1 : n_elems
    for k = 1 : n_gauss_points
        fprintf( SUPG_ANN_dataset, 'e_%1.0f_p_%1.0f       \t', i, k );
    end
end
fprintf( SUPG_ANN_dataset, '\n' );
for mu = linspace( 1e-5, 1e-3, 200 )
%          [ 1e-6, 2e-6, 4e-6, 8e-6,...
%            1e-5, 2e-5, 3e-5, 4e-5, 6e-5, 8e-5,...
%            1e-4, 2e-4, 3e-4, 4e-4, 5e-4, 6e-4, 7e-4, 8e-4, 9e-4,...
%            1e-3 ] 
    
%     [errors,solutions,femregion,Dati,Peclet,tau] = C_main2D('Test1',1,0,mu);
%     exact_solution = Dati.exact_sol;
%     n_dofs = femregion.ndof;

    % Exact solution
   % exact_solution = '-atan(((x-0.5).^2+(y-0.5).^2-1/16)./(sqrt(mu)))';  % Test2, tipo panettone
   exact_solution = 'sin(2*pi*x).*(y-y.^2)';          % Test3, due gobbe (GuidoVidulisADRExactSol)
% exact_solution = 'exp(-y).*(-x.^2+x)'; %inventata 
 %exact_solution = '0.1 .* exp(4.*y) .* (y-1) .* (x-1) .* x'; %invetata2
    
    % Compute sampling points
    h = 1 / ( sqrt( n_dofs ) - 1 );
    x = [];
    y = [];
    
    if compute_on_gauss_points == 0
        
        for i = 1 : sqrt( n_dofs )
           x = [ x, 0 : h : 1 ]; 
        end
        for i = 0 : sqrt( n_dofs ) - 1
           y = [ y, ones( 1, sqrt( n_dofs ) ) * h * i ]; 
        end
        
    elseif compute_on_gauss_points == 1
        
        val_1 = 0.098584391824351;
        val_2 = 0.026415608175648;
        
        current_element = 1;
        
        for Y = 0 : h : h*( sqrt( n_dofs ) - 2 )
           
            for X = 0 : h : h*( sqrt( n_dofs ) - 2 )
               
                x( ( current_element - 1 ) * 4 + 1 ) = X + val_1;
                y( ( current_element - 1 ) * 4 + 1 ) = Y + val_1;
                
                x( ( current_element - 1 ) * 4 + 2 ) = X + val_2;
                y( ( current_element - 1 ) * 4 + 2 ) = Y + val_1;
                
                x( ( current_element - 1 ) * 4 + 3 ) = X + val_1;
                y( ( current_element - 1 ) * 4 + 3 ) = Y + val_2;
                
                x( ( current_element - 1 ) * 4 + 4 ) = X + val_2;
                y( ( current_element - 1 ) * 4 + 4 ) = Y + val_2;
               
                current_element = current_element + 1;
                
            end
           
        end
       
    end
        
    
    % Evaluate exact solution at sampling points
    sampled_exact_solution = eval( exact_solution );
    
    % Print on file
    fprintf( SUPG_ANN_dataset, '%10.9f \t', mu );
    fprintf( SUPG_ANN_dataset, '%10.9f \t', sampled_exact_solution );
    fprintf( SUPG_ANN_dataset, '\n' );
    
end

fclose( SUPG_ANN_dataset );

