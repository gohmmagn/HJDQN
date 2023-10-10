path_c2d = 'h5files/HJDQN_2023-07-09T121841_linear_PDE_2D_state.h5';
path_u2d = 'h5files/HJDQN_2023-07-09T121841_uncontrolled_linear_PDE_2D_state.h5';
fem_matrices2d = 'fem_matrices/fem_matrices_2023-07-09T121841_linear_PDE_2D.mat';

path_c1d = 'h5files/HJDQN_2023-07-09T133851_linear_PDE_1D_state.h5';
path_u1d = 'h5files/HJDQN_2023-07-09T133851_uncontrolled_linear_PDE_1D_state.h5';
fem_matrices1d = 'fem_matrices/fem_matrices_2023-07-09T133851_linear_PDE_1D.mat';

path_unl1d = 'h5files/HJDQN_2023-09-10T063458_uncontrolled_nonlinear_PDE_1D_state.h5';
fem_matricesnl1d = 'fem_matrices/fem_matrices_2023-09-10T063458_nonlinear_PDE_1D.mat';

path_model1d = 'h5files/Linear1dPDEEnv-v0/HJDQN_2023-07-18T134204_0_27775.pth.tar_state.h5';
path_model2d = 'h5files/Linear2dPDEEnv-v0/HJDQN_2023-07-24T101201_0_15554.pth.tar_state.h5';
path_modelNl1d = 'h5files/NonLinearPDEEnv-v0/HJDQN_2023-10-10T115414_0_11110.pth.tar_state.h5';

% subplot(1,2,1);
% h5_visualizer(path_c1d,2,1);
% subplot(1,2,2);
% h5_visualizer(path_model2d,2,2);
% 
% l2error2d = calculateL2Error(path_c2d, path_model2d, fem_matrices2d, 2, 2);
% l2norm2d = l2norm(path_c2d, fem_matrices2d, 2, 2);
% relativeError2d = l2error2d/l2norm2d;
% 
% l2error1d = calculateL2Error(path_c1d, path_model1d, fem_matrices1d, 2, 1);
% l2norm1d = l2norm(path_c1d, fem_matrices1d, 2, 1);
% relativeError1d = l2error1d/l2norm1d;
%
%h5_visualizer(path_unl1d,2,1);
h5_visualizer(path_modelNl1d,4,1);
%
%l2normnl1d = l2norm(path_unl1d, fem_matricesnl1d, 2, 1);
%l2normnl1d = l2norm(path_modelNl1d, fem_matricesnl1d, 2, 1);

function [l2Norm] = l2norm(filename, fem_matrices, T_end, dim)

    % Output argument.
    l2Norm = 0;

    % Function Information.
    data_functions = h5info(filename,"/Function/y_n");
    y_n_names = data_functions.Datasets;
    
    % Time interval Information.
    num_steps = length(y_n_names) - 1;
    T_full = 0:(T_end/num_steps):T_end;
    
    % Mesh information.
    mesh = h5read(filename,strcat("/Mesh/mesh/","geometry"));

    % Get mass matrix and time increment.
    fem_components = load(fem_matrices);

    M = fem_components.M;
    dt = fem_components.dt;

    % Set surface information.
    Z = zeros(length(mesh),size(T_full,2));
    
    if dim == 1

        for i = 1:size(T_full,2)
        
            Z(:,i) = h5read(filename,strcat("/Function/y_n/",y_n_names(i).Name)).';
        
        end
    
        Y = Z(2:end-1,:);
    
        l2Norm = sqrt(sum(sum(dt*(Y)'*M*(Y),1),2));

    end

    if dim == 2

        deleteRows = fem_components.deleteRows+1;

        for i = 1:size(T_full,2)
        
            Z(:,i) = h5read(filename,strcat("/Function/y_n/",y_n_names(i).Name)).';
        
        end
    
        Z(deleteRows,:) = [];
        
        l2Norm = sqrt(sum(sum(dt*Z'*M*Z,1),2));

    end


end

function [] = h5_visualizer(filename,T_end,dim,T_int,limz,az,sec)

    if ~exist('T_int','var')
        T_int = T_end;
    end

    if ~exist('limz','var')
        limz = 3;
    end

    if ~exist('az','var')
        az = 2;
    end
    
    if ~exist('sec','var')
        sec = 1/60;
    end
    
    % Function Information.
    data_functions = h5info(filename,"/Function/y_n");
    y_n_names = data_functions.Datasets;
    
    % Time interval Information.
    num_steps = length(y_n_names) - 1;
    T_full = 0:(T_end/num_steps):T_end;
    closest = interp1(T_full,T_full,T_int,'nearest');
    T_sub = 0:(T_end/num_steps):closest;
    
    % Mesh information.
    mesh = h5read(filename,strcat("/Mesh/mesh/","geometry"));
    
    if dim == 1
    
        % Create surface grid.
        [T,X] = meshgrid(T_sub,mesh(1,:));
        
        % Set surface information.
        Z = zeros(size(T,1),size(X,2));
    
        for i = 1:size(T_sub,2)
        
            Z(:,i) = h5read(filename,strcat("/Function/y_n/",y_n_names(i).Name)).';
        
        end
        
        surf(T,X,Z,'EdgeColor','none');
    
    end
    
    if dim == 2
    
        % Create surface grid.
        nx = round(sqrt(length(mesh)),0);
        [Y, IY] = sort(mesh(2,:),2,"ascend");
        X = mesh(1,:);
        X = X(IY);
        
        for i = 1:nx
        
            Xi = X((i-1)*nx+1:i*nx);
            IYi = IY((i-1)*nx+1:i*nx);
            [Xi_s,IXi] = sort(Xi,2,"ascend");
            X((i-1)*nx+1:i*nx) = Xi_s;
            IY((i-1)*nx+1:i*nx) = IYi(IXi);
        
        end
        
        U = reshape(X,nx,nx);
        V = reshape(Y,nx,nx);
        
        % Set surface information.
        Z = zeros(size(U,1),size(U,2),size(T_sub,2));
        
        for i = 1:size(T_sub,2)
        
            y_n = h5read(filename,strcat("/Function/y_n/",y_n_names(i).Name));
            Z(:,:,i) = reshape(y_n(IY),nx,nx);
        
        end
        
        % Plot solution.
        for i = 1:size(T_sub,2)
        
            surf(U,V,Z(:,:,i),'EdgeColor','none');
            zlim([-0.1 limz]);
            colorbar;
            view(az);
            pause(sec);
            shading interp; 
            drawnow;
        
        end
    
    end

end

function [l2error] = calculateL2Error(filename1, filename2, fem_matrices, T_end, dim)

    % Output argument.
    l2error = 0;

    % Function Information.
    data_functions1 = h5info(filename1,"/Function/y_n");
    data_functions2 = h5info(filename2,"/Function/y_n");
    y_n_names1 = data_functions1.Datasets;
    y_n_names2 = data_functions2.Datasets;
    
    % Time interval Information.
    num_steps = length(y_n_names1) - 1;
    T_full = 0:(T_end/num_steps):T_end;
    
    % Mesh information.
    mesh = h5read(filename1,strcat("/Mesh/mesh/","geometry"));

    % Get mass matrix and time increment.
    fem_components = load(fem_matrices);

    M = fem_components.M;
    dt = fem_components.dt;

    % Set surface information.
    Z1 = zeros(length(mesh),size(T_full,2));
    Z2 = zeros(length(mesh),size(T_full,2));
    
    if dim == 1

        for i = 1:size(T_full,2)
        
            Z1(:,i) = h5read(filename1,strcat("/Function/y_n/",y_n_names1(i).Name)).';
            Z2(:,i) = h5read(filename2,strcat("/Function/y_n/",y_n_names2(i).Name)).';
        
        end
    
        Y1 = Z1(2:end-1,:);
        Y2 = Z2(2:end-1,:);
    
        l2error = sqrt(sum(sum(dt*(Y1-Y2)'*M*(Y1-Y2),1),2));

    end

    if dim == 2

        deleteRows = fem_components.deleteRows+1;

        for i = 1:size(T_full,2)
        
            Z1(:,i) = h5read(filename1,strcat("/Function/y_n/",y_n_names1(i).Name)).';
            Z2(:,i) = h5read(filename2,strcat("/Function/y_n/",y_n_names2(i).Name)).';
        
        end
    
        Z1(deleteRows,:) = [];
        Z2(deleteRows,:) = [];
        
        l2error = sqrt(sum(sum(dt*(Z1-Z2)'*M*(Z1-Z2),1),2));

    end

end