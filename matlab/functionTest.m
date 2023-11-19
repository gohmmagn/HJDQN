dim = 1;
pathControlled = "";
pathUncontrolled = "";
pathModel = "";
pathFemMatrices = "";

% Linear Case

subplot(1,3,1);
h5_visualizer(pathUncontrolled,pathFemMatrices,dim);
subplot(1,3,2);
h5_visualizer(pathControlled,pathFemMatrices,dim);
subplot(1,3,3);
h5_visualizer(pathModel,pathFemMatrices,dim);

l2errControlledModel = calculateL2Error(pathControlled, pathModel, pathFemMatrices, dim);
l2normControlled = l2norm(pathControlled, pathFemMatrices, dim);
l2relErrControlledModel = l2errControlledModel/l2normControlled;
l2normUncontrolled = l2norm(pathUncontrolled, pathFemMatrices, dim);
l2normModel = l2norm(pathModel, pathFemMatrices, dim);

% Non-Linear Case

subplot(1,2,1);
h5_visualizer(pathUncontrolled,pathFemMatrices,dim);
subplot(1,2,2);
h5_visualizer(pathModel,pathFemMatrices,dim);

l2errControlledModel = "-";
l2normControlled = "-";
l2relErrControlledModel = "-";
l2normUncontrolled = l2norm(pathUncontrolled, pathFemMatrices,dim);
l2normModel = l2norm(pathModel, pathFemMatrices,dim);

function [l2Norm] = l2norm(filename,fem_matrices,dim)

    % Output argument.
    l2Norm = 0;

    % Function Information.
    data_functions = h5info(filename,"/Function/y_n");
    y_n_names = data_functions.Datasets;

    % Get FEM components.
    fem_components = load(fem_matrices);

    % Get time components.
    dt = fem_components.dt;
    num_steps = fem_components.num_steps;
    T_end = dt*num_steps;
    
    % Time interval Information.
    time_steps = length(y_n_names) - 1;
    T_full = 0:(T_end/time_steps):T_end;
    
    % Mesh information.
    mesh = h5read(filename,strcat("/Mesh/mesh/","geometry"));

    % Get mass matrix.
    M = fem_components.M;

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

function [] = h5_visualizer(filename,fem_matrices,dim,T_int,limz,az,sec)

    % Get num_steps and dt.
    fem_components = load(fem_matrices);

    dt = fem_components.dt;
    num_steps = fem_components.num_steps;

    T_end = dt*num_steps;

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
        sec = 1/25;
    end
    
    % Function Information.
    data_functions = h5info(filename,"/Function/y_n");
    y_n_names = data_functions.Datasets;

    % Resort state values (necesarry when T_end > 10)
    if T_end > 10
        [~,newIndex] = sort(cellfun(@(str) str2double(strrep(str,'_','.')), {y_n_names.Name}.'));
        y_n_names = y_n_names(newIndex);
    end
    
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

function [l2error] = calculateL2Error(filename1,filename2,fem_matrices,dim)

    % Output argument.
    l2error = 0;

    % Function Information.
    data_functions1 = h5info(filename1,"/Function/y_n");
    data_functions2 = h5info(filename2,"/Function/y_n");
    y_n_names1 = data_functions1.Datasets;
    y_n_names2 = data_functions2.Datasets;
    
    % Get FEM components.
    fem_components = load(fem_matrices);

    % Get time components.
    dt = fem_components.dt;
    num_steps = fem_components.num_steps;
    T_end = dt*num_steps;

    % Time interval Information.
    time_steps = length(y_n_names1) - 1;
    T_full = 0:(T_end/time_steps):T_end;
    
    % Mesh information.
    mesh = h5read(filename1,strcat("/Mesh/mesh/","geometry"));

    % Get mass matrix.
    M = fem_components.M;

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