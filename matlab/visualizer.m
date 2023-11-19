classdef visualizer < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                        matlab.ui.Figure
        ModelEnvironmentListBox         matlab.ui.control.ListBox
        ModelEnvironmentListBoxLabel    matlab.ui.control.Label
        RefreshFilesButton              matlab.ui.control.Button
        VisualizeSelectionButton        matlab.ui.control.Button
        UITable                         matlab.ui.control.Table
        SolutionCalculatedbyModelListBox  matlab.ui.control.ListBox
        SolutionCalculatedbyModelListBoxLabel  matlab.ui.control.Label
        FEMMatricesofExactSoltionsListBox  matlab.ui.control.ListBox
        FEMMatricesofExactSoltionsListBoxLabel  matlab.ui.control.Label
        ExactSolutionUncontrolledListBox  matlab.ui.control.ListBox
        ExactSolutionUncontrolledListBoxLabel  matlab.ui.control.Label
        ExactSolutionControlledListBox  matlab.ui.control.ListBox
        ExactSolutionControlledListBoxLabel  matlab.ui.control.Label
    end
    
    properties (Access = public)

        % h5Files: Folders h5files dir.
        dirFoldersh5Files

        % h5Files: Exact controlled files 1D.
        exactControlledSolutionLinear1D
        
        % h5Files: Exact uncontrolled files 1D.
        exactUncontrolledSolutionLinear1D
        
        % h5Files: Exact controlled files 2D.
        exactControlledSolutionLinear2D
        
        % h5Files: Exact uncontrolled files 2D.
        exactUncontrolledSolutionLinear2D
        
        % h5Files: Exact uncontrolled files non-linear.
        exactUncontrolledSolutionNonLinear1D

        % fem-matrices: 1D
        femMatricesLinear1D
        
        % fem-matrices: 2D
        femMatricesLinear2D
        
        % fem-matrices: non-linear
        femMatricesNonLinear1D

    end
    
    % Callbacks that handle component events
    methods (Access = private)

        % Button pushed function: VisualizeSelectionButton
        function visualizeSelectedSolution(app, ~)
            env = app.ModelEnvironmentListBox.Value;
            if strcmp(env,'Linear1dPDEEnv-v0')
                app.setUITable(env,1);
                visualizeEnvironments(app,env,1);
            end
            if strcmp(env,'NonLinearPDEEnv-v0')
                app.setUITable(env,1);
                visualizeEnvironments(app,env,1);
            end
            if strcmp(env,'Linear2dPDEEnv-v0')
                app.setUITable(env,2);
                visualizeEnvironments(app,env,2);
            end
        end

        % Button pushed function: RefreshFilesButton
        function refreshButtonPushed(app, ~)
            % fem-matrices:
            dirContentsFemMatrices = dir("fem_matrices");
            femMatricesStringSplit = cellfun(@(x) strsplit(x,"_"), {dirContentsFemMatrices.name}, UniformOutput=false);

            % h5Files: Only files.
            dirContentsh5Files = dir("h5files");
            idxh5FilesFiles = cell2mat({dirContentsh5Files.isdir}) == false;
            dirFilesh5Files = dirContentsh5Files(idxh5FilesFiles);
            
            % h5Files: All exact solution files.
            h5FilesStringSplitAll = cellfun(@(x) strsplit(x,"_"), {dirFilesh5Files.name}, UniformOutput=false);
            idxExactUncontrolledSolution = cellfun(@(x) strcmp(x{:,3},'uncontrolled'), h5FilesStringSplitAll);
            h5FilesStringSplitControlled = h5FilesStringSplitAll(~idxExactUncontrolledSolution);
            dirFilesh5FilesControlled = dirFilesh5Files(~idxExactUncontrolledSolution);
                   
            env = app.ModelEnvironmentListBox.Value;
            if strcmp(env,'Linear1dPDEEnv-v0')
                
                % h5Files: Exact controlled files 1D.
                idxExactControlledSolutionLinear1D = cellfun(@(x) sum(ismember(x,{'1D','linear'}))==2, h5FilesStringSplitControlled);
                exactControlledSolutionLinear1DUpdate = dirFilesh5FilesControlled(idxExactControlledSolutionLinear1D);
                app.ExactSolutionControlledListBox.Items = {exactControlledSolutionLinear1DUpdate.name};

                % h5Files: Exact uncontrolled files 1D.
                idxExactUncontrolledSolutionLinear1D = cellfun(@(x) sum(ismember(x,{'1D','uncontrolled','linear'}))==3, h5FilesStringSplitAll);
                exactUncontrolledSolutionLinear1DUpdate = dirFilesh5Files(idxExactUncontrolledSolutionLinear1D);
                app.ExactSolutionUncontrolledListBox.Items = {exactUncontrolledSolutionLinear1DUpdate.name};

                % fem-matrices: 1D.
                idxFemMatricesLinear1D = cellfun(@(x) sum(ismember(x,{'1D.mat','linear'}))==2, femMatricesStringSplit);
                femMatricesLinear1DUpdate = dirContentsFemMatrices(idxFemMatricesLinear1D);
                app.FEMMatricesofExactSoltionsListBox.Items = {femMatricesLinear1DUpdate.name};

                % Model files.
                dirContentsLinear1dPDEEnv = dir('h5files\Linear1dPDEEnv-v0\*.h5');
                app.SolutionCalculatedbyModelListBox.Items = {dirContentsLinear1dPDEEnv.name};

            end
            if strcmp(env,'Linear2dPDEEnv-v0')

                % h5Files: Exact controlled files 2D.
                idxExactControlledSolutionLinear2D = cellfun(@(x) sum(ismember(x,{'2D','linear'}))==2, h5FilesStringSplitControlled);
                exactControlledSolutionLinear2DUpdate = dirFilesh5FilesControlled(idxExactControlledSolutionLinear2D);
                app.ExactSolutionControlledListBox.Items = {exactControlledSolutionLinear2DUpdate.name};
                
                % h5Files: Exact uncontrolled files 2D.
                idxExactUncontrolledSolutionLinear2D = cellfun(@(x) sum(ismember(x,{'2D','uncontrolled','linear'}))==3, h5FilesStringSplitAll);
                exactUncontrolledSolutionLinear2DUpdate = dirFilesh5Files(idxExactUncontrolledSolutionLinear2D);
                app.ExactSolutionUncontrolledListBox.Items = {exactUncontrolledSolutionLinear2DUpdate.name};

                % fem-matrices: 2D
                idxFemMatricesLinear2D = cellfun(@(x) sum(ismember(x,{'2D.mat','linear'}))==2, femMatricesStringSplit);
                femMatricesLinear2DUpdate = dirContentsFemMatrices(idxFemMatricesLinear2D);
                app.FEMMatricesofExactSoltionsListBox.Items = {femMatricesLinear2DUpdate.name};

                % Model files.
                dirContentsLinear2dPDEEnv = dir('h5files\Linear2dPDEEnv-v0\*.h5');
                app.SolutionCalculatedbyModelListBox.Items = {dirContentsLinear2dPDEEnv.name};
            
            end
            if strcmp(env,'NonLinearPDEEnv-v0')
                app.ExactSolutionControlledListBox.Items = {};

                % h5Files: Exact uncontrolled files non-linear.
                idxExactUncontrolledSolutionNonLinear1D = cellfun(@(x) sum(ismember(x,{'1D','uncontrolled','nonlinear'}))==3, h5FilesStringSplitAll);
                exactUncontrolledSolutionNonLinear1DUpdate = dirFilesh5Files(idxExactUncontrolledSolutionNonLinear1D);
                app.ExactSolutionUncontrolledListBox.Items = {exactUncontrolledSolutionNonLinear1DUpdate.name};
    
                % fem-matrices: non-linear
                idxFemMatricesNonLinear1D = cellfun(@(x) sum(ismember(x,{'1D.mat','nonlinear'}))==2, femMatricesStringSplit);
                femMatricesNonLinear1DUpdate = dirContentsFemMatrices(idxFemMatricesNonLinear1D);
                app.FEMMatricesofExactSoltionsListBox.Items = {femMatricesNonLinear1DUpdate.name};
    
                % Model files.
                dirContentsNonLinear1dPDEEnv = dir('h5files\NonLinearPDEEnv-v0\*.h5');
                app.SolutionCalculatedbyModelListBox.Items = {dirContentsNonLinear1dPDEEnv.name};
            end
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)
            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 1002 391];
            app.UIFigure.Name = 'MATLAB App';

            % Create ModelEnvironmentListBoxLabel
            app.ModelEnvironmentListBoxLabel = uilabel(app.UIFigure);
            app.ModelEnvironmentListBoxLabel.HorizontalAlignment = 'right';
            app.ModelEnvironmentListBoxLabel.Position = [109 352 108 22];
            app.ModelEnvironmentListBoxLabel.Text = 'Model Environment';

            % Create ModelEnvironmentListBox
            app.ModelEnvironmentListBox = uilistbox(app.UIFigure, "Items", {app.dirFoldersh5Files.name});
            app.ModelEnvironmentListBox.Value = app.dirFoldersh5Files(1).name;
            app.ModelEnvironmentListBox.Position = [13 268 300 74];

            % Create ExactSolutionControlledListBoxLabel
            app.ExactSolutionControlledListBoxLabel = uilabel(app.UIFigure);
            app.ExactSolutionControlledListBoxLabel.HorizontalAlignment = 'right';
            app.ExactSolutionControlledListBoxLabel.Position = [93 231 140 22];
            app.ExactSolutionControlledListBoxLabel.Text = 'Exact Solution Controlled';

            % Create ExactSolutionControlledListBox
            app.ExactSolutionControlledListBox = uilistbox(app.UIFigure, "Items", {app.exactControlledSolutionLinear1D.name});
            app.ExactSolutionControlledListBox.Position = [13 151 300 74];

            % Create ExactSolutionUncontrolledListBoxLabel
            app.ExactSolutionUncontrolledListBoxLabel = uilabel(app.UIFigure);
            app.ExactSolutionUncontrolledListBoxLabel.HorizontalAlignment = 'right';
            app.ExactSolutionUncontrolledListBoxLabel.Position = [434 231 152 22];
            app.ExactSolutionUncontrolledListBoxLabel.Text = 'Exact Solution Uncontrolled';

            % Create ExactSolutionUncontrolledListBox
            app.ExactSolutionUncontrolledListBox = uilistbox(app.UIFigure, "Items", {app.exactUncontrolledSolutionLinear1D.name});
            app.ExactSolutionUncontrolledListBox.Position = [352 151 308 74];

            % Create FEMMatricesofExactSoltionsListBoxLabel
            app.FEMMatricesofExactSoltionsListBoxLabel = uilabel(app.UIFigure);
            app.FEMMatricesofExactSoltionsListBoxLabel.HorizontalAlignment = 'right';
            app.FEMMatricesofExactSoltionsListBoxLabel.Position = [77 107 172 22];
            app.FEMMatricesofExactSoltionsListBoxLabel.Text = 'FEM Matrices of Exact Soltions';

            % Create FEMMatricesofExactSoltionsListBox
            app.FEMMatricesofExactSoltionsListBox = uilistbox(app.UIFigure, "Items", {app.femMatricesLinear1D.name});
            app.FEMMatricesofExactSoltionsListBox.Position = [13 25 300 74];

            % Create SolutionCalculatedbyModelListBoxLabel
            app.SolutionCalculatedbyModelListBoxLabel = uilabel(app.UIFigure);
            app.SolutionCalculatedbyModelListBoxLabel.HorizontalAlignment = 'right';
            app.SolutionCalculatedbyModelListBoxLabel.Position = [426 105 160 22];
            app.SolutionCalculatedbyModelListBoxLabel.Text = 'Solution Calculated by Model';

            % Create SolutionCalculatedbyModelListBox
            app.SolutionCalculatedbyModelListBox = uilistbox(app.UIFigure, "Items", {dir('h5files\Linear1dPDEEnv-v0\*.h5').name});
            app.SolutionCalculatedbyModelListBox.Position = [352 25 308 74];

            % Create UITable
            app.UITable = uitable(app.UIFigure, "Data",zeros(5,1));
            app.UITable.ColumnName = {'Errors and Norms'};
            app.UITable.RowName = {'L2 err: Controlled - Model','L2 rel err: Controlled - Model','L2 Norm: Controlled','L2 Norm: Uncontrolled','L2 Norm: Model'};
            app.UITable.Position = [679 96 302 185];

            % Create VisualizeSelectionButton
            app.VisualizeSelectionButton = uibutton(app.UIFigure, 'push');
            app.VisualizeSelectionButton.ButtonPushedFcn = createCallbackFcn(app, @visualizeSelectedSolution, true);
            app.VisualizeSelectionButton.Position = [470 294 116 23];
            app.VisualizeSelectionButton.Text = 'Visualize Selection';

            % Create RefreshFilesButton
            app.RefreshFilesButton = uibutton(app.UIFigure, 'push');
            app.RefreshFilesButton.ButtonPushedFcn = createCallbackFcn(app, @refreshButtonPushed, true);
            app.RefreshFilesButton.Position = [352 294 100 23];
            app.RefreshFilesButton.Text = 'Refresh Files';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = visualizer()

            % h5Files: Only files.
            dirContentsh5Files = dir("h5files");
            idxh5FilesFiles = cell2mat({dirContentsh5Files.isdir}) == false;
            dirFilesh5Files = dirContentsh5Files(idxh5FilesFiles);
            
            % h5Files: Only folders.
            app.dirFoldersh5Files = dirContentsh5Files(~idxh5FilesFiles);
            idxh5FilesFolders = cellfun(@(x) sum(ismember(x,{'Linear1dPDEEnv-v0', 'Linear2dPDEEnv-v0', 'NonLinearPDEEnv-v0'}))>0, {app.dirFoldersh5Files.name});
            app.dirFoldersh5Files = app.dirFoldersh5Files(idxh5FilesFolders);
            
            % h5Files: All exact solution files.
            h5FilesStringSplitAll = cellfun(@(x) strsplit(x,"_"), {dirFilesh5Files.name}, UniformOutput=false);
            idxExactUncontrolledSolution = cellfun(@(x) strcmp(x{:,3},'uncontrolled'), h5FilesStringSplitAll);
            h5FilesStringSplitControlled = h5FilesStringSplitAll(~idxExactUncontrolledSolution);
            dirFilesh5FilesControlled = dirFilesh5Files(~idxExactUncontrolledSolution);

            % h5Files: Exact controlled files 1D.
            idxExactControlledSolutionLinear1D = cellfun(@(x) sum(ismember(x,{'1D','linear'}))==2, h5FilesStringSplitControlled);
            app.exactControlledSolutionLinear1D = dirFilesh5FilesControlled(idxExactControlledSolutionLinear1D);
            
            % h5Files: Exact uncontrolled files 1D.
            idxExactUncontrolledSolutionLinear1D = cellfun(@(x) sum(ismember(x,{'1D','uncontrolled','linear'}))==3, h5FilesStringSplitAll);
            app.exactUncontrolledSolutionLinear1D = dirFilesh5Files(idxExactUncontrolledSolutionLinear1D);
            
            % h5Files: Exact controlled files 2D.
            idxExactControlledSolutionLinear2D = cellfun(@(x) sum(ismember(x,{'2D','linear'}))==2, h5FilesStringSplitControlled);
            app.exactControlledSolutionLinear2D = dirFilesh5FilesControlled(idxExactControlledSolutionLinear2D);
            
            % h5Files: Exact uncontrolled files 2D.
            idxExactUncontrolledSolutionLinear2D = cellfun(@(x) sum(ismember(x,{'2D','uncontrolled','linear'}))==3, h5FilesStringSplitAll);
            app.exactUncontrolledSolutionLinear2D = dirFilesh5Files(idxExactUncontrolledSolutionLinear2D);
            
            % h5Files: Exact uncontrolled files non-linear.
            idxExactUncontrolledSolutionNonLinear1D = cellfun(@(x) sum(ismember(x,{'1D','uncontrolled','nonlinear'}))==3, h5FilesStringSplitAll);
            app.exactUncontrolledSolutionNonLinear1D = dirFilesh5Files(idxExactUncontrolledSolutionNonLinear1D);

            % fem-matrices:
            dirContentsFemMatrices = dir("fem_matrices");
            femMatricesStringSplit = cellfun(@(x) strsplit(x,"_"), {dirContentsFemMatrices.name}, UniformOutput=false);
            
            % fem-matrices: 1D
            idxFemMatricesLinear1D = cellfun(@(x) sum(ismember(x,{'1D.mat','linear'}))==2, femMatricesStringSplit);
            app.femMatricesLinear1D = dirContentsFemMatrices(idxFemMatricesLinear1D);
            
            % fem-matrices: 2D
            idxFemMatricesLinear2D = cellfun(@(x) sum(ismember(x,{'2D.mat','linear'}))==2, femMatricesStringSplit);
            app.femMatricesLinear2D = dirContentsFemMatrices(idxFemMatricesLinear2D);
            
            % fem-matrices: non-linear
            idxFemMatricesNonLinear1D = cellfun(@(x) sum(ismember(x,{'1D.mat','nonlinear'}))==2, femMatricesStringSplit);
            app.femMatricesNonLinear1D = dirContentsFemMatrices(idxFemMatricesNonLinear1D);

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end

        function [pathControlled,pathUncontrolled,pathModel,pathFemMatrices] = getFilePaths(app,env)
            if strcmp(env,'NonLinearPDEEnv-v0')
                pathControlled = "";
                pathUncontrolled = strcat('h5files/',app.ExactSolutionUncontrolledListBox.Value);
                pathModel = strcat('h5files/',env,'/',app.SolutionCalculatedbyModelListBox.Value);
                pathFemMatrices = strcat('fem_matrices/',app.FEMMatricesofExactSoltionsListBox.Value);
            else
                pathControlled = strcat('h5files/',app.ExactSolutionControlledListBox.Value);
                pathUncontrolled = strcat('h5files/',app.ExactSolutionUncontrolledListBox.Value);
                pathModel = strcat('h5files/',env,'/',app.SolutionCalculatedbyModelListBox.Value);
                pathFemMatrices = strcat('fem_matrices/',app.FEMMatricesofExactSoltionsListBox.Value);
            end
        end

        function [] = visualizeEnvironments(app,env,dim)
            if strcmp(env,'NonLinearPDEEnv-v0')
                [~,pathUncontrolled,pathModel,pathFemMatrices] = app.getFilePaths(env);
                subplot(1,2,1);
                app.h5_visualizer(pathUncontrolled,pathFemMatrices,dim,'Exact Uncontrolled Solution');
                subplot(1,2,2);
                app.h5_visualizer(pathModel,pathFemMatrices,dim,'Model Solution');
            else
                [pathControlled,pathUncontrolled,pathModel,pathFemMatrices] = app.getFilePaths(env);
                subplot(1,3,1);
                app.h5_visualizer(pathUncontrolled,pathFemMatrices,dim,'Exact Uncontrolled Solution');
                subplot(1,3,2);
                app.h5_visualizer(pathControlled,pathFemMatrices,dim,'Exact Controlled Solution');
                subplot(1,3,3);
                app.h5_visualizer(pathModel,pathFemMatrices,dim,'Model Solution');
            end
        end

        function [] = setUITable(app,env,dim)
            if strcmp(env,'NonLinearPDEEnv-v0')
                [~,pathUncontrolled,pathModel,pathFemMatrices] = app.getFilePaths(env);
                l2errControlledModel = "-";
                l2normControlled = "-";
                l2relErrControlledModel = "-";
                l2normUncontrolled = app.l2norm(pathUncontrolled, pathFemMatrices,dim);
                l2normModel = app.l2norm(pathModel, pathFemMatrices,dim);
                app.UITable.Data = reshape([l2errControlledModel, l2relErrControlledModel, l2normControlled, l2normUncontrolled, l2normModel],5,1);
            else
                [pathControlled,pathUncontrolled,pathModel,pathFemMatrices] = app.getFilePaths(env);
                l2errControlledModel = app.calculateL2Error(pathControlled, pathModel, pathFemMatrices, dim);
                l2normControlled = app.l2norm(pathControlled, pathFemMatrices, dim);
                l2relErrControlledModel = l2errControlledModel/l2normControlled;
                l2normUncontrolled = app.l2norm(pathUncontrolled, pathFemMatrices, dim);
                l2normModel = app.l2norm(pathModel, pathFemMatrices, dim);
                app.UITable.Data = reshape([l2errControlledModel, l2relErrControlledModel, l2normControlled, l2normUncontrolled, l2normModel],5,1);
            end
        end

        function [l2Norm] = l2norm(~,filename,fem_matrices,dim)

            % Output argument.
            l2Norm = 0;
        
            % Function Information.
            data_functions = h5info(filename,"/Function/y_n");
            y_n_names = data_functions.Datasets;
        
            % Get FEM components.
            fem_components = load(fem_matrices);
        
            % Get time components.
            dt = double(fem_components.dt);
            num_steps = fem_components.num_steps;
            T_end = double(dt*num_steps);
            
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
        
        function [] = h5_visualizer(~,filename,fem_matrices,dim,figTitle,T_int,limz,az,sec)
        
            % Get num_steps and dt.
            fem_components = load(fem_matrices);
        
            dt = double(fem_components.dt);
            num_steps = fem_components.num_steps;
        
            T_end = double(dt*num_steps);
        
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
                title(figTitle);
            
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
                    title(figTitle);
                    zlim([-0.1 limz]);
                    colorbar;
                    view(az);
                    pause(sec);
                    shading interp; 
                    drawnow;
                
                end
            
            end
        
        end
        
        function [l2error] = calculateL2Error(~,filename1,filename2,fem_matrices,dim)
        
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
            dt = double(fem_components.dt);
            num_steps = fem_components.num_steps;
            T_end = double(dt*num_steps);
        
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

    end
end