paths = kIn.(1);
timestamps = kIn.(2);
Labels=["Name", "n", "m", "num_steps", "dt", "Omega", "nu", "a_Name", "b_Name", "alpha", "beta", "acctuators"];

for i = 1:size(paths,1)

    path = paths{i,:};
    timestamp = timestamps(i,:);

    fem_matrices = load(path);

    Ad = fem_matrices.Ad;
    B = fem_matrices.B;
    M = fem_matrices.M;
    Q = fem_matrices.Q;
    R = fem_matrices.R;
    S = zeros(size(B));
    Omega = sprintf("%g ",reshape(fem_matrices.Omega, 1, []));
    n = num2str(fem_matrices.n);
    m = num2str(fem_matrices.m);
    num_steps = num2str(fem_matrices.num_steps);
    dt = num2str(fem_matrices.dt);
    nu = num2str(fem_matrices.nu);
    a_Name = num2str(fem_matrices.a_Name);
    b_Name = num2str(fem_matrices.b_Name);
    acctuators = sprintf("%g ",reshape(fem_matrices.acctuators, 1, []));
    alpha = num2str(fem_matrices.alpha);
    beta = num2str(fem_matrices.beta);

    [X,K,L,info] = icare(Ad,B,Q,R,S,M);

    if info.Report == 0

        data = [strcat("K_",timestamp), n, m, num_steps, dt, Omega, nu, a_Name, b_Name, alpha, beta, acctuators];
        writetable(array2table(data, 'VariableNames', Labels), 'riccati_solution_matrices/ricatti_solution_dictonary.csv', 'WriteMode','append');
        save(strcat('riccati_solution_matrices/K_',timestamp,'.mat'),"K");

    end

end