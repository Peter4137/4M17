%% Section 1

%% Initialisation
clear
n = zeros(5,1); % dimension n of each problem
timings = zeros(5,3); % running time for each problem for each algorithm
results = zeros(5,3); % evaluation of the objective function for each algorithm
L1_iterations=zeros(5,1);

%% BATCH PROCESSING ALL PROBLEMS

for problem_number=1:2 
    %% Data loading
    
    if problem_number==1
        load('A1.mat')
        load('b1.mat')
        A=A1;
        b=b1;
    end
    if problem_number==2
        load('A2.mat')
        load('b2.mat')
        A=A2;
        b=b2;
    end
    if problem_number==3
        load('A3.mat')
        load('b3.mat')
        A=A3;
        b=b3;
    end
    if problem_number==4
        load('A4.mat')
        load('b4.mat')
        A=A4;
        b=b4;
    end
    if problem_number==5
        load('A5.mat')
        load('b5.mat') 
        A=A5;
        b=b5;
    end
    dim = size(A); % computing the size of the matrix A
    
    %% L1-norm problem definition
    % Here we define the matrices A~, b~ that we will use to solve the l1
    % norm problem. 
   
    I=eye(dim(1)); %identity matrix
    A_1norm=[-A I; A I];
    b_1norm=[-b; b];
    c_1norm=[zeros(dim(2),1);ones(dim(1),1)];

    %% Linf-norm problem definition
    % Here we define the matrices A~, b~ that we will use to solve the linf
    % norm problem.
    
    vec_of_ones=ones(dim(1),1);
    A_inf_norm=[-A vec_of_ones; A vec_of_ones];
    b_inf_norm=[-b; b];
    c_inf_norm=[zeros(dim(2),1); 1];


    %% Solving the L1 minimization problem with linprog(f,A,b)
    % In this case f = c_1norm, A=-A_1norm, b=-b_1norm
    
    tic
    [x_norm1,fval,exitflag,output]=linprog(c_1norm, -A_1norm, -b_1norm);
    time_1=toc;
    value_1= sum(abs(A*x_norm1(1:dim(2))-b)); %1-norm computation
    no_it_1= output.iterations;

    %% Solving the L2 minimization problem with linsolve(A,b)
    
    tic
    x_norm2=linsolve(A,b);
    time_2=toc;
    value_2= sqrt(sum(power((A*x_norm2-b),2))); %2-norm computation

    %% Solving the Linf minimization problem with linprog(f,A,b)
    % In this case f = c_inf_norm, A=-A_inf_norm, b=-b_inf_norm
    
    tic
    x_inf=linprog(c_inf_norm, -A_inf_norm, -b_inf_norm);
    time_inf=toc;
    value_inf= max(abs(A*x_inf(1:dim(2))-b)); 

    %% Printing optimisation details
    
    disp(['P' num2str(problem_number) ':'])
    disp('L1 norm optimisation')
    disp(['Objective function value: ' num2str(value_1)])
    disp(['Time required: ' num2str(time_1)])

    disp('L2 norm optimisation')
    disp(['Objective function value: ' num2str(value_2)])
    disp(['Time required: ' num2str(time_2)])

    disp('Linf norm optimisation')
    disp(['Objective function value: ' num2str(value_inf)])
    disp(['Time required: ' num2str(time_inf)])
    
    %% Saving
    
    n(problem_number)=dim(2);
    timings(problem_number, :) = [time_1, time_2, time_inf];
    results(problem_number, :) = [value_1, value_2, value_inf];
    L1_iterations(problem_number) = [no_it_1];
    
end

%% Plotting the histogram of the residuals of the norm approximation
% problem for the 5th pair of data (A5,b5) for each of the norm types.

figure
subplot(3,1,1)
hist(A*x_norm1(1:dim(2))-b)
subplot(3,1,2)
hist(A*x_norm2-b)
subplot(3,1,3)
hist(A*x_inf(1:dim(2))-b)
title('Histograms of the residuals of the norm approximation');

%% Plotting the running time of the algorithms against the dimension n of x

figure
hold on
for i=1:3
    plot(n, timings(:,i), 'LineWidth',2)
end
title('Running time against dimensionality');