clear
set(0,'defaulttextinterpreter','latex')
set(0,'DefaultTextFontname', 'CMU Serif')
set(0,'DefaultAxesFontName', 'CMU Serif')

load('A5.mat');
load('b5.mat');
A = A5;
b = b5;

dimens = size(A);

%l_1
A_ = [-A,eye(dimens(1));A,eye(dimens(1))];
b_ = [-b;b];
c_ = [zeros(dimens(2),1);ones(dimens(1),1)];

disp("l_1 problem")
tic
x_1 = linprog(c_, -A_, -b_);
time = toc;
disp(["time: " num2str(time)])
disp(["l1 norm:"  num2str(sum(abs(A*x_1(1:dimens(2))-b)))])

%l_2
disp("l_2 problem")
tic
x_2 = linsolve(A,b);
time = toc;
disp(["time: " num2str(time)])
disp(["l2 norm:"  num2str(sqrt(sum(power((A*x_2-b),2))))])

%l_inf
disp("l_inf problem")
A_ = [-A,ones(dimens(1),1);A,ones(dimens(1),1)];
b_ = [-b;b];
c_ = [zeros(dimens(2),1);1];

tic
x_inf = linprog(c_, -A_, -b_);
time = toc;
disp(["time: " num2str(time)])
disp(["linf norm:"  num2str(max(abs(A*x_inf(1:dimens(2))-b)))])

%histogram of results
figure
subplot(3,1,1)
histogram(A*x_1(1:dimens(2))-b)
title("$l_1$ residuals")
subplot(3,1,2)
histogram(A*x_2-b)
title("$l_2$ residuals")
subplot(3,1,3)
histogram(A*x_inf(1:dimens(2))-b)
title("$l_{\infty}$ residuals");