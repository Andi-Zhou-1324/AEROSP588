clc
clear
close all
%%
syms x1 x2
f = x1^4 + 3*x1^3 + 3*x2^2 - 6*x1*x2 - 2*x2;
df_dx1 = diff(f, x1);
df_dx2 = diff(f, x2);
equation1 = df_dx1 == 0;
equation2 = df_dx2 == 0;
solutions = solve([equation1, equation2], [x1, x2]);
solutions.x1 = double(solutions.x1);
solutions.x2 = double(solutions.x2);

solution_array = [solutions.x1,solutions.x2];

subbed_answer = [double(subs(f,[x1,x2],[solution_array(1,:)]));
                 double(subs(f,[x1,x2],[solution_array(2,:)]));
                 double(subs(f,[x1,x2],[solution_array(3,:)]))];