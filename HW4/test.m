clc
clear

x_1 = [1.8,0.6];

J_h_1 = [1/2*x_1(1), 2*x_1(2)];

lambda_1 = -0.8;

H_L = [1.076, -0.275;
     -0.275, 0.256];

A = zeros(3);
A(1:2,1:2) = H_L;
A(end,1:2) = J_h_1;
A(1:2,end) = J_h_1';

nabla_L = [1+(1/2)*lambda_1*x_1(1);
           2+2*lambda_1*x_1(2)];

h       =  1/4*x_1(1)^2 + x_1(2)^2 - 1;

A\[-nabla_L;-h]