% yin-yang demo for the generic backpropagation

function genbackpropYY2

% number of points sampled randomly
%N = 150;
%[x, d] = yinyang(N);

% shifting d to {-1, 1}
%d = d*2-1; 

% injecting bias
%x = [ones(N,1) x];

% defining activation function & its derivative
theta = @tansig;
dtheta = @(x) 1-theta(x).*theta(x);

% number of neurons in the consecutive layers
layers = [35 24 12 10];

d = dlmread('d.txt');
d_teszt = dlmread('dTest.txt');
x = dlmread('x.txt');
x_teszt = dlmread('xTest.txt')

W = genbackprop(x, d, layers, theta, dtheta, 500, 0.01, 0.003);

% simple visualization
[X, Y] = meshgrid(linspace(-1,1,40), linspace(-1,1,40));
Z = zeros(size(X));
for i = 1:size(Z,1)
  for j = 1:size(Z,2)
    in = [1 X(i,j) Y(i,j)]';
    in
    W
    out = forwardprop(in, W, theta);
    Z(i,j) = out{end};
  end
end

surf(X,Y,Z)
