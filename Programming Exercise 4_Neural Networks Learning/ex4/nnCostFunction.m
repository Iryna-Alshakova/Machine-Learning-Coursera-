function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


y_matrix=eye(num_labels)(y,:);
a1=[ones(m,1) X]; %(5000X401)
z2=Theta1*a1'; %(25X401)*(401X5000)=(25X5000)
 %(5000X25)
a2=[ones(1,size(z2,2)); (1./(1+e.^(-z2)))]; %(26X5000)
z3=Theta2*a2; %(10X26)*(26X5000)=(10X5000)
a3=1./(1+e.^(-z3));  %(10X5000)

J=(1/m).*sum(sum(-y_matrix'.*log(a3)-(1-y_matrix)'.*log(1-a3)));

temp1=Theta1;
temp1(:,1)=0;
temp2=Theta2;
temp2(:,1)=0;
J=J+(lambda/(2*m))*(sum(sum(temp1.^2))+sum(sum(temp2.^2)));
  % Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
%delta3=zeros(size(y_matrix, 2),1);
%for t=1:m
   
 % a1=[1 X(t,:)]';
 %z2=Theta1*a1; %(25X401)*(401X5000)=(25X5000)
 %(5000X25)
 %a2=[ones(1,size(z2,2)); (1./(1+e.^(-z2)))]; %(26X5000)
 %z3=Theta2*a2; %(10X26)*(26X5000)=(10X5000)
%a3=1./(1+e.^(-z3));
 
 %for k=size(y_matrix, 2);
 %endfor
 
 
 delta3=a3-y_matrix';
 delta2=(delta3'*Theta2(:,2:end)).*sigmoidGradient(z2)'; %(25x10)*(10x5000).*(25x5000)
 
Theta1_grad=Theta1_grad+delta2'*a1;%(25x5000)(50000X401)
 Theta2_grad=Theta2_grad+delta3*a2';
%endfor


% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Theta1_grad=(1/m).*Theta1_grad+(lambda/m)*temp1;
 Theta2_grad=(1/m).*Theta2_grad+(lambda/m)*temp2;
 
 grad=[Theta1_grad(:);Theta2_grad(:)];





















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
