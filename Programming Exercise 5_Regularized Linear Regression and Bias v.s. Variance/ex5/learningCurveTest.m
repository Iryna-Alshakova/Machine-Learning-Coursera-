function [error_test] = ...
    learningCurveTest(Xtest, ytest,theta, lambda)
    
 
n=length(ytest)
featureNormalize(Xtest)
X_poly_test(:,1) = Xtest
for i=2:8
  X_poly_test(:,i) = Xtest.*X_poly_test(:,i-1)
end  
X_poly_test=[ones(size(Xtest),1) X_poly_test]
 temp_test=theta
temp_test(1)=0
    lambda=3
  J=(1/(2*n))*sum((X_poly_test*theta-ytest).^2)+(lambda/(2*n))*sum(temp_test.^2)
grad=(1/n).*(X_poly_test'*(X_poly_test*theta-ytest))+(lambda/n)*temp_test  
    end