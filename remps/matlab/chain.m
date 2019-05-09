clear all
close all
clc

par = 0.5;

low = 2;
high1 = 10;
high2 = 8;

R = [[low, 0]; 
     [low, 0];
     [low, high1];
     [low, high2]];

     
fP = @(omega) [[omega, 1-omega];
                 [1-omega*par, omega*par];
                 [omega, 1-omega];
                 [1-omega*par, omega*par]];

fpi = @(theta) [[theta, 1-theta, 0, 0];      
                [0, 0, theta, 1-theta]];

fpi2 = @(theta) [[theta, 1-theta]; 
                 [theta, 1-theta]];
  
n_states = 2;
n_actions = 2;
%% Plot of the avg reward

n = 20;
T = linspace(0, 1, n);
O = linspace(0, 1, n);
J = zeros(n);

for i = 1:n
    for j = 1:n
        omega = O(i);
        theta = T(j);
        
        P = fP(omega);
        pi = fpi(theta);
        %The stationary distribution is the eigenvector associated to the
        %eigenvalue 1
        [V, D] = eig((pi * P)');
        [~, index] = max(diag(D));
        mu = V(:, index)/ sum(V(:, index));
        mu_pi = pi' * mu;
        p = P .* repmat(mu_pi, 1, n_states);
        J(i,j) = sum(sum(p .* R));
    end
end

[T, O] = meshgrid(T, O);

%Contour plot
f = figure();
[C, h] = contourf(T, O, J, 100);
set(h,'LineColor','none');
colorbar;
title('Average return');
xlabel('\theta');
ylabel('\omega');

%Surface
figure();
surf(T, O, J);
colorbar;
title('Average return');
xlabel('\theta');
ylabel('\omega');

%% REPS

theta = 0.2; 
omega = 0.8;

%One-hot encoding features
phi = eye(n_states);

epsilon = 2;
phi_s = repelem(phi, n_actions*n_states, 1);
phi_sprime = repmat(phi, n_actions*n_states, 1);
options = optimoptions('fmincon', 'Display','off');

T = [];
O = [];

%\eta must be non-negative!
lb = [0, -inf, -inf];
ub = [inf, inf, inf];
for i = 1:20
    
    T = [T, theta];
    O = [O, omega];
    
    P = fP(omega);
    pi = fpi(theta);
    [V, D] = eig((pi * P)');
    [~, index] = max(diag(D));
    mu = V(:, index)/ sum(V(:, index));
    mu_pi = pi' * mu;
    p = P .* repmat(mu_pi, 1, n_states);
    J = sum(sum(p .* R));
    
    fprintf('Before \t theta=%.3g \t omega=%.3g \t J=%.3g\n', theta, omega, J);
    
    pt = p';
    Rt = R';
    g = @(x) 1/x(1) * log(pt(:)' * exp(epsilon + x(1) * (Rt(:) + (phi_sprime - phi_s) * x(2:end)')));

    [x, fval, exitflag, output] = fmincon(g, [1., 0., 0.], [], [], [], [], lb, ub, [], options);
    eta = 1/x(1);
    xi = x(2:end);

    delta = Rt(:) + (phi_sprime - phi_s) * x(2:end)';
    new_p = pt(:) .* exp(1/eta * delta);
    new_p = new_p / sum(new_p);
    new_p = reshape(new_p,  n_states, n_states*n_actions)';
    new_J = sum(sum(new_p .* R));
    
    temp = reshape(sum(new_p, 2), n_states, n_actions)';
    new_pi2 = temp ./ repmat(sum(temp, 2),1, n_actions);
    
    new_P = new_p ./ repmat(sum(new_p, 2), 1, n_states);
    
    p_s = zeros(n_states, 1);
    new_pi = zeros(n_states, n_states*n_actions);
    for s = 1:n_states
        p_s(s) = sum(sum(p((s-1)*n_actions+1:s*n_actions, :)));
        new_pi(s, (s-1)*n_actions+1:s*n_actions) = new_pi2(s, :);
    end
    
    %{
    %Likelihood policy and model separated
    p_sa = repmat(p_s, 1, n_actions) .* new_pi2;
    ll_pi = @(x) -sum(sum(p_sa .* new_pi2 .* log(fpi2(x))));
    ll_P = @(x) -sum(sum(new_p .* log(fP(x))));
    [theta, fval, exitflag, output] = fmincon(ll_pi, theta, [], [], [], [], 0, 1, [], options);
    [omega, fval, exitflag, output] = fmincon(ll_P, omega, [], [], [], [], 0, 1, [], options);
    
    
    %Likelihood policy and model joint (state kernel)
    new_Ppi = new_pi * new_P;
    ll = @(x) -sum(sum(repmat(p_s, 1, n_actions) .* new_Ppi .* log(fpi(x(2)) * fP(x(1)))));
    [x, fval, exitflag, output] = fmincon(ll, [omega, theta], [], [], [], [], [0,0], [1,1], [], options);
    omega = x(1);
    theta = x(2);
    %}
    
    
    %Likelihood on stationary distribution
    % mu = mu pi P
    new_Ppi = new_pi * new_P;
    [V, D] = eig((pi * P)');
    [~, index] = max(diag(D));
    new_mu = V(:, index)/ sum(V(:, index));
    %ceq = @(x) [0, norm(x(3:end) - x(3:end) * fpi(x(2)) * fP(x(1)))];
    ll = @(x) -sum(sum(new_p .* log(fP(x(1)) .* repmat(fpi(x(2))' * x(3:end)', 1, n_states)))); 
    x = fmincon(ll, [omega, theta, new_mu'], [], [], [0, 0, ones(1, n_states)], 1, [0,0, zeros(1, n_states)], [1,1, ones(1, n_states)], [], options);
    omega = x(1);
    theta = x(2);
    
    
    fprintf('After \t theta=%.3g \t omega=%.3g \t J=%.3g\n', theta, omega, new_J);
    
    
end

figure;
plot(T, O, '-or');
xlim([0, 1]);
ylim([0, 1]);
title('Updates');
xlabel('\theta');
ylabel('\omega');

%% Gradient