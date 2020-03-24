%% CE 295 - Energy Systems and Control
%   HW 3 : Optimal Economic Dispatch in Distribution Feeders with Renewables
%   Oski Bear, SID 18681868
%   Prof. Moura
%   Last updated: February 20, 2018

% BEAR_OSKI_HW3.m

clear; close all;
fs = 15;    % Font Size for plots

%% 13 Node IEEE Test Feeder Parameters

%%% Node (aka Bus) Data
% l_j^P: Active power consumption [MW]
l_P = [   ];

% l_j^Q: Reactive power consumption [MVAr]
l_Q = [   ];

% l_j^S: Apparent power consumption [MVA]
l_S = sqrt(l_P.^2 + l_Q.^2);

% s_j,max: Maximal generating power [MW]
s_max = [   ];

% c_j: Marginal generation cost [USD/MW]
c = [    ];

% V_min, V_max: Minimum and maximum nodal voltages [V]
v_min = 0.95;
v_max = 1.05;

%%% Edge (aka Line) Data
% r_ij: Resistance [p.u.]
r = [
    ];

% x_ij: Reactance [p.u.]
x = [
    ];

% I_max_ij: Maximal line current [p.u.]
I_max = [
    ];

% A_ij: Adjacency matrix; A_ij = 1 if i is parent of j
A = [
    ];


%%% Set Data (add +1 everywhere for Matlab indexing)
% \rho(j): Parent node of node j
rho = [   ]+1;


%% Problem 1

% Plot active and reactive power consumption
figure(1);
bar(   );

%% Problem 2

% Assumptions:
%   - Disregard the entire network diagram
%   - Balance supply and demand, without any network considerations
%   - Goal is to minimize generation costs, given by c^T s

% Solve with CVX
cvx_begin
    variables % declare your optimization variables here
    minimize(   ) % objective function here
    subject to % constraints
    
        % Balance power generation with power consumption

        
        % Loop over each node
        for jj = 1:13
            
            % Non-negative power generation

            
            % Compute apparent power from active & reactive power

            
        end
        
        % Apparent Power Limits

        
cvx_end

% Output Results
fprintf(1,'------------------- PROBLEM 2 --------------------\n');
fprintf(1,'--------------------------------------------------\n');
fprintf(1,'Minimum Generating Cost : %4.2f USD\n',cvx_optval);
fprintf(1,'\n');
fprintf(1,'Node 0 [Grid]  Gen Power : p_0 = %1.3f MW | q_0 = %1.3f MW | s_0 = %1.3f MW\n',p(1),q(1),s(1));
fprintf(1,'Node 3 [Gas]   Gen Power : p_3 = %1.3f MW | q_3 = %1.3f MW | s_3 = %1.3f MW\n',p(4),q(4),s(4));
fprintf(1,'Node 9 [Solar] Gen Power : p_9 = %1.3f MW | q_9 = %1.3f MW | s_9 = %1.3f MW\n',p(10),q(10),s(10));
fprintf(1,'\n');
fprintf(1,'Total active power   : %1.3f MW   consumed | %1.3f MW   generated\n',sum(l_P),sum(p));
fprintf(1,'Total reactive power : %1.3f MVAr consumed | %1.3f MVAr generated\n',sum(l_Q),sum(q));
fprintf(1,'Total apparent power : %1.3f MVA  consumed | %1.3f MVA  generated\n',sum(l_S),sum(s));



%% Problem 3

% Assumptions:
%   - Disregard L_ij, the squared magnitude of complex line current
%   - Disregard nodal voltage equation
%   - Disregard nodal voltage limits
%   - Disregard maximum line current
%   - Goal is to minimize generation costs, given by c^T s

% Solve with CVX
cvx_begin
    variables 
    dual variable 
    minimize(    )
    subject to
    
        % Boundary condition for power line flows
        P( 1 , 1 ) == 0;
        Q( 1 , 1 ) == 0;
        
        % Loop over each node
        for jj = 1:13
            
            % Parent node, i = \rho(j)

            
            % Line Power Flows

            
            % Compute apparent power from active & reactive power

            
        end
        
        % Apparent Power Limits

        
cvx_end

% Output Results
fprintf(1,'------------------- PROBLEM 3 --------------------\n');
fprintf(1,'--------------------------------------------------\n');
fprintf(1,'Minimum Generating Cost : %4.2f USD\n',cvx_optval);
fprintf(1,'\n');
fprintf(1,'Node 0 [Grid]  Gen Power : p_0 = %1.3f MW | q_0 = %1.3f MW | s_0 = %1.3f MW || mu_s0 = %3.0f USD/MW\n',p(1),q(1),s(1),mu_s(1));
fprintf(1,'Node 3 [Gas]   Gen Power : p_3 = %1.3f MW | q_3 = %1.3f MW | s_3 = %1.3f MW || mu_s3 = %3.0f USD/MW\n',p(4),q(4),s(4),mu_s(4));
fprintf(1,'Node 9 [Solar] Gen Power : p_9 = %1.3f MW | q_9 = %1.3f MW | s_9 = %1.3f MW || mu_s9 = %3.0f USD/MW\n',p(10),q(10),s(10),mu_s(10));
fprintf(1,'\n');
fprintf(1,'Total active power   : %1.3f MW   consumed | %1.3f MW   generated\n',sum(l_P),sum(p));
fprintf(1,'Total reactive power : %1.3f MVAr consumed | %1.3f MVAr generated\n',sum(l_Q),sum(q));
fprintf(1,'Total apparent power : %1.3f MVA  consumed | %1.3f MVA  generated\n',sum(l_S),sum(s));

%% Problem 4

% Assumptions:
%   - Add back all previously disregarded terms and constraints
%   - Relax squared line current equation into inequality
%   - Goal is to minimize generation costs, given by c^T s

% Solve with CVX
cvx_begin
    variables 
    dual variables 
    minimize(   )
    subject to
    
        % Boundary condition for power line flows
        P( 1 , 1 ) == 0;
        Q( 1 , 1 ) == 0;
        
        % Boundary condition for squared line current
        L( 1 , 1 ) == 0;
        
        % Fix node 0 voltage to be 1 "per unit" (p.u.)
        V(1) == 1;
        
        % Loop over each node
        for jj = 1:13
            
            % Parent node, i = \rho(j)

            
            % Line Power Flows

            
            % Nodal voltage

            
            % Squared current magnitude on lines

            
            % Compute apparent power from active & reactive power

            
        end
        
        % Squared line current limits

            
        % Nodal voltage limits

        
        % Apparent Power Limits

        
cvx_end

% Output Results
fprintf(1,'------------------- PROBLEM 4 --------------------\n');
fprintf(1,'--------------------------------------------------\n');
fprintf(1,'Minimum Generating Cost : %4.2f USD\n',cvx_optval);
fprintf(1,'\n');
fprintf(1,'Node 0 [Grid]  Gen Power : p_0 = %1.3f MW | q_0 = %1.3f MW | s_0 = %1.3f MW || mu_s0 = %3.0f USD/MW\n',p(1),q(1),s(1),mu_s(1));
fprintf(1,'Node 3 [Gas]   Gen Power : p_3 = %1.3f MW | q_3 = %1.3f MW | s_3 = %1.3f MW || mu_s3 = %3.0f USD/MW\n',p(4),q(4),s(4),mu_s(4));
fprintf(1,'Node 9 [Solar] Gen Power : p_9 = %1.3f MW | q_9 = %1.3f MW | s_9 = %1.3f MW || mu_s9 = %3.0f USD/MW\n',p(10),q(10),s(10),mu_s(10));
fprintf(1,'\n');
fprintf(1,'Total active power   : %1.3f MW   consumed | %1.3f MW   generated\n',sum(l_P),sum(p));
fprintf(1,'Total reactive power : %1.3f MVAr consumed | %1.3f MVAr generated\n',sum(l_Q),sum(q));
fprintf(1,'Total apparent power : %1.3f MVA  consumed | %1.3f MVA  generated\n',sum(l_S),sum(s));
fprintf(1,'\n');
for jj = 1:13
    fprintf(1,'Node %2.0f Voltage : %1.3f p.u.\n',jj,sqrt(V(jj)));
end


%% Problem 5

% Assumptions:
%   - Assume solar generator at node 9 has uncertain power capacity
%   - Goal is to minimize generation costs, given by c^T s, in face of uncertainty


