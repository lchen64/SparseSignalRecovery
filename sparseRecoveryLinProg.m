function sparseRecoveryLinProg

N = 256;
S = 5;
M_list = 10:1:50;
epsilon_error = 1e-6;
prob_recovery=zeros(size(M_list)); m=1;
options = optimoptions('linprog','Algorithm','dual-simplex'); 
options.Display = 'off'; 

for M = M_list
   
    success_counter=0;
    for i=1:100
        x = zeros(N,1);
        q = randperm(N);
        x(q(1:S))=randn(S,1);
        
        A = rand(M,N);
        A = orth(A')';
        y = A*x;
        
        %Sparse Recovery via Linear-Programming
        A_tild = [A,-A];
        x_tild =  linprog(ones(2*size(x,1),1),-eye(2*size(x,1)),zeros(2*size(x,1),1),A_tild,y,[],[],options);
        x_hat = x_tild(1:N)-x_tild(N+1:2*N);
        
        error = norm(x_hat-x,2);
        if error<=epsilon_error
            success_counter = success_counter + 1;
        end
    end
    prob_recovery(m)=success_counter/100; m=m+1;
end

plot(M_list,squeeze(prob_recovery(:)),'*');
ylim([0 1])
xlabel('Number of Measurements');
ylabel('Probability of Successful Recovery of Signal');    
title('Phase Transition Plot');
set(gca, 'fontsize', 12);
saveas(gcf,'LinPPhaseTransitionPlot.pdf')

end