# Estimation algorithm for generalized bilinear model (GBM) with g(E(Y)) = XA' + BZ' + XCZ' + UDV'.
#
# Author: Jeffrey W. Miller
# Date: 11/4/2020
#
# This file is released under the MIT "Expat" License.

module Fit

export fit, fit_leastsquares

using Statistics: mean
using LinearAlgebra: norm, pinv, rank
import LinearAlgebra
Identity = LinearAlgebra.I

include("common.jl"); using .Common
include("outcome_nb.jl"); using .Outcome_NB


# _________________________________________________________________________________________________

# Estimate the parameters (A,B,C,D,U,V,S,T,omega) of a Negative-Binomial generalized bilinear model (NB-GBM):
#    Y_ij ~ NegativeBinomial(mean = exp(theta_ij), dispersion = exp(s_i + t_j + omega))
#    where theta = XA' + BZ' + XCZ' + UDV'
#    subject to the identifiability constraints Z'*A=0, X'*B=0, X'*U=0, Z'*V=0, U'*U=I, V'*V=I, mean(exp.(S))=1, mean(exp.(T))=1,
#    and D is diagonal with D_11 > D_22 > ... > D_MM > 0.  We do not enforce sign identifiability constraints on U and V.
#
# INPUTS (REQUIRED):
#    Y = I-by-J matrix of outcomes, where Y[i,j] = outcome for feature i, sample j.
#    X = I-by-K matrix of feature covariates, where X[i,k] = value of covariate k for feature i.
#    Z = J-by-L matrix of sample covariates, where Z[j,l] = value of covariate l for sample j.
#    M (Int64) = dimension of latent factors (i.e., rank of UDV').
#
#    It is assumed that X[i,1]==1 for all i, Z[j,1]==1 for all j, and X[:,k] and Z[:,l] are zero mean, unit variance for all k>1 and l>1.
#
# INPUTS (OPTIONAL):
#    max_iterations (Int64) = maximum number of iterations to run the algorithm.
#    tolerance (Float64) = stop when the relative change in log-likelihood is less than this value or max_iterations has been reached.
#    max_step (Float64) = maximum step size for each Fisher scoring / Newton-Raphson update.
#    verbose (Bool) = print status of algorithm while running.
#    prior (Prior) = prior parameters.  (Must be a struct of type "Prior" as defined in this file.)
#    init_params (tuple) = tuple of parameters (A,B,C,D,U,V,S,T,omega) to use for initializing the algorithm.
#    Offset = I-by-J matrix of offsets to theta.
#
# OUTPUTS:
#    A = J-by-K matrix of coefficients for feature covariates
#    B = I-by-L matrix of coefficients for sample covariates
#    C = K-by-L matrix of intercepts and interaction coefficients
#    D = M-by-1 vector of latent factor weights
#    U = I-by-M matrix of feature-wise latent factors
#    V = J-by-M matrix of sample-wise latent factors
#    S = I-by-1 matrix of feature-specific log-dispersion offsets
#    T = J-by-1 matrix of sample-specific log-dispersion offsets
#    omega = overall log-dispersion parameter
#
# DIMENSIONS OF INPUTS/OUTPUTS:
#    I = number of features
#    J = number of samples
#    K = number of feature covariates
#    L = number of sample covariates
#    M = number of latent factors
#
function fit(Y,X,Z,M; max_iterations=50, tolerance=1e-6, max_step=5.0, verbose=true, prior=Prior(), init_params=(), Offset=0.0*Y, update_S=true, update_T=true)
    # Define dimensions and parameters
    I,K = size(X)
    J,L = size(Z)
    A = zeros(J,K)
    B = zeros(I,L)
    C = zeros(K,L)
    D = zeros(M)
    U = zeros(I,M)
    V = zeros(J,M)
    S = zeros(I,1)
    T = zeros(J,1)
    omega = 0.0
    
    # Define auxiliary values
    Mu = zeros(I,J)
    W = zeros(I,J)
    E = zeros(I,J)
    logp_old = Inf
    b_S = max_step*ones(I)
    b_T = max_step*ones(J)
    p = prior
    
    # Check dimensions and constraints
    validate_inputs(Y,X,Z,M)
    
    # Precompute pseudoinverses for X and Z
    pinv_Z = pinv(Z)
    pinv_X = pinv(X)
    
    # Initialize
    if isempty(init_params)
        # Initialize A,B,C by minimizing the sum-of-squared residuals
        A,B,C = fit_leastsquares(log.(Y.+0.125)-Offset,X,Z,M; ABC_only=true)  # first we only the A,B,C terms to avoid overfitting in the UDV' term
        
        U,D,V = tsvd(1e-8*randn(I,J); k=M)  # randomly initialize UDV' to something random and negligible (to avoid numerical issues due to zeros)
        Mu = exp.(X*A' + B*Z' + X*C*Z' + U*(D.*V') + Offset)
        r = exp.(-(S .+ T' .+ omega))
        for iter = 1:4
            S,T,omega,r,b_S,b_T = update_logdispersions(S,T,omega,Y,Mu,r,p,max_step,b_S,b_T,true,true,verbose)  # initialize S,T,omega using a few optimization steps
        end
    else
        (A,B,C,D,U,V,S,T,omega) = deepcopy(init_params)
    end
    logMu = X*A' + B*Z' + X*C*Z' + U*(D.*V') + Offset
    Mu = exp.(logMu)
    r = exp.(-(S.+T'.+omega))
    
    # Record-keeping
    logp = zeros(max_iterations)
    
    # Estimate parameters
    for iteration = 1:max_iterations
        # Update A
        compute_MuWE!(Mu,W,E,Y,logMu,r)
        logMu = logMu - X*A' - X*C*Z'
        A,C = update_main_effects(A,C,X,Z,W,E,p.lambda_A,max_step,verbose)
        logMu = logMu + X*A' + X*C*Z'
        
        # Update B
        compute_MuWE!(Mu,W,E,Y,logMu,r)
        logMu = logMu - B*Z' - X*C*Z'
        B,Ct = update_main_effects(B,C',Z,X,W',E',p.lambda_B,max_step,verbose)
        C = Ct'
        logMu = logMu + B*Z' + X*C*Z'
        
        # Update C
        compute_MuWE!(Mu,W,E,Y,logMu,r)
        logMu = logMu - X*C*Z'
        C = update_interactions(C,X,Z,W,E,p.lambda_C,max_step)
        logMu = logMu + X*C*Z'
        
        if M > 0
            # Update V,D
            compute_MuWE!(Mu,W,E,Y,logMu,r)
            B,Ct,V,D,U = update_UD(B,C',V,D,U,Z,X,pinv_Z,pinv_X,W',E',p.lambda_V,max_step,verbose); C = Ct'
            logMu = X*A' + B*Z' + X*C*Z' + U*(D.*V') + Offset
            
            # Update U,D
            compute_MuWE!(Mu,W,E,Y,logMu,r)
            A,C,U,D,V = update_UD(A,C,U,D,V,X,Z,pinv_X,pinv_Z,W,E,p.lambda_U,max_step,verbose)
            logMu = X*A' + B*Z' + X*C*Z' + U*(D.*V') + Offset
            
            # Update D
            compute_MuWE!(Mu,W,E,Y,logMu,r)
            logMu = logMu - U*(D.*V')
            D,U,V = update_D(D,U,V,W,E,p.lambda_D,max_step)
            logMu = logMu + U*(D.*V')
        end
        
        # Update S, T, and omega
        Mu = exp.(logMu)
        S,T,omega,r,b_S,b_T = update_logdispersions(S,T,omega,Y,Mu,r,p,max_step,b_S,b_T,update_S,update_T,verbose)
        
        # Check for obvious issues
        if any(isnan.(A)) || any(isnan.(B)) || any(isnan.(C)) || any(isnan.(D)) || any(isnan.(U)) || any(isnan.(V)) || any(isnan.(S)) || any(isnan.(T))
            @warn("NaN's detected")
        end
        if (rank(U)<M) || (rank(V)<M)
            @warn("Truncated SVD failure leading to insufficient rank.")
        end
        
        # Check for convergence
        logprior_value = (logprior(A,p.lambda_A) + logprior(B,p.lambda_B) + logprior(C,p.lambda_C) + logprior(D,p.lambda_D) + 
            logprior(U,p.lambda_U) + logprior(V,p.lambda_V) +  logprior(S,p.lambda_S,p.mu_S) + logprior(T,p.lambda_T,p.mu_T))
        logp_new = loglikelihood(Y,Mu,r) + logprior_value
        if verbose; println("iteration $iteration: logp = ",logp_new); end
        if (abs(logp_old/logp_new - 1) < tolerance); break; end
        logp_old = logp_new
        logp[iteration] = logp_new
        
        if (iteration==max_iterations); @warn("Maximum number of iterations reached. Possible nonconvergence."); end
    end
    
    # Softmax bias correction for S and T
    min_S,min_T = -4.0,-4.0
    S = min_S .+ log.(exp.(S .- min_S) .+ 1)
    shift = log(mean(exp.(S)))
    S = S .- shift
    omega = omega + shift
    T = min_T .+ log.(exp.(T .- min_T) .+ 1)
    shift = log(mean(exp.(T)))
    T = T .- shift
    omega = omega + shift
    
    # Check constraints
    validate_outputs(Y,X,Z,A,B,C,D,U,V,S,T)
    
    return A,B,C,D,U,V,S,T,omega,logp
end




# _________________________________________________________________________________________________
# Functions for estimation in NB-GBM

# Recover A, B, C, D, U, and V, given logMu, X, Z, and M.
function fit_leastsquares(logMu,X,Z,M; ABC_only=false)
    Q = pinv(X)*logMu
    C = Q*pinv(Z)'
    A = (Q - C*Z')'
    B = logMu*pinv(Z)' - X*C
    if ABC_only; return A,B,C; end
    UDVt = logMu - (X*A' + B*Z' + X*C*Z')
    U,D,V = tsvd(UDVt; k=M)
    return A,B,C,D,U,V
end

# Compute X'*diag(w)*X and store it in XtWX
function compute_XtWX!(XtWX,X,w)
    I,K = size(X)
    for k1 = 1:K, k2 = 1:K
        value = 0.0
        for i = 1:I
            value += X[i,k1]*w[i]*X[i,k2]
        end
        XtWX[k1,k2] = value
    end
end

# Compute (X'*(Wj.*X) + diagm(Lambda)) \ (X'*Ej - Lambda.*Aj)
function weighted_least_squares_step(X,Wj,Ej,Aj,Lambda)
    I,K = size(X)
    XtWX = zeros(K,K)
    compute_XtWX!(XtWX,X,Wj)
    for k = 1:K; XtWX[k,k] += Lambda[k]; end
    XE = zeros(K)
    for k = 1:K
        value = 0.0
        for i = 1:I
            value += X[i,k]*Ej[i]
        end
        XE[k] = value - Lambda[k]*Aj[k]
    end
    return XtWX \ XE
end

# Perform Newton step on A (or B)
function update_main_effects(A,C,X,Z,W,E,lambda,max_step,verbose)
    Lambda = lambda*ones(size(X,2))
    max_step_enforced = false
    for j = 1:size(W,2)
        # delta = (X'*(W[:,j].*X) + diagm(Lambda)) \ (X'*E[:,j] - Lambda.*A[j,:])
        delta = weighted_least_squares_step(X,W[:,j],E[:,j],A[j,:],Lambda)
        A[j,:] = A[j,:] + delta*min(sqrt(size(A,2))*max_step/norm(delta), 1)
        if sqrt(size(A,2))*max_step < norm(delta); max_step_enforced = true; end
    end
    if max_step_enforced && verbose; println("max_step enforced in A[j,:] update for one or more j."); end
    Shift = pinv(Z)*A
    A = A - Z*Shift  # enforce constraint that Z'*A = 0 by projecting onto nullspace of Z'.
    C = C + Shift'  # compensate to preserve likelihood
    return A,C
end

# Perform Newton step on C
function update_interactions(C,X,Z,W,E,lambda_C,max_step)
    L = size(Z,2)
    F_C = hvcat(L, permutedims([X'*((W*(Z[:,l1].*Z[:,l2])).*X) for l1=1:L, l2=1:L])...)
    delta = (F_C + lambda_C*Identity) \ (vec(X'*E*Z) - lambda_C*C[:])
    C[:] = C[:] + delta*min(sqrt(prod(size(C)))*max_step/norm(delta), 1)
    # (no constraints to enforce on C)
    return C
end

# Update U,D
function update_UD(A,C,U,D,V,X,Z,pinv_X,pinv_Z,W,E,lambda_U,max_step,verbose)
    I = size(X,1)
    J = size(Z,1)
    M = length(D)
    G = U.*D'
    Lambda = lambda_U./(D.^2)
    max_step_enforced = false
    for i = 1:I
        delta = weighted_least_squares_step(V,W[i,:],E[i,:],G[i,:],Lambda)
        # delta = (V'*(W[i,:].*V) + diagm(Lambda_U)) \ (V'*E[i,:] - Lambda.*G[i,:])
        G[i,:] = G[i,:] + delta*min(sqrt(M)*max_step/norm(delta), 1)
        if sqrt(M)*max_step < norm(delta); max_step_enforced = true; end
    end
    if max_step_enforced && verbose; println("max_step enforced in G[i,:] update for one or more i."); end
    pinv_X_G = pinv_X*G
    Go = G - X*pinv_X_G
    Ao = A + V*pinv_X_G'
    pinv_Z_Ao = pinv_Z*Ao
    A = Ao - Z*pinv_Z_Ao
    C = C + pinv_Z_Ao'
    U,D,V = tsvd(Go*V'; k=M)
    return A,C,U,D,V
end

# Update D      
function update_D(D,U,V,W,E,lambda_D,max_step)
    M = length(D)
    F_D = [sum((U[:,m1].*U[:,m2]).*W.*(V[:,m1].*V[:,m2])') for m1=1:M, m2=1:M]
    # F_D = compute_F_D(U,V,W)
    grad_D = vec(sum((U'*E).*V',dims=2))
    delta = (F_D + Identity*lambda_D) \ (grad_D - lambda_D*D)
    D = D + delta*min(sqrt(M)*max_step/norm(delta), 1)
    o = sortperm(D; rev=true); D = D[o]; U = U[:,o]; V = V[:,o]
    return D,U,V
end




end # module Fit










