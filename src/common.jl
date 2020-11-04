# Common functions for GBMs.
#
# Author: Jeffrey W. Miller
# Date: 11/4/2020
#
# This file is released under the MIT "Expat" License.

module Common

export SS, logprior, Prior, check_constraints, compute_misc, compute_residuals


using Distributions

using LinearAlgebra: svdvals
import LinearAlgebra
Identity = LinearAlgebra.I

# Sum of squares
SS(A) = sum(A.*A)

logprior(A,lambda,mu=0.0) = sum(logpdf.(Normal(mu,1/lambda),A))

# Prior structure
#    lambda_* = precision of the Normal prior on the entries of A, B, C, D, U, V, S, or T.
#    mu_S, mu_T = means of the Normal priors on the entries of S and T, respectively, e.g., S[i] ~ Normal(mu_S, 1/lambda_S).
#    The prior mean for A, B, C, D, U, and V is zero, e.g., A[j,k] ~ N(0, 1/lambda_A).
mutable struct Prior
    lambda_A::Float64
    lambda_B::Float64
    lambda_C::Float64
    lambda_D::Float64
    lambda_U::Float64
    lambda_V::Float64
    lambda_S::Float64
    lambda_T::Float64
    mu_S::Float64
    mu_T::Float64
    Prior() = (p = new();
        p.lambda_A = p.lambda_B = p.lambda_C = p.lambda_D = p.lambda_U = p.lambda_V = 1.0;
        p.lambda_S = p.lambda_T = 1.0;
        p.mu_S = p.mu_T = 0.0;
        return p
    )
end


# Check identifiability constraints and interpretation identities
function check_constraints(Y,X,Z,A,B,C,D,U,V,S,T)
    I,M = size(U)
    @assert(maximum(abs.(X'*B)) < 1e-10)
    @assert(maximum(abs.(Z'*A)) < 1e-10)
    if M > 0
        @assert(maximum(abs.(X'*U)) < 1e-10)
        @assert(maximum(abs.(Z'*V)) < 1e-10)
        @assert(maximum(abs.(U'*U - Identity)) < 1e-8)
        @assert(maximum(abs.(V'*V - Identity)) < 1e-8)
        if M > 1; @assert(maximum(diff(D)) <= 0); end
        # (could check sign constraints here)
    end
    @assert(abs(mean(exp.(S)) .- 1) < 1e-10)
    @assert(abs(mean(exp.(T)) .- 1) < 1e-10)
    @assert(minimum(svdvals(X)) > 0)
    @assert(minimum(svdvals(Z)) > 0)
    logMu = X*A' + B*Z' + X*C*Z' + U*(D.*V')
    @assert(abs.(mean(logMu) - C[1,1]) < 1e-12)
    @assert(maximum(abs.(vec(mean(logMu, dims=2)) - B[:,1] - X*C[:,1])) < 1e-12)
    @assert(maximum(abs.(vec(mean(logMu, dims=1)) - A[:,1] - Z*C[1,:])) < 1e-12)
end


# Compute miscellaneous
function compute_misc(Y,X,Z,A,B,C,D,U,V,S,T,omega; Offset=0.0*Y)
    I,J = size(Y)
    
    # Compute Mu, W, E, and r
    Mu = zeros(I,J)
    W = zeros(I,J)
    E = zeros(I,J)
    logMu = X*A' + B*Z' + X*C*Z' + U*(D.*V') + Offset
    r = exp.(-(S.+T'.+omega))
    compute_MuWE!(Mu,W,E,Y,logMu,r)
    
    # Verify that the constraints are satisfied
    check_constraints(Y,X,Z,A,B,C,D,U,V,S,T)
    
    # Compute proportion of variation explained
    ss = [SS(X*A'), SS(B*Z'), SS(X*C*Z'), SS(U*(D.*V'))]
    
    # Compute pseudo-residuals
    # E_res = E./W  # pseudo-residuals
    # S_res = sqrt.(1.0./W)  # pseudo-scales
    
    return E,W,ss
end


# Compute adjusted GBM residuals
#    rx = indices of X covariates to retain in residuals (do not adjust them out)
#    rz = indices of Z covariates to retain in residuals (do not adjust them out)
#    ru = indices of latent factors to retain in residuals (do not adjust them out)
function compute_residuals(Y,X,Z,A,B,C,D,U,V; rx=[], rz=[], ru=[])
    logMu = X*A' + B*Z' + X*C*Z' + U*(D.*V')
    logMu_retain = X[:,rx]*A[:,rx]' + B[:,rz]*Z[:,rz]' + X[:,rx]*C[rx,rz]*Z[:,rz]' + U[:,ru]*(D[ru].*V[:,ru]')
    Residuals = log.(Y .+ 0.125) .- logMu
    return logMu_retain .+ Residuals
end



end # module Common












