# Common functions for GBMs.
#
# Copyright (c) 2020: Jeffrey W. Miller.
# This file is released under the MIT "Expat" License.

module Common

export SS, logprior, Prior, validate_inputs, validate_outputs, sum_of_squares, residuals, tsvd

using Distributions
import PROPACK
import LinearAlgebra
Identity = LinearAlgebra.I

include("outcome_nb.jl"); using .Outcome_NB


# Wrapper for tsvd function in PROPACK.  Re-tries multiple times if errors occur, and suppresses standard output.
function tsvd(A; k=1, trynum=1, maxtries=5)
    if trynum == maxtries
        U,D,V,bnd,nprod,ntprod = PROPACK.tsvd(A; k=k)
    else
        try
            U,D,V,bnd,nprod,ntprod = PROPACK.tsvd(A; k=k)
            return U,D,V
        catch
        end
        U,D,V = tsvd(A; k=k, trynum=trynum+1, maxtries=maxtries)
    end
    return U,D,V
end

minsvdval(A) = PROPACK.tsvdvals(A; k=size(A,2))[1][end]


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


# Check constraints on Y, X, Z, and M
function validate_inputs(Y,X,Z,M)
    @assert(isa(Y,Matrix) && isa(X,Matrix) && isa(Z,Matrix), "Y, X, and Z must be matrices.")
    I,K = size(X)
    J,L = size(Z)
    @assert(I == size(Y,1), "Y and X must have the same number of rows.") 
    @assert(J == size(Y,2), "Y and Z' must have the same number of columns.") 
    @assert(K > 0, "X must have at least one column.")
    @assert(L > 0, "Z must have at least one column.")
    @assert((M>=0) && isa(M,Int), "M must be a nonnegative integer.")
    @assert(M < min(I,J))
    @assert(all(X[:,1].==1), "The first column of X must be all ones.")
    @assert(all(Z[:,1].==1), "The first column of Z must be all ones.")
    if K>1; @assert(maximum(abs.(sum(X[:,2:end],dims=1))) < 1e-10, "Except for column 1, the columns of X must sum to zero."); end
    if L>1; @assert(maximum(abs.(sum(Z[:,2:end],dims=1))) < 1e-10, "Except for column 1, the columns of Z must sum to zero."); end
    @assert(minsvdval(X) > 0, "X must be full rank, that is, X'*X must be invertible.")
    @assert(minsvdval(Z) > 0, "Z must be full rank, that is, Z'*Z must be invertible.")
end


# Check identifiability constraints and interpretation identities
function validate_outputs(Y,X,Z,A,B,C,D,U,V,S,T)
    I,M = size(U)
    @assert(maximum(abs.(X'*B)) < 1e-10, "X'*B is nonzero.")
    @assert(maximum(abs.(Z'*A)) < 1e-10, "Z'*A is nonzero.")
    if M > 0
        @assert(maximum(abs.(X'*U)) < 1e-10, "X'*U is nonzero.")
        @assert(maximum(abs.(Z'*V)) < 1e-10, "Z'*V is nonzero.")
        @assert(maximum(abs.(U'*U - Identity)) < 1e-8, "U is not orthonormal, that is, U'*U-Identity is nonzero.")
        @assert(maximum(abs.(V'*V - Identity)) < 1e-8, "V is not orthonormal, that is, V'*V-Identity is nonzero.")
        if M > 1; @assert(maximum(diff(D)) <= 0, "The entries of D are not in decreasing order."); end
        # (could check sign constraints here)
    end
    @assert(abs(mean(exp.(S)) .- 1) < 1e-10, "The identifiability constraint on S is not satisfied.")
    @assert(abs(mean(exp.(T)) .- 1) < 1e-10, "The identifiability constraint on T is not satisfied.")
    logMu = X*A' + B*Z' + X*C*Z' + U*(D.*V')
    @assert(abs.(mean(logMu) - C[1,1]) < 1e-12)
    @assert(maximum(abs.(vec(mean(logMu, dims=2)) - B[:,1] - X*C[:,1])) < 1e-12, "One or more interpretability identities are not satisfied.")
    @assert(maximum(abs.(vec(mean(logMu, dims=1)) - A[:,1] - Z*C[1,:])) < 1e-12, "One or more interpretability identities are not satisfied.")
end


# Compute sum-of-squares of each model component
function sum_of_squares(X,Z,A,B,C,D,U,V)
    return [SS(X*A'), SS(B*Z'), SS(X*C*Z'), SS(U*(D.*V'))]
end


# Compute residuals (or adjusted residuals)
#    rx = indices of X covariates to retain in residuals (do not adjust them out)
#    rz = indices of Z covariates to retain in residuals (do not adjust them out)
#    ru = indices of latent factors to retain in residuals (do not adjust them out)
function residuals(Y,X,Z,A,B,C,D,U,V,S,T,omega; rx=[], rz=[], ru=[])
    logMu = X*A' + B*Z' + X*C*Z' + U*(D.*V')
    logMu_retain = X[:,rx]*A[:,rx]' + B[:,rz]*Z[:,rz]' + X[:,rx]*C[rx,rz]*Z[:,rz]' + U[:,ru]*(D[ru].*V[:,ru]')
    Eps = log.(Y .+ 0.125) .- logMu
    Eps = logMu_retain .+ Eps
    r = exp.(-(S .+ T' .+ omega))
    Mu,W,E = compute_MuWE(Y,logMu,r)
    Sigma_Eps = 1.0./sqrt.(W)
    return Eps,Sigma_Eps
end



end # module Common












