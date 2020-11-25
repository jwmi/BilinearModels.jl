# Functions for negative binomial outcome model in GBM.
#
# Copyright (c) 2020: Jeffrey W. Miller.
# This file is released under the MIT "Expat" License.

module Outcome_NB

export compute_MuWE!, compute_MuWE, loglikelihood, compute_logdispersion_derivatives, update_logdispersions

using Statistics: mean
using SpecialFunctions: logabsgamma, digamma, trigamma
lgamma_(x) = logabsgamma(x)[1]


# Compute Mu, W, and E
function compute_MuWE!(Mu,W,E,Y,logMu,r)
    Mu .= exp.(logMu)
    W .= (r.*Mu)./(r.+Mu)
    E .= (Y - Mu).*(W./Mu)
end


function compute_MuWE(Y,logMu,r)
    I,J = size(Y)
    Mu = zeros(I,J)
    W = zeros(I,J)
    E = zeros(I,J)
    compute_MuWE!(Mu,W,E,Y,logMu,r)
    return Mu,W,E
end


# log-likelihood of the Negative-Binomial distribution with mean Mu and inverse-dispersion r, on data Y.
# loglikelihood(Y,Mu,r) = sum(lgamma_.(r.+Y) .- lgamma_.(Y.+1) .- lgamma_.(r) .+ r.*log.(r) .+ Y.*log.(Mu) .- (r.+Y).*log.(r.+Mu))
# loglikelihood(Y,Mu,r) = sum(lgamma_.(r.+Y) .- lgamma_.(r) .- lgamma_.(Y.+1) .- Y.*log1p.(r./Mu) .- r.*log1p.(Mu./r))
function loglikelihood(Y,Mu,r)
    I,J = size(Y)
    ll = 0.0
    for i = 1:I
        for j = 1:J
            ll += lgamma_(r[i,j]+Y[i,j]) - lgamma_(r[i,j]) - lgamma_(Y[i,j]+1) - Y[i,j]*log1p(r[i,j]/Mu[i,j]) - r[i,j]*log1p(Mu[i,j]/r[i,j])
        end
    end
    return ll
end


digamma_delta(y,r) = ((r > 10^8) ? log1p(y/r) : digamma(y+r)-digamma(r))
trigamma_delta(y,r) = ((r > 10^8) ? -(y/r)/(y+r) : trigamma(y+r)-trigamma(r))

function compute_logdispersion_derivatives(Y,Mu,r)
    I,J = size(Y)
    D1 = zeros(I,J)
    D2 = zeros(I,J)
    for i = 1:I, j = 1:J
        D1[i,j] = -r[i,j]*(digamma_delta(Y[i,j],r[i,j]) - log1p(Mu[i,j]/r[i,j]) - (Y[i,j]-Mu[i,j])/(r[i,j]+Mu[i,j]))
        
        if isinf(Mu[i,j]^2)
            D2[i,j] = -D1[i,j] + (r[i,j]^2)*trigamma_delta(Y[i,j],r[i,j]) + r[i,j]
        else
            D2[i,j] = -D1[i,j] + (r[i,j]^2)*trigamma_delta(Y[i,j],r[i,j]) + (Y[i,j] + Mu[i,j]^2/r[i,j]) / (1.0 + Mu[i,j]/r[i,j])^2
        end
    end
    return D1,D2
end



function update_logdispersions(S,T,omega,Y,Mu,r,p,max_step,b_S,b_T,update_S,update_T,verbose)
    I,J = size(Y)
    
    if update_S
    # Update S
    D1,D2 = compute_logdispersion_derivatives(Y,Mu,r)
    g = vec(sum(D1; dims=2)) .- p.lambda_S*(S .- p.mu_S)
    h = vec(sum(D2; dims=2)) .- p.lambda_S
    delta = (g./h).*(h.<0) .+ (-g).*(h.>=0)
    S = S - delta.*min.(b_S./abs.(delta), 1)
    if any(abs.(delta) .> b_S) && verbose; println(" max step size enforced in S[i] update for one or more i."); end
    b_S = [(abs(delta[i]) > b_S[i] ?  b_S[i]/2 : max_step) for i = 1:I]
    shift = log(mean(exp.(S)))
    S = S .- shift  # Project onto constrained space
    omega = omega + shift  # Compensate to preserve likelihood
    r = exp.(-(S.+T'.+omega))
    end
    
    if update_T    
    # Update T
    D1,D2 = compute_logdispersion_derivatives(Y,Mu,r)
    g = vec(sum(D1; dims=1)) .- p.lambda_T*(T .- p.mu_T)
    h = vec(sum(D2; dims=1)) .- p.lambda_T
    delta = (g./h).*(h.<0) .+ (-g).*(h.>=0)
    T = T - delta.*min.(b_T./abs.(delta), 1)
    if any(abs.(delta) .> b_T) && verbose; println(" max step size enforced in T[j] update for one or more j."); end
    b_T = [(abs(delta[j]) > b_T[j] ?  b_T[j]/2 : max_step) for j = 1:J]
    shift = log(mean(exp.(T)))
    T = T .- shift  # Project onto constrained space
    omega = omega + shift  # Compensate to preserve likelihood
    r = exp.(-(S.+T'.+omega))
    end
     
    return S,T,omega,r,b_S,b_T
end



end # module Outcome_NB















