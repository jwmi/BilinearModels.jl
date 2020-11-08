# Functions for simulating data from GBMs
#
# Author: Jeffrey W. Miller
# Date: 11/4/2020
#
# This file is released under the MIT "Expat" License.

module Sim

export simulate_parameters_and_data, permute_to_match_latent_dimensions!

using Distributions
using Statistics: mean
using LinearAlgebra: pinv, qr


# NegativeBinomial distribution with mean=mu and dispersion=1/r.
#   r = inverse dispersion
#   p = r/(mu+r)  # success probability
# Note that Julia uses the Wolfram definition which is that r is the number of successes until stopping.
NegBin(mu,r) = NegativeBinomial(r, r/(mu+r))

# Standardize x to zero mean, unit variance
center_scale(x,dim=1) = (z = x .- mean(x,dims=dim); z./sqrt.(mean(z.^2,dims=dim)))

# Enforce bound on absolute values of x
bound(x,maxabsval) = max(min(x,maxabsval),-maxabsval)


function simulate_covariate_matrix(I,K; maxabsval=100, cdist="Normal")
    if cdist=="Normal"
        cdistn = Normal()
    elseif cdist=="Bernoulli"
        cdistn = Bernoulli()
    elseif cdist=="Gamma"
        cdistn = Gamma(2,1/sqrt(2))
    else
        @error("Unknown value of cdist for simulating data.")
    end
    
    if cdist=="Normal"
        X = randn(I,K)*randn(K,K)
    else
        # The following line is mathematically equivalent to the approach used, and is algorithmically identical to the Normal case due to the use of center_scale:
        # W = randn(I,K); Q = randn(K,K); C = Q'*Q; sigma = sqrt.(diag(C)); X = (W*Q)./sigma'
    
        Q = randn(K,K); C = Q'*Q; sigma = sqrt.(diag(C)); C = (C./sigma)./sigma'; C = (C+C')/2  # generate a random correlation matrix
        X = rand(MvNormal(C),I)'  # feature covariates
        X = quantile.(cdistn,cdf.(Normal(),X))  # copula transformation
    end
    X = bound.(X,maxabsval)  # enforce bound on magnitude
    X[:,1] .= 1.0  # constant for intercepts
    X[:,2:end] = center_scale(X[:,2:end],1)  # standardize the covariates
    return X
end


# Simulate true parameters and data
function simulate_parameters_and_data(I,J,K,L,M; maxabsval=100, omega0=-2.3, odist="NegativeBinomial", cdist="Normal", pdist="Normal")
    if pdist=="Normal"
        pdistn = Normal()
    elseif pdist=="Gamma"
        pdistn = Gamma(2,1/sqrt(2))
    else
        @error("Unknown value of pdist for simulating data.")
    end
    
    # Simulate covariates
    X = simulate_covariate_matrix(I,K; maxabsval=100, cdist=cdist)  # feature covariates
    Z = simulate_covariate_matrix(J,L; maxabsval=100, cdist=cdist)  # sample covariates

    # Simulate parameters
    A0 = rand(pdistn,J,K)/(2*sqrt(K))  # true feature coefficients
    B0 = rand(pdistn,I,L)/(2*sqrt(L))  # true sample coefficients
    C0 = rand(pdistn,K,L)/sqrt(K*L); if K>0; C0[1,1] = C0[1,1] + 3.0; end  # true intercepts and interaction coefficients
    B0 = B0 - X*(pinv(X)*B0)  # enforce constraint that X'*B = 0 by projecting onto nullspace of X'.
    A0 = A0 - Z*(pinv(Z)*A0)  # enforce constraint that Z'*A = 0 by projecting onto nullspace of Z'.
    
    U0 = Matrix(qr(randn(I,M)).Q)  # generate U0 uniformly from the Stiefel manifold
    V0 = Matrix(qr(randn(J,M)).Q)  # generate V0 uniformly from the Stiefel manifold
    if M > 0
        U0 = U0 - X*(pinv(X)*U0)  # enforce constraint that X'*U = 0 by projecting onto nullspace of X'.
        V0 = V0 - Z*(pinv(Z)*V0)  # enforce constraint that Z'*V = 0 by projecting onto nullspace of Z'.
    end
    D0 = (sqrt(I)+sqrt(J))*(collect(0:M-1)/max(M-1,1) .+ 1)  # use evenly spaced D values following the Marchenko-Pastur scaling
    
    # omega0 = true overall log-dispersion
    S0 = randn(I,1); S0 = S0 .- log(mean(exp.(S0)))  # true feature-specific log-dispersion offsets
    T0 = randn(J,1); T0 = T0 .- log(mean(exp.(T0)))  # true sample-specific log-dispersion offsets

    # Simulate data
    Mu0 = exp.(X*A0' + B0*Z' + X*C0*Z' + U0*(D0.*V0'))
    r0 = exp.(-S0.-T0'.-omega0)
    if odist=="NB"
        Y = rand.(NegBin.(Mu0,r0))
    elseif odist=="Poisson"
        Y = rand.(Poisson.(Mu0))
    elseif odist=="Geometric"
        Y = rand.(Geometric.(1.0./(Mu0 .+ 1)))
    elseif odist=="LNP"
        Sigma2 = log.(1.0./r0 .+ 1.0)
        epsilon = rand.(Normal.(-0.5*Sigma2, sqrt.(Sigma2)))
        Y = rand.(Poisson.(Mu0.*exp.(epsilon)))
    else
        @error("Unknown value of odist for simulating data.")
    end
    
    return Y,X,Z,A0,B0,C0,D0,U0,V0,S0,T0,omega0
end



# Match latent dimensions and latent factor signs
function permute_to_match_latent_dimensions!(D,U,V,D0,U0,V0)
    M = length(D)
    D1,U1,V1 = copy(D),copy(U),copy(V)
    mlist0 = collect(1:M)
    mlist1 = collect(1:M)
    for m = 1:M
        Cor = cor(U0[:,mlist0],U1[:,mlist1])
        index = findmax(abs.(Cor))[2]
        i0,i1 = index[1],index[2]
        sgn = sign(Cor[i0,i1])
        m0,m1 = mlist0[i0],mlist1[i1]
        U[:,m0] = U1[:,m1]*sgn
        V[:,m0] = V1[:,m1]*sgn
        D[m0] = D1[m1]
        deleteat!(mlist0,i0)
        deleteat!(mlist1,i1)
    end
end


end # module Sim





















