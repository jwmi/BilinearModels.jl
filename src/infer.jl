# Inference algorithm for generalized bilinear model (GBM) with g(E(Y)) = XA' + BZ' + XCZ' + UDV'.
#
# Copyright (c) 2020: Jeffrey W. Miller.
# This file is released under the MIT "Expat" License.

module Infer

export infer

using LinearAlgebra: diag
import LinearAlgebra
Identity = LinearAlgebra.I
Diagonal = LinearAlgebra.Diagonal

include("common.jl"); using .Common
include("outcome_nb.jl"); using .Outcome_NB

nchoose2(n) = div(n*(n-1),2)
block(i,K) = ((i-1)*K+1):(i*K)


# _________________________________________________________________________________________________

# Compute standard errors for estimates of the parameters of a Negative-Binomial generalized bilinear model (NB-GBM).
#
# INPUTS:
#    Y, X, Z, A, B, C, D, U, V, S, T, and omega (see gbm_estimation for parameter descriptions).
# 
# OUTPUTS:
#    se_A = J-by-K matrix of (approximate) standard errors for A estimates
#    se_B = I-by-L matrix of (approximate) standard errors for B estimates
#    se_C = K-by-L matrix of (approximate) standard errors for C estimates
#    se_U = I-by-M matrix of (approximate) standard errors for U estimates
#    se_V = J-by-M matrix of (approximate) standard errors for V estimates
#    se_S = I-by-1 matrix of (approximate) standard errors for S estimates
#    se_T = J-by-1 matrix of (approximate) standard errors for T estimates
#
function infer(Y,X,Z,A,B,C,D,U,V,S,T,omega; prior=Prior(), Offset=0.0*Y)
    I,K = size(X)
    J,L = size(Z)
    M = length(D)
    
    p = prior
                
    Mu = zeros(I,J)
    W = zeros(I,J)
    E = zeros(I,J)
    logMu = X*A' + B*Z' + X*C*Z' + U*(D.*V') + Offset
    r = exp.(-(S.+T'.+omega))
    compute_MuWE!(Mu,W,E,Y,logMu,r)
    
    # Compute Fisher information matrices
    F_A = [X'*(W[:,j].*X) for j=1:J]
    F_A_inv = [inv(F_A[j] + p.lambda_A*Identity) for j=1:J]
    
    F_B = [Z'*(W[i,:].*Z) for i=1:I]
    F_B_inv = [inv(F_B[i] + p.lambda_B*Identity) for i=1:I]
    
    F_C = hvcat(L, permutedims([X'*((W*(Z[:,l1].*Z[:,l2])).*X) for l1=1:L, l2=1:L])...)
    F_C_inv = inv(F_C + p.lambda_C*Identity)
    
    UD = U.*D'
    VD = V.*D'
    F_U = [VD'*(W[i,:].*VD) for i=1:I]
    F_V = [UD'*(W[:,j].*UD) for j=1:J]
    # F_D = [sum((U[:,m1].*U[:,m2]).*W.*(V[:,m1].*V[:,m2])') for m1=1:M, m2=1:M]
    F_D = compute_F_D(U,V,W)
    F_U_inv = [inv(F_U[i] + p.lambda_U*Identity) for i=1:I]
    F_V_inv = [inv(F_V[j] + p.lambda_V*Identity) for j=1:J]
    F_D_inv = inv(F_D + p.lambda_D*Identity)
    
    if true
        # Compute full matrix for (U,V,Q) only, and propagate uncertainty to A,B,C.
        
        # Uncertainty in U and V
        F_U_by_V = zeros(I*M,J*M)
        for i = 1:I, j = 1:J
            for m1 = 1:M, m2 = 1:M
                F_U_by_V[(i-1)*M+m1, (j-1)*M+m2] = W[i,j]*V[j,m1]*D[m1]*U[i,m2]*D[m2]
            end
        end
        
        Q_U,Q_V = compute_constraint_Jacobian_UV(X,Z,U,V)
        
        # F_UiQ = vcat([F_U_inv[i]*view(Q_U,block(i,M),:) for i=1:I]...)
        F_UiQ = zeros(size(Q_U))
        for i = 1:I
            F_UiQ[block(i,M),:] = F_U_inv[i]*view(Q_U,block(i,M),:)
        end
        F_VUUiQ = F_U_by_V'*F_UiQ
        inv_QFUQ = inv(Q_U'*F_UiQ)
        # F_UiUV = vcat([F_U_inv[i]*view(F_U_by_V,block(i,M),:) for i=1:I]...)
        F_UiUV = zeros(I*M,J*M)
        for i = 1:I
            F_UiUV[block(i,M),:] = F_U_inv[i]*view(F_U_by_V,block(i,M),:)
        end
        F_VUUiUV = F_U_by_V'*F_UiUV  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<  This is the expensive term?  O(I * J^2 * M^3)
        Temp = -F_VUUiUV + F_VUUiQ * inv_QFUQ * F_VUUiQ'
        for j=1:J; Temp[block(j,M),block(j,M)] += F_V[j] + p.lambda_V*Identity; end
        
        C_V = inv([Temp  Q_V ; Q_V' zeros(size(Q_V,2),size(Q_V,2))])[1:J*M,1:J*M]
        
        diag_T = vcat(map(diag,F_U_inv)...) - vec(sum(F_UiQ'.*(inv_QFUQ * F_UiQ'), dims=1))
        F_UVT = F_UiUV' - F_VUUiQ * inv_QFUQ * F_UiQ'
        var_U_approx = diag_T + vec(sum(F_UVT.*(C_V*F_UVT),dims=1))
        var_V_approx = diag(C_V)
        
        # dW_dMu_Mu = Mu.*(r./(r.+Mu)).^2
        # dE_dMu_Mu = -Mu.*r.*(r.+Y)./(r.+Mu).^2
        # grad_U = E*(V.*D')
        # grad_V = E'*(U.*D')
        # var_A_baseline = vcat(map(diag,F_A_inv)...)
        # var_B_baseline = vcat(map(diag,F_B_inv)...)
        # var_V_from_A = compute_var_V_from_A(U.*D',X,var_A_baseline,F_V_inv,grad_V,dW_dMu_Mu,dE_dMu_Mu)
        # var_V_from_B = compute_var_V_from_B(U.*D',Z,var_B_baseline,F_V_inv,grad_V,dW_dMu_Mu,dE_dMu_Mu)
        # var_U_from_B = compute_var_U_from_B(V.*D',Z,var_B_baseline,F_U_inv,grad_U,dW_dMu_Mu,dE_dMu_Mu)
        # var_V_approx += var_V_from_A + var_V_from_B
        # var_U_approx += var_U_from_B
    
        # Propagate uncertainty in U and V through to D
        var_D_approx = diag(F_D_inv)  # TODO: NOT YET IMPLEMENTED
        
        # Propagate uncertainty in U and V through to A and B
        grad_A = E'*X
        grad_B = E*Z
        dW_dMu_Mu = Mu.*(r./(r.+Mu)).^2
        dE_dMu_Mu = -Mu.*r.*(r.+Y)./(r.+Mu).^2
        var_B_from_U = compute_var_B_from_U(Z,V,D,var_U_approx,F_B_inv,grad_B,dW_dMu_Mu,dE_dMu_Mu)
        var_B_from_V = compute_var_B_from_V(Z,U,D,var_V_approx,F_B_inv,grad_B,dW_dMu_Mu,dE_dMu_Mu)
        var_A_from_V = compute_var_A_from_V(X,U,D,var_V_approx,F_A_inv,grad_A,dW_dMu_Mu,dE_dMu_Mu)
        var_A_from_U = compute_var_A_from_U(X,V,D,var_U_approx,F_A_inv,grad_A,dW_dMu_Mu,dE_dMu_Mu)
        var_A_approx = vcat(map(diag,F_A_inv)...) + var_A_from_U + var_A_from_V
        var_B_approx = vcat(map(diag,F_B_inv)...) + var_B_from_U + var_B_from_V
        
        # Propagate uncertainty in A and B through to C
        grad_C = vec(X'*E*Z)
        var_C_from_A = compute_var_C_from_A(X,Z,F_A_inv,F_C_inv,grad_C,dW_dMu_Mu,dE_dMu_Mu)
        var_C_from_B = compute_var_C_from_B(X,Z,F_B_inv,F_C_inv,grad_C,dW_dMu_Mu,dE_dMu_Mu)
        var_C_approx = diag(F_C_inv) + var_C_from_A + var_C_from_B
        
    else
    
        # For testing purposes, compute the full Fisher information matrix
        F_A_full = zeros(J*K,J*K); for j = 1:J; F_A_full[block(j,K),block(j,K)] = F_A[j]; end
        F_B_full = zeros(I*L,I*L); for i = 1:I; F_B_full[block(i,L),block(i,L)] = F_B[i]; end
        F_C_full = F_C
        
        F_U_full = zeros(I*M,I*M); for i = 1:I; F_U_full[block(i,M),block(i,M)] = F_U[i]; end
        F_V_full = zeros(J*M,J*M); for j = 1:J; F_V_full[block(j,M),block(j,M)] = F_V[j]; end
        F_D_full = F_D

        F_A_by_B = hvcat(I, permutedims([W[i,j]*X[i,:]*Z[j,:]' for j=1:J, i=1:I])...)
        F_A_by_C = hvcat(L, permutedims([Z[j,l]*X'*(W[:,j].*X) for j=1:J, l=1:L])...)
        F_A_by_U = hvcat(I, permutedims([W[i,j]*X[i,:]*(V[j,:].*D)' for j=1:J, i=1:I])...)
        F_A_by_V = hvcat(J, permutedims([X'*(W[:,j1].*U.*D')*(j1==j2) for j1=1:J, j2=1:J])...)
        F_A_by_D = vcat([X'*(W[:,j].*U.*V[j,:]') for j=1:J]...)
        
        F_B_by_C = hvcat(L, permutedims([(Z'*(W[i,:].*Z[:,l]))*X[i,:]' for i=1:I, l=1:L])...)
        F_B_by_U = hvcat(I, permutedims([Z'*(W[i1,:].*V.*D')*(i1==i2)  for i1=1:I, i2=1:I])...)
        F_B_by_V = hvcat(J, permutedims([W[i,j]*Z[j,:]*(U[i,:].*D)'   for i=1:I, j=1:J])...)
        F_B_by_D = vcat([Z'*(W[i,:].*V.*U[i,:]') for i=1:I]...)
        
        F_C_by_U = hvcat(I, permutedims([X[i,:]*((W[i,:].*Z[:,l])'*(V.*D'))  for l=1:L, i=1:I])...)
        F_C_by_V = hvcat(J, permutedims([X'*(W[:,j].*Z[j,l].*U.*D') for l=1:L, j=1:J])...)
        F_C_by_D = hvcat(M, permutedims([(X.*U[:,m])'*W*(Z[:,l].*V[:,m]) for l=1:L, m=1:M])...)
        
        F_U_by_V = hvcat(J, permutedims([W[i,j]*(V[j,:].*D)*(U[i,:].*D)' for i=1:I, j=1:J])...)
        F_U_by_D = vcat([(V.*D')'*(W[i,:].*V.*U[i,:]') for i=1:I]...)
        F_V_by_D = vcat([(U.*D')'*(W[:,j].*U.*V[j,:]') for j=1:J]...)
        
        Q,Q_A,Q_B,Q_U,Q_V = compute_constraint_Jacobian(X,Z,A,B,U,V)
        
        F_ABC = [F_A_full  F_A_by_B  F_A_by_C
                 F_A_by_B' F_B_full  F_B_by_C
                 F_A_by_C' F_B_by_C' F_C_full]
        F_UVD = [F_U_full  F_U_by_V  F_U_by_D
                 F_U_by_V' F_V_full  F_V_by_D
                 F_U_by_D' F_V_by_D' F_D_full]
        F_ABC_by_UVD = [F_A_by_U F_A_by_V F_A_by_D
                        F_B_by_U F_B_by_V F_B_by_D
                        F_C_by_U F_C_by_V F_C_by_D]
        F_ABCUDV = [F_ABC         F_ABC_by_UVD 
                    F_ABC_by_UVD' F_UVD       ]
        N = size(Q,2)
        F_full = [F_ABCUDV  Q
                  Q'        zeros(N,N)]
        F_full_inv = inv(F_full)  # todo: Should we be using the priors here?
        
        S_A = 1:J*K
        S_B = (1:I*L).+maximum(S_A)
        S_C = (1:K*L).+maximum(S_B)
        S_U = (1:I*M).+maximum(S_C)
        S_V = (1:J*M).+maximum(S_U)
        S_D = (1:M).+maximum(S_V)
        S_Q = (1:N).+maximum(S_D)
        
        var_A_approx = diag(F_full_inv)[S_A]
        var_B_approx = diag(F_full_inv)[S_B]
        var_C_approx = diag(F_full_inv)[S_C]
        var_U_approx = diag(F_full_inv)[S_U]
        var_V_approx = diag(F_full_inv)[S_V]
        var_D_approx = diag(F_full_inv)[S_D]
        
    end
    
    se_A = sqrt.(max.(eps(0.0),reshape(var_A_approx, K,J))')
    se_B = sqrt.(max.(eps(0.0),reshape(var_B_approx, L,I))')
    se_C = sqrt.(max.(eps(0.0),reshape(var_C_approx, K,L)))
    se_U = sqrt.(max.(eps(0.0),reshape(var_U_approx, M,I))')
    se_V = sqrt.(max.(eps(0.0),reshape(var_V_approx, M,J))')
    se_D = sqrt.(max.(eps(0.0),var_D_approx))
    
    # Delta propagation for S and T
    D1,D2 = compute_logdispersion_derivatives(Y,Mu,r)
    g_S = vec(sum(D1; dims=2)) .- p.lambda_S*(S .- p.mu_S)
    g_T = vec(sum(D1; dims=1)) .- p.lambda_T*(T .- p.mu_T)
    h_S = vec(sum(D2; dims=2)) .- p.lambda_S
    h_T = vec(sum(D2; dims=1)) .- p.lambda_T
    
    # Compute observed information for S and T (use observed information since expected info can't be analytically computed)
    F_S = [(h_S[i]<=0 ? -h_S[i] : NaN) for i = 1:I]
    F_T = [(h_T[j]<=0 ? -h_T[j] : NaN) for j = 1:J]
    
    var_S_from_U,var_S_from_V = compute_var_S_from_UV(Y,Mu,W,E,r,U,V,D,g_S,h_S,se_U,se_V)
    var_S_from_B,var_S_from_A = compute_var_S_from_UV(Y,Mu,W,E,r,X,Z,1,g_S,h_S,se_B,se_A)
    var_T_from_U,var_T_from_V = compute_var_T_from_UV(Y,Mu,W,E,r,U,V,D,g_T,h_T,se_U,se_V)
    var_T_from_B,var_T_from_A = compute_var_T_from_UV(Y,Mu,W,E,r,X,Z,1,g_T,h_T,se_B,se_A)
    
    # Calculate standard errors for S and T
    se_S = sqrt.(1.0./F_S + var_S_from_U + var_S_from_V + var_S_from_A + var_S_from_B)
    se_T = sqrt.(1.0./F_T + var_T_from_U + var_T_from_V + var_T_from_A + var_T_from_B)
    
    # Original approach: Just use F_S and F_T
    # h_S = vec(sum(D2; dims=2)) .- p.lambda_S
    # D1,D2 = compute_logdispersion_derivatives(Y,Mu,r)
    # h_T = vec(sum(D2; dims=1)) .- p.lambda_T
    # F_S = [(h_S[i]<=0 ? -h_S[i] : NaN) for i = 1:I]
    # F_T = [(h_T[j]<=0 ? -h_T[j] : NaN) for j = 1:J]
    # se_S = sqrt.(1.0./F_S)
    # se_T = sqrt.(1.0./F_T)
    
    if any(isnan.(se_A)); @warn("One or more invalid standard errors in se_A."); end
    if any(isnan.(se_B)); @warn("One or more invalid standard errors in se_B."); end
    if any(isnan.(se_C)); @warn("One or more invalid standard errors in se_C."); end
    if any(isnan.(se_D)); @warn("One or more invalid standard errors in se_D."); end
    if any(isnan.(se_U)); @warn("One or more invalid standard errors in se_U."); end
    if any(isnan.(se_V)); @warn("One or more invalid standard errors in se_V."); end
    if any(isnan.(se_S)); @warn("One or more invalid standard errors in se_S."); end
    if any(isnan.(se_T)); @warn("One or more invalid standard errors in se_T."); end
    
    return se_A,se_B,se_C,se_U,se_V,se_S,se_T
end




# __________________________________________________________________________________________________
# Functions for inference in NB-GBM

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

function compute_F_D(U,V,W)
    I,M = size(U)
    J,M = size(V)
    F_D = zeros(M,M)
    for m1 = 1:M, m2 = 1:M
        value = 0.0
        for j = 1:J
            uw = 0.0
            for i = 1:I
                uw += U[i,m1]*U[i,m2]*W[i,j]
            end
            value += uw*V[j,m1]*V[j,m2]
        end
        F_D[m1,m2] = value
    end
    return F_D
end

# Jacobian (transposed) of identifiability constraints on U and V
# (i.e., X'*U=0, Z'*V=0, U'*U=I, and V'*V=I)
function compute_constraint_Jacobian_UV(X,Z,U,V)
    I,M = size(U)
    J,M = size(V)
    
    Q_U_x = kron(X, Diagonal(ones(M)))  # Constraint: vec(U'*X)=0
    Q_V_z = kron(Z, Diagonal(ones(M)))  # Constraint: vec(V'*Z)=0
    
    # Orthogonality contraints on U and V
    Nc = nchoose2(M+1)
    Q_U_u = zeros(I*M,Nc)
    Q_V_v = zeros(J*M,Nc)
    for m = 1:M
        cols = [nchoose2(m).+(1:m) ; nchoose2.(m.+(1:(M-m))).+m]
        Q_U_u[(1:M:I*M).+(m-1), cols] = U
        Q_V_v[(1:M:J*M).+(m-1), cols] = V
        # Note: the factor of 2 for the m'th column can be ignored since the contraints are equivalent.
    end
    
    Q_U = [Q_U_u Q_U_x]
    Q_V = [Q_V_v Q_V_z]
    return Q_U,Q_V
end

function compute_var_C_from_A(X,Z,F_A_inv,F_C_inv,grad_C,dW_dMu_Mu,dE_dMu_Mu)
    I,K = size(X)
    J,L = size(Z)
    dC_dA = zeros(K*L,J*K)  # Jacobian
    dW_da = zeros(I)
    dE_da = zeros(I)
    ZjZjt = zeros(L,L)
    XtWX = zeros(K,K)
    for j = 1:J
        for l1 = 1:L, l2 = 1:L
            ZjZjt[l1,l2] = Z[j,l1]*Z[j,l2]
        end
        for k = 1:K
            for i = 1:I
                dW_da[i] = dW_dMu_Mu[i,j]*X[i,k]
                dE_da[i] = dE_dMu_Mu[i,j]*X[i,k]
            end
            compute_XtWX!(XtWX,X,dW_da)  # Compute XtWX = X'*(dW_da.*X)
            dF_da = kron(ZjZjt, XtWX)
            dC_dA[:,(j-1)*K+k] = F_C_inv*(-dF_da*(F_C_inv*grad_C) + vec((X'*dE_da)*Z[j,:]'))
        end
    end
    F_A_inv_dC_dAt = zeros(J*K,K*L)
    for j = 1:J
        F_A_inv_dC_dAt[block(j,K),:] = F_A_inv[j]*dC_dA[:,((j-1)*K+1):(j*K)]'
    end
    return vec(sum(dC_dA'.*F_A_inv_dC_dAt,dims=1))
end

function compute_var_C_from_B(X,Z,F_B_inv,F_C_inv,grad_C,dW_dMu_Mu,dE_dMu_Mu)
    I,K = size(X)
    J,L = size(Z)
    dC_dB = zeros(K*L,I*L)  # Jacobian
    dW_db = zeros(J)
    dE_db = zeros(J)
    XiXit = zeros(K,K)
    ZtWZ = zeros(L,L)  
    for i = 1:I
        for k1 = 1:K, k2 = 1:K
            XiXit[k1,k2] = X[i,k1]*X[i,k2]
        end
        for l = 1:L
            for j = 1:J
                dW_db[j] = dW_dMu_Mu[i,j]*Z[j,l]
                dE_db[j] = dE_dMu_Mu[i,j]*Z[j,l]
            end
            compute_XtWX!(ZtWZ,Z,dW_db)  # Compute ZtWZ = Z'*(dW_db.*Z)
            dF_db = kron(ZtWZ, XiXit)
            dC_dB[:,(i-1)*L+l] = F_C_inv*(-dF_db*(F_C_inv*grad_C) + vec(X[i,:]*(dE_db'*Z)))
        end
    end
    F_B_inv_dC_dBt = zeros(I*L,K*L)
    for i = 1:I
        F_B_inv_dC_dBt[block(i,L),:] = F_B_inv[i]*dC_dB[:,((i-1)*L+1):(i*L)]'
    end
    return vec(sum(dC_dB'.*F_B_inv_dC_dBt,dims=1))
end

function compute_var_A_from_V(X,U,D,var_V,F_A_inv,grad_A,dW_dMu_Mu,dE_dMu_Mu)
    J,K = size(grad_A)
    I,M = size(U)
    var_A_from_V = zeros(J*K)
    dA_dV = zeros(K,M)
    dW_dv = zeros(I)
    dE_dv = zeros(I)
    XtWX = zeros(K,K)
    for j = 1:J
        for m = 1:M
            for i = 1:I
                dW_dv[i] = dW_dMu_Mu[i,j]*U[i,m]*D[m]
                dE_dv[i] = dE_dMu_Mu[i,j]*U[i,m]*D[m]
            end
            compute_XtWX!(XtWX,X,dW_dv)  # Compute XtWX = X'*(dW_dv.*X)
            dA_dV[:,m] = (-F_A_inv[j]*XtWX)*(F_A_inv[j]*grad_A[j,:]) + F_A_inv[j]*(X'*dE_dv)
        end
        var_A_from_V[block(j,K)] = sum(dA_dV'.*(var_V[block(j,M)].*dA_dV'), dims=1)
    end
    return var_A_from_V
end

function compute_var_A_from_U(X,V,D,var_U,F_A_inv,grad_A,dW_dMu_Mu,dE_dMu_Mu)
    I,K = size(X)
    J,M = size(V)
    dA_dU = zeros(J*K,I*M)  # Jacobian
    for j = 1:J
        Temp = -F_A_inv[j]*(X.*dW_dMu_Mu[:,j])' .* (X*F_A_inv[j]*grad_A[j,:])' + F_A_inv[j]*(X.*dE_dMu_Mu[:,j])'
        for m = 1:M
            dA_dU[block(j,K), (1:M:I*M).+(m-1)] = Temp*V[j,m]*D[m]
        end
    end
    return vec(sum(dA_dU'.*(var_U.*dA_dU'),dims=1))
end

function compute_var_B_from_U(Z,V,D,var_U,F_B_inv,grad_B,dW_dMu_Mu,dE_dMu_Mu)
    return compute_var_A_from_V(Z,V,D,var_U,F_B_inv,grad_B,dW_dMu_Mu',dE_dMu_Mu')
end

function compute_var_B_from_V(Z,U,D,var_V,F_B_inv,grad_B,dW_dMu_Mu,dE_dMu_Mu)
    return compute_var_A_from_U(Z,U,D,var_V,F_B_inv,grad_B,dW_dMu_Mu',dE_dMu_Mu')  # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< double-check that this is correct
end

function compute_var_V_from_A(UD,X,var_A,F_V_inv,grad_V,dW_dMu_Mu,dE_dMu_Mu)
    return compute_var_A_from_V(UD,X,ones(size(X,2)),var_A,F_V_inv,grad_V,dW_dMu_Mu,dE_dMu_Mu)
end

function compute_var_V_from_B(UD,Z,var_B,F_V_inv,grad_V,dW_dMu_Mu,dE_dMu_Mu)
    return compute_var_A_from_U(UD,Z,ones(size(Z,2)),var_B,F_V_inv,grad_V,dW_dMu_Mu,dE_dMu_Mu)
end

function compute_var_U_from_B(VD,Z,var_B,F_U_inv,grad_U,dW_dMu_Mu,dE_dMu_Mu)
    return compute_var_B_from_U(VD,Z,ones(size(Z,2)),var_B,F_U_inv,grad_U,dW_dMu_Mu,dE_dMu_Mu)
end

function compute_var_S_from_UV(Y,Mu,W,E,r,U,V,D,g_S,h_S,se_U,se_V)
    Q = -W.*E./r
    R = -2*W.*E./(r.+Mu)

    # Delta propagation to account for uncertainty in S due to U
    dg_du = Q*(V.*D')
    dh_du = -dg_du + R*(V.*D')
    ds_du = -dg_du./h_S + dh_du.*(g_S./h_S.^2)
    var_S_from_U = vec(sum(ds_du.*(ds_du.*(se_U.^2)); dims=2))
        
    # Delta propagation to account for uncertainty in S due to V
    I = size(U,1)
    var_S_from_V = zeros(I)
    for i = 1:I
        dg_dv = Q[i,:].*(U[i,:].*D)'
        dh_dv = -dg_dv + R[i,:].*(U[i,:].*D)'
        ds_dv = -dg_dv/h_S[i] + dh_dv*(g_S[i]/h_S[i]^2)
        var_S_from_V[i] = sum(ds_dv.*(ds_dv.*(se_V.^2)))
    end
        
    return var_S_from_U,var_S_from_V
end

function compute_var_T_from_UV(Y,Mu,W,E,r,U,V,D,g_T,h_T,se_U,se_V)
     var_T_from_V,var_T_from_U = compute_var_S_from_UV(Y',Mu',W',E',r',V,U,D,g_T,h_T,se_V,se_U)
     return var_T_from_U,var_T_from_V
end


# _________________________________________________________________________________________________
# The following function (compute_constraint_Jacobian) is used only in the full Fisher information matrix version.

# Jacobian (transposed) of identifiability constraints on A, B, U, and V
# (i.e., Z'*A=0, X'*B=0, X'*U=0, Z'*V=0, U'*U=I, and V'*V=I)
function compute_constraint_Jacobian(X,Z,A,B,U,V)
    I,K = size(X)
    J,L = size(Z)
    M = size(U,2)
    
    # Constraints on A and B
    Q_A = kron(Z, Diagonal(ones(K)))  # if we combine with F_A, these might fall into the same block diagonal structure as F_A
    Q_B = kron(X, Diagonal(ones(L)))  # ... F_B ...
    
    Q_U,Q_V = compute_constraint_Jacobian_UV(X,Z,U,V)
    
    S_A = 1:J*K
    S_B = (1:I*L).+maximum(S_A)
    S_C = (1:K*L).+maximum(S_B)
    S_U = (1:I*M).+maximum(S_C)
    S_V = (1:J*M).+maximum(S_U)
    S_D = (1:M).+maximum(S_V)
    
    Q = zeros(maximum(S_D), K*L+K*L+K*M+Nc+L*M+Nc)
    Q[S_A,1:K*L] = Q_A
    Q[S_B,(1:K*L).+K*L] = Q_B
    Q[S_U,(1:(K*M+Nc)).+K*L.+K*L] = Q_U
    Q[S_V,(1:(L*M+Nc)).+K*L.+K*L.+K*M.+Nc] = Q_V
    
    return Q,Q_A,Q_B,Q_U,Q_V
end




end # module Infer
















