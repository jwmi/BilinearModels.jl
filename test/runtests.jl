
using Test
using BilinearModels
using Statistics
using Random


GBM = BilinearModels
rms(x) = sqrt(mean(x.^2))
relative_mse(x,y) = mean((x-y).^2) / mean(y.^2)


if false

@testset "fit_leastsquares" begin
    logMu = [1 2 3; 4 5 6; 7 8 0]
    I,J = size(logMu)
    A,B,C,D,U,V = GBM.fit_leastsquares(logMu,ones(I,1),ones(J,1),0)
    A0 = reshape([0,1,-1],J,1)
    B0 = reshape([-2,1,1],I,1)
    C0 = reshape([4],1,1)
    @test rms(A-A0) < 1e-12  # M = 0
    @test rms(B-B0) < 1e-12  # M = 0
    @test rms(C-C0) < 1e-12  # M = 0
    
    logMu = [1 2 3 4; 0 6 7 8; 1 3 4 0; 1 4 5 6]
    I,J = size(logMu)
    A,B,C,D,U,V = GBM.fit_leastsquares(logMu,ones(I,1),ones(J,1),1)
    A0 = reshape([-2.6875, 0.3125, 1.3125, 1.0625],J,1)
    B0 = reshape([-0.9375, 1.8125, -1.4375, 0.5625],I,1)
    C0 = reshape([3.4375],1,1)
    @test rms(A-A0) < 1e-12  # M = 1
    @test rms(B-B0) < 1e-12  # M = 1
    @test rms(C-C0) < 1e-12  # M = 1
end


@testset "fit and infer 3x3" begin
    Y = [1 2 3; 4 5 6; 7 8 0]
    I,J = size(Y)
    X = ones(I,1)
    Z = ones(J,1)

    @testset "fit 3x3 M=0" begin
        A,B,C,D,U,V,S,T,omega,logp = GBM.fit(Y,X,Z,0; verbose=false)
        A0 = reshape([-0.11119, 0.11507, -0.00388],J,1)
        B0 = reshape([-0.58296, 0.20316, 0.3798],I,1)
        C0 = reshape([1.28247],1,1)
        @test rms(A-A0) < 1e-4
        @test rms(B-B0) < 1e-4
        @test rms(C-C0) < 1e-4
    end
    
    @testset "infer 3x3 M=0" begin
        A,B,C,D,U,V,S,T,omega,logp = GBM.fit(Y,X,Z,0; verbose=false)
        se_A,se_B,se_C,se_D,se_U,se_V,se_S,se_T = GBM.infer(Y,X,Z,A,B,C,D,U,V,S,T,omega)
        se_A0 = reshape([0.37311, 0.34465, 0.44011],J,1)
        se_B0 = reshape([0.42922, 0.3305, 0.4022],I,1)
        se_C0 = reshape([0.39282],1,1)
        @test rms(se_A-se_A0) < 1e-4
        @test rms(se_B-se_B0) < 1e-4
        @test rms(se_C-se_C0) < 1e-4
    end
end


@testset "fit and infer 5x5" begin
    Y = [  27    20   12    9   19
         1760  2490  110   15   17
           11    22   97  959  110
           55    51   13   53  416
           30   154   91   77    2]
    I,J = size(Y)
    X = ones(I,1)
    Z = ones(J,1)
    @testset "fit 5x5 M=0" begin
        Random.seed!(0)
        A,B,C,D,U,V,S,T,omega,logp = GBM.fit(Y,X,Z,0; verbose=false)
        A0 = reshape([0.05279, 0.14563, -0.7032, -0.08462, 0.58939],J,1)
        B0 = reshape([-1.63601, 1.63977, 0.57607, -0.62367, 0.04384],I,1)
        C0 = reshape([4.58132],1,1)
        @test rms(A-A0) < 1e-4
        @test rms(B-B0) < 1e-4
        @test rms(C-C0) < 1e-4
    end
    
    @testset "fit 5x5 M=1" begin
        Random.seed!(0)
        A,B,C,D,U,V,S,T,omega,logp = GBM.fit(Y,X,Z,1; max_iterations=100, verbose=false)
        A0 = reshape([0.09317, 0.59052, -0.19255, 0.25732, -0.74847],J,1)
        B0 = reshape([-1.37734, 1.38807, 0.66823, -0.16445, -0.51451],I,1)
        C0 = reshape([4.26076],1,1)
        D0 = [4.16186]
        U0 = reshape([0.20318, -0.47531, 0.12818, 0.66611, -0.52217],I,1)
        V0 = reshape([-0.0385, -0.30784, -0.3962, -0.11406, 0.8566],J,1)
        @test rms(A-A0) < 1e-4
        @test rms(B-B0) < 1e-4
        @test rms(C-C0) < 1e-4
        @test rms(D-D0) < 1e-4
        @test rms(U-U0) < 1e-4
        @test rms(V-V0) < 1e-4
    end

    @testset "fit 5x5 M=2" begin
        Random.seed!(0)
        A,B,C,D,U,V,S,T,omega,logp = GBM.fit(Y,X,Z,2; max_iterations=200, verbose=false)
        A0 = reshape([0.10861, 0.63701, -0.26518, 0.07359, -0.55403],J,1)
        B0 = reshape([-1.23275, 1.14285, 0.3443, 0.14662, -0.40102],I,1)
        C0 = reshape([4.00043],1,1)
        D0 = [5.88215, 3.90616]
        U0 = reshape([-0.01696, 0.70766, -0.61116, -0.28701, 0.20747, -0.2651, -0.15798, 0.32764, -0.58191, 0.67734],I,2)
        V0 = reshape([0.48904, 0.49905, 0.02309, -0.50706, -0.50413, -0.29492, 0.06175, 0.44077, 0.48501, -0.69261],J,2)
        GBM.permute_to_match_latent_dimensions!(D,U,V,D0,U0,V0)
        @test rms(A-A0) < 1e-4
        @test rms(B-B0) < 1e-4
        @test rms(C-C0) < 1e-4
        @test rms(D-D0) < 1e-4
        @test rms(U-U0) < 1e-4
        @test rms(V-V0) < 1e-4
    end
    
end


@testset "fit simulations" begin
    I = 1000  # number of features
    J = 100  # number of samples
    scenarios = [(1,1,0),(1,1,2),(2,2,0),(2,2,2),(3,3,0),(3,3,3)]
    odist = "NB"  # true outcome distribution to use for generating the data
    cdist = "Normal"  # covariate distribution
    pdist = "Normal"  # true parameter distribution
    
    for (K,L,M) in scenarios
        @testset "fit sim K=$K L=$L M=$M" begin
            Random.seed!(0)
            Y,X,Z,A0,B0,C0,D0,U0,V0,S0,T0,omega0 = GBM.simulate_parameters_and_data(I,J,K,L,M; odist=odist, cdist=cdist, pdist=pdist)
            A,B,C,D,U,V,S,T,omega,logp = GBM.fit(Y,X,Z,M; verbose=false)
            GBM.permute_to_match_latent_dimensions!(D,U,V,D0,U0,V0)
            @test relative_mse(A,A0) < 1e-1
            @test relative_mse(B,B0) < 1e-1
            @test relative_mse(C,C0) < 1e-1
            if M > 0
                @test relative_mse(D,D0) < 1e-1
                @test relative_mse(U,U0) < 1e-1
                @test relative_mse(V,V0) < 1e-1
            end
            @test relative_mse(S,S0) < 1e-1
            @test relative_mse(T,T0) < 1e-1
            @test relative_mse(omega,omega0) < 1e-1
        end
    end
end




@testset "validate_inputs" begin
    Y = [19  6   69  143
         11  1  120   73
          9  4   28   30
         28  0  152  510
         24  4   33   37]
    I,J = size(Y)
    X = [1.0  -1.8
         1.0   0.3
         1.0   0.8
         1.0  -0.3
         1.0   1.0]
    Z = [1.0   0.4
         1.0  -1.7
         1.0   0.3
         1.0   1.0]
    M = 2
    
    A,B,C,D,U,V,S,T,omega,logp = GBM.fit(Y,X,Z,M; max_iterations=200, verbose=false)
    @test (try GBM.validate_inputs(Y,X,Z,M); true; catch; false; end)
    @test (try GBM.validate_outputs(Y,X,Z,A,B,C,D,U,V,S,T); true; catch; false; end)
    
    @test (try GBM.validate_inputs(Y[1:end-1,:],X,Z,M); false; catch; true; end)
    @test (try GBM.validate_inputs(Y[:,1:end-1],X,Z,M); false; catch; true; end)
    X1 = copy(X); X1[1,1] = 1.1
    @test (try GBM.validate_inputs(Y,X1,Z,M); false; catch; true; end)
    X1 = copy(X); X1[1,2] = -1.9
    @test (try GBM.validate_inputs(Y,X1,Z,M); false; catch; true; end)
    Z1 = copy(Z); Z1[1,1] = 1.1
    @test (try GBM.validate_inputs(Y,X1,Z,M); false; catch; true; end)
    Z1 = copy(Z); Z1[1,2] = 0.5
    @test (try GBM.validate_inputs(Y,X1,Z,M); false; catch; true; end)
    @test (try GBM.validate_inputs(Y,X[:,[]],Z,M); false; catch; true; end)
    @test (try GBM.validate_inputs(Y,X,Z[:,[]],M); false; catch; true; end)
    @test (try GBM.validate_inputs(Y,X,Z,-1); false; catch; true; end)
    @test (try GBM.validate_inputs(Y,X,Z,1.1); false; catch; true; end)
    X1 = copy(X); X1[:,2] .= 0.0
    @test (try GBM.validate_inputs(Y,X1,Z,M); false; catch; true; end)
    Z1 = copy(X); Z1[:,2] .= 0.0
    @test (try GBM.validate_inputs(Y,X,Z1,M); false; catch; true; end)
end


@testset "validate_outputs" begin
    Y = [19  6   69  143
         11  1  120   73
          9  4   28   30
         28  0  152  510
         24  4   33   37]
    I,J = size(Y)
    X = [1.0  -1.8
         1.0   0.3
         1.0   0.8
         1.0  -0.3
         1.0   1.0]
    Z = [1.0   0.4
         1.0  -1.7
         1.0   0.3
         1.0   1.0]
    M = 2
    
    A,B,C,D,U,V,S,T,omega,logp = GBM.fit(Y,X,Z,M; max_iterations=200, verbose=false)
    @test (try GBM.validate_inputs(Y,X,Z,M); true; catch; false; end)
    @test (try GBM.validate_outputs(Y,X,Z,A,B,C,D,U,V,S,T); true; catch; false; end)
    
    @test (try GBM.validate_outputs(Y,X,Z,A,B.+1,C,D,U,V,S,T); false; catch; true; end)
    @test (try GBM.validate_outputs(Y,X,Z,A.+1,B,C,D,U,V,S,T); false; catch; true; end)
    @test (try GBM.validate_outputs(Y,X,Z,A,B,C,D,U.+1,V,S,T); false; catch; true; end)
    @test (try GBM.validate_outputs(Y,X,Z,A,B,C,D,U,V.+1,S,T); false; catch; true; end)
    @test (try GBM.validate_outputs(Y,X,Z,A,B,C,D,U,V,S.+0.1,T); false; catch; true; end)
    @test (try GBM.validate_outputs(Y,X,Z,A,B,C,D,U,V,S,T.+0.1); false; catch; true; end)
end

end


@testset "Sums of squares & residuals" begin
    @test GBM.SS([1 2; 3 4]) == 30

    # Sums of squares
    Y = [19  6   69  143
         11  1  120   73
          9  4   28   30
         28  0  152  510
         24  4   33   37]
    I,J = size(Y)
    X = [1.0  -1.8
         1.0   0.3
         1.0   0.8
         1.0  -0.3
         1.0   1.0]
    Z = [1.0   0.4
         1.0  -1.7
         1.0   0.3
         1.0   1.0]
    M = 2
    A,B,C,D,U,V,S,T,omega,logp = GBM.fit(Y,X,Z,M; max_iterations=200, verbose=false)
    ss0 = [6.11299, 10.97372, 226.70418, 1.29258]
    @test rms(GBM.sum_of_squares(X,Z,A,B,C,D,U,V) - ss0) < 1e-4
    
    
    # Residuals
    Eps0 = [ 0.02828   0.04522  -0.00233   0.00171
             0.05744   0.04422   0.00622  -0.01013
            -0.13529   0.22086  -0.0258    0.05224
            -0.03172  -0.75872   0.00721  -0.01063
             0.07058  -0.04084   0.01175  -0.01598]
    @test rms(GBM.residuals(Y,X,Z,A,B,C,D,U,V) - Eps0) < 1e-4
    
    Eps0 = [ 2.22565  2.99952  3.69259  3.03659
             2.24561  2.98932  3.69196  3.01556
             1.87941  2.9925   3.48646  2.90446
             2.43919  2.46913  3.97568  3.2978
             2.4989   3.14442  3.93763  3.24986]
    @test rms(GBM.residuals(Y,X,Z,A,B,C,D,U,V; rx=[1], rz=[1]) - Eps0) < 1e-4
    
    Eps0 = [-0.35833   0.13973  -0.03547   0.32695
             0.17787   0.00814   0.04215  -0.13042
            -0.01119   0.19618  -0.03698  -0.03599
            -0.17071  -0.7159   -0.03879   0.13155
             0.35164  -0.1174    0.06614  -0.27488]
    @test rms(GBM.residuals(Y,X,Z,A,B,C,D,U,V; rx=[2], ru=[2]) - Eps0) < 1e-4

end




























