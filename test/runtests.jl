
using Test
using BilinearModels
using Statistics

GBM = BilinearModels
rms(x) = sqrt(mean(x.^2))


@testset "Common" begin
    A = [1 2; 3 4]
    @test GBM.SS(A) == 30
end

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


@testset "fit" begin
	Y = [1 2 3; 4 5 6; 7 8 0]
	I,J = size(Y)
	A,B,C,D,U,V = GBM.fit(Y,ones(I,1),ones(J,1),0; verbose=false)
	A0 = reshape([-0.11119, 0.11507, -0.00388],J,1)
	B0 = reshape([-0.58296, 0.20316, 0.3798],I,1)
	C0 = reshape([1.28247],1,1)
    @test rms(A-A0) < 1e-4  # M = 0
    @test rms(B-B0) < 1e-4  # M = 0
    @test rms(C-C0) < 1e-4  # M = 0
	
	# Y = [1 2 3 4; 0 6 7 8; 1 3 4 0; 1 4 5 6]
	# I,J = size(Y)
	# A,B,C,D,U,V = GBM.fit(Y,ones(I,1),ones(J,1),1)
	# A0 = reshape(,J,1)
	# B0 = reshape(,I,1)
	# C0 = reshape(,1,1)
    # @test rms(A-A0) < 1e-12  # M = 1
    # @test rms(B-B0) < 1e-12  # M = 1
    # @test rms(C-C0) < 1e-12  # M = 1
end







