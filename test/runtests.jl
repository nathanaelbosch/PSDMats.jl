using Test
using PSDMatrices
using LinearAlgebra
using Suppressor
using Aqua

M_square = [1 1; 2 20]
M_tall = [1 1; 2 20; 3 30]
M_wide = [1 1 1; 2 20 200]
M_neg = -ones(1, 1)
eltypes = (Int64, Float64, BigFloat)

@testset "PSDMatrices.jl" begin
    @testset "eltype=$t | shape=$(size(Mbase))" for t in eltypes,
        Mbase in (M_square, M_tall, M_wide, M_neg)

        M = t.(Mbase)
        S = PSDMatrix(M)
        X = rand(t, size(S, 1), size(S, 2))

        @testset "Base" begin
            @test eltype(S) == t
            @test size(S) == size(Matrix(S))
            @test S == S
            @test S ≈ S
            @test copy(S) == S
            @test !(copy(S) === S)
            @test typeof(similar(S)) == typeof(S)
            @test (S2 = similar(S); copy!(S2, S); S2 == S)
            if size(M, 1) == size(M, 2)
                @test Matrix(inv(S)) ≈ inv(Matrix(S))
            end
            @suppress_out @test isnothing(show(S))
            @suppress_out @test isnothing(show(stdout, MIME("text/plain"), S))
            @test iszero(S) == iszero(S.R)
        end

        @testset "LinearAlgebra" begin
            @test diag(S) ≈ diag(Matrix(S))
            @test det(S) ≈ det(Matrix(S))
            @test logdet(S) ≈ logdet(Matrix(S))
            if (size(M, 1) >= size(M, 2))
                @test S \ X ≈ Matrix(S) \ X
                @test X / S ≈ X / Matrix(S)

                v = rand(t, size(S, 2))
                @test transpose(v) / S ≈ transpose(v) / Matrix(S)
                @test adjoint(v) / S ≈ adjoint(v) / Matrix(S)
            end
        end

        @testset "Exports" begin
            @test norm(Matrix(S) - M' * M) == 0.0

            @test let x = rand(t), a = rand(t)
                X_A_Xt(a, x) ≈ x * a * x'
            end

            SM = Matrix(S)
            @test X_A_Xt(SM, X) ≈ X * SM * X'
            @test X_A_Xt!(copy(SM), SM, X) ≈ X_A_Xt(SM, X)

            @test Matrix(X_A_Xt(S, X)) ≈ X_A_Xt(Matrix(S), X)
            @test begin
                product_eltype = typeof(one(eltype(X)) * one(eltype(S)))
                S2 = PSDMatrix(zeros(product_eltype, size(S.R)...))
                X_A_Xt!(S2, S, X)
                Matrix(S2) ≈ Matrix(X_A_Xt(S, X))
            end
            @test Matrix(add_qr(S, S)) ≈ Matrix(S) + Matrix(S)
            if (size(M, 1) >= size(M, 2))
                @test Matrix(add_cholesky(S, S)) ≈ Matrix(S) + Matrix(S)
                tri = triangularize_factor(S)
                @test tri.R isa UpperTriangular
                @test Matrix(tri) ≈ Matrix(S)
            end
        end
    end
    @testset "Aqua" begin
        Aqua.test_all(PSDMatrices)
    end
end
nothing
