module QMPProblems

export random_instance
export Problem

using LinearAlgebra
using SparseArrays
using LinearMaps
using KrylovKit

using ..QMPSolver: Problem

function symmetrize(A::SparseMatrixCSC{Float64, Int64})
    n, _ = size(A)
    I, J, V = findnz(A)
    return sparse(vcat(I,J),vcat(J,I),vcat(V,V),n,n)
end

function sprand_sym_norm(n::Int, density::Float64)
    A = sprand(n, n, density/2.0)
    sA = symmetrize(A)
    vals, _, _ = eigsolve(sA, 1, :LM; issymmetric=true, ishermitian=true)
    if vals[1] != 0.0
        sA .*= (1/abs(vals[1]))
    end
    return sA
end

function ABc_to_M(A::SparseMatrixCSC{Float64, Int64},B::Matrix{Float64},
    c::Float64,k::Int)
    #= 
    Given   A, B, c
    output  A/2     B/2
            Bᵀ/2    c/k⋅Iₖ
    =#
    return vcat(hcat(A .* 0.5, B .* 0.5), hcat(B' .* 0.5, (c / k) .* sparse(1.0I,k,k)))
end

function random_instance(n::Int, m::Int, k::Int, μ::Float64, density::Float64)
    # generate As, Bs randomly
    As = [sprand_sym_norm(n, density) for _ in 1:m]
    Bs = [randn(n, k) for _ in 1:m]
    for i=1:m
        normalize!(Bs[i])
    end
    cs = zeros(m) # instantiate cs to be populated later

    # set objective to be minimum norm
    A₀ = sparse(1.0I, n, n)
    A₀ .*= 1.0
    B₀ = zeros(n, k)
    c₀ = 0.0

    # pick γ at random then set γ★ such that λ₁(A(γ★)) = μ
    γ = randn(m)
    normalize!(γ)

    temp = zeros(n)
    function Aγ_rmul!(y, x)
        y .= 0
        for i=1:m
            mul!(temp, As[i], x)
            y .+= γ[i] .* temp
        end
    end
    Aγ = LinearMap(Aγ_rmul!, n; ismutating=true, issymmetric=true)
    vals, _, _ = eigsolve(Aγ, n, 1, :SR; issymmetric=true, ishermitian=true)

    γ★ = ((μ - 1) / vals[1]) .* γ
    
    # solve for X★
    temp = zeros(n)
    function A★_rmul!(y, x)
        mul!(y, A₀, x)
        for i=1:m
            mul!(temp, As[i], x)
            y .+= γ★[i] .* temp
        end
    end
    A★ = LinearMap(A★_rmul!, n; ismutating=true, issymmetric=true)
    B★ = copy(B₀)
    for i=1:m
        B★ .+= γ★[i] .* Bs[i]
    end
    X★ = zeros(n,k)
    for i=1:k
        temp★, _ = linsolve(A★, B★[:,i]; isposdef=true)
        temp★ .*= -1
        X★[:,i] .= temp★
    end

    # compute cs to make X★ feasible and compute opt
    temp = zeros(n, k)
    for i=1:m
        mul!(temp, As[i], X★)
        cs[i] = - 0.5 * dot(X★, temp)
        cs[i] -= dot(Bs[i], X★)
    end

    mul!(temp, A₀, X★)
    opt = 0.5 * dot(X★, temp) + dot(B₀, X★) + c₀

    # create M matrices
    M₀ = ABc_to_M(A₀, B₀, c₀, k)
    Ms = [ABc_to_M(As[i], Bs[i], cs[i], k) for i=1:m]

    return Problem(n, k, m, A₀, B₀, c₀, M₀, As, Bs, cs, Ms, 1, 1), X★, γ★, opt
end

end