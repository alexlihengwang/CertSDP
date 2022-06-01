module QMPSolver

export Problem
export Iterate_info

export certSDP, cssdp, scs_solve, proxSDP_solve, sketchy_cgal

using LinearAlgebra
using SparseArrays
using LinearMaps
using Arpack

using Convex, MosekTools, SCS
using MathOptInterface
using ProxSDP
const MOI = MathOptInterface
using JuMP

struct Problem
    #=
        min_{X ∈ R^(n \times k)} {q₀(X) : qᵢ(X) = 0, ∀ i ∈ [m]}

        where ∀ i ∈ [0,...,m]:
        qᵢ(X) = ⟨X, Aᵢ X⟩ / 2 + ⟨Bᵢ, X⟩ + cᵢ
        Mᵢ = (Aᵢ/2  Bᵢ/2)
             (Bᵢ/2⊺ cᵢ/k⋅Iₖ)
    =#
    n::Int
    k::Int
    m::Int
    A₀::SparseMatrixCSC{Float64,Int64}
    B₀::Matrix{Float64}
    c₀::Float64
    M₀::SparseMatrixCSC{Float64,Int64}
    As::Vector{SparseMatrixCSC{Float64,Int64}}
    Bs::Vector{Matrix{Float64}}
    cs::Vector{Float64}
    Ms::Vector{SparseMatrixCSC{Float64,Int64}}
    A_bound::Float64
    B_bound::Float64
end

include("./utility.jl")
include("./iterate_info.jl")
include("./certSDP.jl")
include("./first_order_info.jl")
include("./construct_qmmp.jl")
include("./prox_map.jl")
include("./cautious_agd.jl")

include("./other_solvers.jl")

end