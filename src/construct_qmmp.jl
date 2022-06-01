struct Cq_mem
    n::Vector{Float64}
    m::Vector{Float64}
    nk_1::Matrix{Float64}
    nk_2::Matrix{Float64}
end

function Cq_mem(n::Int64, k::Int64, m::Int64)
    Cq_mem(
        zeros(n),
        zeros(m),
        zeros(n, k),
        zeros(n, k))
end
function construct_qmmp(qmp::Problem, γ::Vector{Float64}, X::Matrix{Float64}, mem::Cq_mem)
    #= 
        return false if A(γ) is not positive definite
        else, return true and data for QMMP, i.e.,
            r, μ̃, L̃, subopt_bound, and R
            such that 
                𝒰 = B(γ, r)
                μ̃ ⪯ A(γ) ⪯ L̃ for all γ ∈ 𝒰
                Q_𝒰 (X) - min Q_𝒰 ≤ subopt_bound
                ‖ X - argmin Q_𝒰 ‖ ≤ R
    =#

    n, m = qmp.n, qmp.m

    function Aγ_rmul!(y, x)
        mul!(y, qmp.A₀, x)
        for i = 1:m
            mul!(mem.n, qmp.As[i], x)
            y .+= γ[i] .* mem.n
        end
    end

    min_vals, _ = eig_from_rmul(Aγ_rmul!, n, 1, :SR)
    λ_min = min_vals[1]

    max_vals, _ = eig_from_rmul(Aγ_rmul!, n, 1, :LR)
    λ_max = max_vals[1]

    κ = λ_max / λ_min

    if λ_min <= 0 || κ >= 1e6
        return false
    end

    # construct B(γ,r) ⊆ Γ
    r = λ_min * (κ / (1 + 2 * κ)) / (sqrt(m) * qmp.A_bound)
    μ̃ = λ_min - r * (sqrt(m) * qmp.A_bound)
    L̃ = λ_max + r * (sqrt(m) * qmp.A_bound)

    # subopt bound
    mul!(mem.nk_1, qmp.A₀, X)
    mem.nk_1 .+= qmp.B₀

    for i = 1:m
        mul!(mem.nk_2, qmp.As[i], X)
        mem.nk_1 .+= γ[i] .* (mem.nk_2 .+ qmp.Bs[i])
        mem.m[i] = dot(X, mem.nk_2) / 2 + dot(qmp.Bs[i], X) + qmp.cs[i]
    end

    subopt_bound = r * norm(mem.m) + norm(mem.nk_1)^2 / (2 * μ̃)  # r * ‖(qᵢ(x))‖ + ‖∇₂q(γ,X)‖²/(2μ̃)

    return true, r, μ̃, L̃, subopt_bound, sqrt(2 * subopt_bound / μ̃)
end