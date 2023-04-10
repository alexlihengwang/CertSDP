struct QMMP
    ñ::Int64
    m̃::Int64
    G11::SubArray{Float64}
    G12::SubArray{Float64}
    G22::Float64
    observations::Vector{Float64}
    t::Float64
    G11_norm::Float64
    Gtop_norm::Float64
end

function QMMP(p::PhaseRetProblem)
    G11 = @view p.G[1:p.m-1, 1:p.n-1]
    G11_norm = svds(G11)[1].S[1]
    G11_norm *= 1.1

    Gtop = @view p.G[1:p.m-1, 1:p.n]
    Gtop_norm = svds(Gtop)[1].S[1]
    Gtop_norm *= 1.1

    return QMMP(
        p.n - 1,
        p.m - 1,
        (@view p.G[1:p.m-1,1:p.n-1]),
        (@view p.G[1:p.m-1,p.n]),
        p.G[p.m,p.n],
        p.observations[1:p.m-1],
        sqrt(p.observations[p.m]) / p.G[p.m,p.n],
        G11_norm,
        Gtop_norm
    )
end


struct Cq_mem
    temp_m̃::Vector{Float64}
    temp_ñ::Vector{Float64}
    x::Vector{Float64}
    y::Vector{Float64}
    x_new::Vector{Float64}
    b_γ::Vector{Float64}
end

function Cq_mem(n::Int64, m::Int64)
    return Cq_mem(
        zeros(m - 1),
        zeros(n - 1),
        zeros(n - 1),
        zeros(n - 1),
        zeros(n - 1),
        zeros(n-1))
end

function rmul_Aγ!(qmmp::QMMP, γ::Vector{Float64}, out, in, temp::Vector{Float64})
    mul!(temp, qmmp.G11, in)
    temp .*= γ
    mul!(out, qmmp.G11', temp)
    out .= 2 .* (in .- out)
end

function construct_qmmp(γ::Vector{Float64}, x::Vector{Float64}, qmmp::QMMP, mem::Cq_mem)
    #= 
        return false if A(γ) is not positive definite
        else, return true and data for QMMP, i.e.,
            r, μ̃, L̃, subopt_bound
            such that 
                𝒰 = B(γ, r)
                μ̃ ⪯ A(γ) ⪯ L̃ for all γ ∈ 𝒰
                Q_𝒰 (x) - min Q_𝒰 ≤ subopt_bound
    =#

    min_vals, _ = eig_from_rmul(
        (out, in) -> rmul_Aγ!(qmmp, γ, out, in, mem.temp_m̃),
        qmmp.ñ, 1, :SR)
    λ_min = min_vals[1]

    max_vals, _ = eig_from_rmul(
        (out, in) -> rmul_Aγ!(qmmp, γ, out, in, mem.temp_m̃),
        qmmp.ñ, 1, :LR)
    λ_max = max_vals[1]

    κ = λ_max / λ_min

    if λ_min <= 0 || κ >= 1e6
        return false
    end

    # construct B(γ,r) ⊆ Γ
    r = λ_min * (κ / (1 + 2 * κ)) / (2 * qmmp.G11_norm^2)
    μ̃ = λ_min - λ_min * (1 - κ / (1 + 2 * κ))
    L̃ = λ_max + λ_min * (1 - κ / (1 + 2 * κ))

    # subopt bound
    mul!(mem.temp_m̃, qmmp.G11, x)
    mem.temp_m̃ .= γ .* (mem.temp_m̃ .+ qmmp.t .* qmmp.G12)
    mul!(mem.temp_ñ, qmmp.G11', mem.temp_m̃)
    mem.temp_ñ .= 2 .* (x .- mem.temp_ñ)

    mul!(mem.temp_m̃, qmmp.G11, x)
    mem.temp_m̃ .+= qmmp.t .* qmmp.G12
    mem.temp_m̃ .^= 2
    mem.temp_m̃ .= qmmp.observations .- mem.temp_m̃

    subopt_bound = r * norm(mem.temp_m̃) + norm(mem.temp_ñ)^2 / (2 * μ̃)  # r * ‖(qᵢ(x))‖ + ‖∇₂q(γ,X)‖²/(2μ̃)

    return true, r, μ̃, L̃, subopt_bound
end