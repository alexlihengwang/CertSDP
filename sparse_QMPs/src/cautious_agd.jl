struct Ca_mem
    lin_term::Vector{Float64}
    X::Matrix{Float64}
    X_new::Matrix{Float64}
    Ξ::Matrix{Float64}
    Ĝ::Matrix{Float64}
    nk::Matrix{Float64}
end

function Ca_mem(n::Int64, k::Int64, m::Int64)
    Ca_mem(
        zeros(m),
        zeros(n, k),
        zeros(n, k),
        zeros(n, k),
        zeros(n, k),
        zeros(n, k))
end

function cautious_agd(qmp::Problem, γ̂::Vector{Float64},
    ret::Matrix{Float64}, r::Float64, μ::Float64, L::Float64,
    gap::Float64, R::Float64, mem::Ca_mem, pm_mem::Pm_mem;
    maxiter::Int=100000, iterate_info::Union{Iterate_info,Nothing}=nothing)
    #= 
        Run Cautious AGD with 𝒰 = 𝔹(γ̂,r)
        warm-start at ret, return at ret
        
        Assume:
            μ ⪯ A(γ) ⪯ L for all γ ∈ 𝒰
            ‖ ret - X★ ‖ ≤ R
            Q_𝒰(ret) - min_X Q_𝒰(X) ≤ gap
    =#


    κ = L / μ
    κ̃ = (L - μ / 2.0) / (μ / 2.0)
    α = sqrt(1.0 / κ̃)
    β = (1.0 - α) / (1.0 + α)

    subopt_bound = 4.0 * gap
    prox_error = (α / 2.0) * gap / κ

    mem.X .= ret
    mem.Ξ .= ret
    flag = false

    B = nothing
    maxqi = nothing
    qi = nothing
    threshold = nothing

    for tt in 0:maxiter
        # ==========================
        # X_new ≈ X_L(Ξ)
        # --------------------------

        # Set Ĝ = ∇₂ q(γ̂,Ξ)
        mul!(mem.Ĝ, qmp.A₀, mem.Ξ)
        mem.Ĝ .+= qmp.B₀
        for i = 1:qmp.m
            mul!(mem.nk, qmp.As[i], mem.Ξ)
            mem.Ĝ .+= γ̂[i] .* (mem.nk .+ qmp.Bs[i])
        end

        # Let B ≥ ‖∇²ψ‖₂
        B = sqrt(qmp.m) * (1 + norm(mem.Ξ))

        # Set lin_term = (qᵢ(Ξ))ᵢ - 𝒢ᵀĜ / L
        𝒢ᵀ_rmul!(qmp, mem.Ξ, mem.lin_term, mem.Ĝ, mem.nk)
        mem.lin_term .*= -1.0 / L

        for i = 1:qmp.m
            mul!(mem.nk, qmp.As[i], mem.Ξ)
            mem.lin_term[i] += dot(mem.Ξ, mem.nk) / 2
            mem.lin_term[i] += dot(qmp.Bs[i], mem.Ξ)
        end
        mem.lin_term .+= qmp.cs

        # compute prox-map
        prox_map(qmp, mem.Ĝ, mem.lin_term,
            r, L, B, mem.Ξ,
            mem.X_new, prox_error, pm_mem; maxiter=100000)

        # --------------------
        mem.Ξ .= (1 + β) .* mem.X_new .- β .* mem.X
        mem.X .= mem.X_new

        subopt_bound *= (1.0 - α / 2.0)
        prox_error *= (1.0 - α / 2.0)

        !isnothing(iterate_info) && push_p!(iterate_info, mem.X)

        if tt % 10 == 0
            maxqi = 0
            for i = 1:qmp.m
                mul!(mem.nk, qmp.As[i], mem.X)
                qi = dot(mem.X, mem.nk) / 2 + dot(qmp.Bs[i], mem.X) + qmp.cs[i]
                maxqi = max(maxqi, abs(qi))
            end

            threshold = (qmp.A_bound * 2 * subopt_bound / μ) + (qmp.A_bound * R + qmp.B_bound) * sqrt(2 * subopt_bound / μ)

            if maxqi > threshold
                break
            elseif (maxqi <= 1e-13) && (subopt_bound <= 1e-13)
                flag = true
                break
            end
        end
    end

    !isnothing(iterate_info) && push_p!(iterate_info, nothing)

    ret .= mem.X
    return flag
end

# Let 𝒢γ = ∑ γᵢ (AᵢΞ + Bᵢ) and 𝒢ᵀ its adjoint
function 𝒢_rmul!(qmp::Problem, Ξ::Matrix{Float64},
    Xout::Matrix{Float64}, γin::Vector{Float64}, temp::Matrix{Float64})
    Xout .= 0
    for i = 1:qmp.m
        mul!(temp, qmp.As[i], Ξ)
        Xout .+= γin[i] .* (temp .+ qmp.Bs[i])
    end
end
function 𝒢ᵀ_rmul!(qmp::Problem, Ξ::Matrix{Float64}, γout::Vector{Float64},
    Xin::Matrix{Float64}, temp::Matrix{Float64})
    γout .= 0
    for i = 1:qmp.m
        mul!(temp, qmp.As[i], Ξ)
        γout[i] = dot(temp, Xin)
        γout[i] += dot(qmp.Bs[i], Xin)
    end
end
