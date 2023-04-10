struct Pm_mem
    γx::Vector{Float64}
    γx_new::Vector{Float64}
    γy::Vector{Float64}
    ñ::Vector{Float64}
    m̃::Vector{Float64}
    m̃2::Vector{Float64}
end

function Pm_mem(n::Int64, m::Int64)
    Pm_mem(
        zeros(m - 1),
        zeros(m - 1),
        zeros(m - 1),
        zeros(n - 1),
        zeros(m - 1),
        zeros(m - 1))
end

function prox_map(prob::QMMP, ĝ::Vector{Float64}, lin_term::Vector{Float64},
    r::Float64, L::Float64, B::Float64, ξ::Vector{Float64},
    ret::Vector{Float64}, ϵ::Float64, mem::Pm_mem;
    maxiter::Int=100000)
    #= 
        save ϵ-minimizer of prox-map in ret.

        Use AGD on 
        max_{δ ∈ B(0,r)} - ‖ 𝒢δ ‖² / (2L) + ⟨lin_term, δ⟩

        assume
            Ĝ = ∇₂ q(γ̂,Ξ)
            B ≥ ‖ 𝒢ᵀ𝒢 ‖₂
            lin_term = (qᵢ(Ξ))ᵢ - 𝒢ᵀĜ / L        
    =#


    if ϵ < 1e-16
        ϵ = 1e-16
    end

    L2ψ = B^2 / L
    mem.γx .= 0
    mem.γy .= mem.γx

    α = 2 * 3 / (3 + sqrt(21))

    converged = false
    for tt = 0:maxiter
        # ==============
        # mem.m̃ = ∇ψ(γy) = - 𝒢ᵀ 𝒢 γy / L + lin_term
        # --------------
        𝒢_rmul!(prob, ξ, mem.ñ, mem.γy, mem.m̃2)
        𝒢ᵀ_rmul!(prob, ξ, mem.m̃, mem.ñ, mem.m̃2)
        mem.m̃ .*= (-1.0 / L)
        mem.m̃ .+= lin_term
        # ==============

        # ==============
        # γx_new = proj_{‖⋅‖ ≤ r} γy + ∇ψ(γy) / L2ψ
        # --------------
        mem.γx_new .= mem.γy .+ mem.m̃ ./ L2ψ

    	γx_new_norm = norm(mem.γx_new)
    	if γx_new_norm > r
    		mem.γx_new .*= (r / γx_new_norm)
    	end
        # ==============

        α_new = (- α^2 + sqrt(α^4 + 4 * α^2)) / 2
        β = α * (1 - α) / (α^2 + α_new)

        mem.γy .= (1 + β) .* mem.γx_new .- β .* mem.γx
        mem.γx .= mem.γx_new
        α = α_new

        # bound suboptimality
        if tt % 10 == 0
            # ==============
            # mem.m = ∇ψ(γx_new) = - 𝒢ᵀ 𝒢 γx_new / L + lin_term
            # --------------
            𝒢_rmul!(prob, ξ, mem.ñ, mem.γx_new, mem.m̃2)
            𝒢ᵀ_rmul!(prob, ξ, mem.m̃, mem.ñ, mem.m̃2)
            mem.m̃ .*= (-1.0 / L)
            mem.m̃ .+= lin_term
            # ==============

            primal_subopt = r * norm(mem.m̃) - dot(mem.m̃, mem.γx_new)

            if primal_subopt <= ϵ
                converged = true
                break
            end
        end
    end

    # ==========
    # ret = ξ - (ĝ + 𝒢 γx_new) / L
    # ----------
    𝒢_rmul!(prob, ξ, mem.ñ, mem.γx, mem.m̃)
    ret .= ξ .- (1.0 / L) .* (ĝ .+ mem.ñ)
    # ==========

    return converged
end