struct Pm_mem
    nk_1::Matrix{Float64}
    nk_2::Matrix{Float64}
    m::Vector{Float64}
    γx::Vector{Float64}
    γy::Vector{Float64}
    γx_new::Vector{Float64}
end

function Pm_mem(n::Int64, k::Int64, m::Int64)
    Pm_mem(
        zeros(n, k),
        zeros(n, k),
        zeros(m),
        zeros(m),
        zeros(m),
        zeros(m))
end

function prox_map(qmp::Problem, Ĝ::Matrix{Float64}, lin_term::Vector{Float64},
    r::Float64, L::Float64, B::Float64, Ξ::Matrix{Float64},
    ret::Matrix{Float64}, ϵ::Float64, mem::Pm_mem;
    maxiter::Int=100000)
    #= 
        save ϵ-minimizer of prox-map in ret.

        Use AGD on 
        max_{δ ∈ B(0,r)} - ‖ Ĝ + 𝒢δ ‖² / (2L) + ⟨lin_term, δ⟩

        assume
            Ĝ = ∇₂ q(γ̂,Ξ)
            B ≥ ‖∇²ψ‖₂
            lin_term = (qᵢ(Ξ))ᵢ - 𝒢ᵀĜ / L
    =#


    if ϵ < 1e-16
        println("ϵ=$ϵ too small in prox_map, setting ϵ = 1e-16")
        ϵ = 1e-16
    end

    L2ψ = B^2 / L
    mem.γy .= mem.γx

    α = 2 * 3 / (3 + sqrt(21))

    for tt = 0:maxiter
        # ==============
        # mem.m = ∇ψ(γy) = - 𝒢ᵀ 𝒢 γy / L + lin_term
        # --------------
        𝒢_rmul!(qmp, Ξ, mem.nk_1, mem.γy, mem.nk_2)
        𝒢ᵀ_rmul!(qmp, Ξ, mem.m, mem.nk_1, mem.nk_2)
        mem.m .*= (-1.0 / L)
        mem.m .+= lin_term
        # ==============

        # ==============
        # γx_new = proj_{‖⋅‖ ≤ r} γy + ∇ψ(γy) / B
        # --------------
        mem.γx_new .= mem.γy .+ mem.m ./ L2ψ

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
            𝒢_rmul!(qmp, Ξ, mem.nk_1, mem.γx_new, mem.nk_2)
            𝒢ᵀ_rmul!(qmp, Ξ, mem.m, mem.nk_1, mem.nk_2)
            mem.m .*= (-1.0 / L)
            mem.m .+= lin_term
            # ==============

            primal_subopt = r * norm(mem.m) - dot(mem.m, mem.γx_new)

            if primal_subopt <= ϵ
                break
            end
        end
    end

    # ==========
    # ret = Ξ - (Ĝ + 𝒢 γx_new) / L
    # ----------
    𝒢_rmul!(qmp, Ξ, mem.nk_1, mem.γx, mem.nk_2)
    ret .= Ξ .- (1.0 / L) .* (Ĝ .+ mem.nk_1)

    # ==========
end