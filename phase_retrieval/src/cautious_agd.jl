struct Ca_mem
    x::Vector{Float64}
    ξ::Vector{Float64}
    x_new::Vector{Float64}
    ĝ::Vector{Float64}
    lin_term::Vector{Float64}
    temp_m̃::Vector{Float64}
    temp_m̃2::Vector{Float64}
    x_full::Vector{Float64}
    sum_errors_hist::Vector{Float64} # save a small amount of sum errors data
end

function Ca_mem(n::Int64, m::Int64)
    return Ca_mem(
        zeros(n - 1),
        zeros(n - 1),
        zeros(n - 1),
        zeros(n - 1),
        zeros(m - 1),
        zeros(m - 1),
        zeros(m - 1),
        zeros(n),
        zeros(10))
end

function cautious_agd(prob::QMMP, γ̂::Vector{Float64}, ret::Vector{Float64}, r::Float64, μ::Float64, L::Float64,
    gap::Float64,
    mem::Ca_mem, pm_mem::Pm_mem;
    maxiter::Int=100000, iterate_info::Union{Iterate_info,Nothing}=nothing, termination_criteria::Float64=1e-8)
    #= 
        Run Cautious AGD with 𝒰 = 𝔹(γ̂,r)
        warm-start at ret, return at ret
        
        Assume:
            μ ⪯ A(γ) ⪯ L for all γ ∈ 𝒰
            Q_𝒰(ret) - min_X Q_𝒰(X) ≤ gap

        A(γ) = 2(I - G_11' Diag(γ) G_11)
        b(γ) = -2 t G_11' Diag(γ) G_12
        c(γ) = t^2 - <G12.^2 .* t^2 - obs_top, γ> 
    =#

    κ = L / μ
    κ̃ = (L - μ / 2.0) / (μ / 2.0)
    α = sqrt(1.0 / κ̃)
    β = (1.0 - α) / (1.0 + α)

    subopt_bound = 4.0 * gap
    prox_error = (α / 2.0) * gap / κ

    mem.x .= ret
    mem.ξ .= ret
    flag = -1 # -1 for run out of iterations, 0 for detecting not certificate, 1 for converged

    mem.sum_errors_hist .= Inf
    
    for tt in 1:maxiter
        # ==========================
        # X_new ≈ X_L(Ξ)
        # --------------------------

        # Set ĝ = ∇₂ q(γ̂,ξ) = A(γ̂)ξ + b(γ̂)
        #   = 2(ξ - G_11' Diag(γ) (G_11 ξ + t G_12))
        mul!(mem.temp_m̃, prob.G11, mem.ξ)
        mem.temp_m̃ .= γ̂ .* (mem.temp_m̃ .+ prob.t .* prob.G12)
        mul!(mem.ĝ, prob.G11', mem.temp_m̃)
        mem.ĝ .= 2 .* (mem.ξ .- mem.ĝ)

        # Let B ≥ ‖𝒢ᵀ𝒢‖₂
        mul!(mem.temp_m̃, prob.G11, mem.ξ)
        mem.temp_m̃ .+= prob.t .* prob.G12
        mem.temp_m̃ .^= 2
        B = 4 * prob.G11_norm^2 * maximum(mem.temp_m̃)

        # Set lin_term = (qᵢ(ξ))ᵢ - Gᵀĝ / L
        mem.lin_term .= prob.observations .- mem.temp_m̃
        𝒢ᵀ_rmul!(prob, mem.ξ, mem.temp_m̃, mem.ĝ, mem.temp_m̃2)
        mem.lin_term .-= (1/L) .* mem.temp_m̃

        # compute prox-map
        prox_converged = prox_map(prob, mem.ĝ, mem.lin_term,
            r, L, B, mem.ξ,
            mem.x_new, prox_error, pm_mem; maxiter=500)

        # --------------------
        mem.ξ .= (1 + β) .* mem.x_new .- β .* mem.x
        mem.x .= mem.x_new

        subopt_bound *= (1.0 - α / 2.0)
        prox_error *= (1.0 - α / 2.0)

        if tt % 10 == 0
            if !isnothing(iterate_info) 
                mem.x_full[1:end - 1] = mem.x
                mem.x_full[end] = prob.t
                push_p!(iterate_info, mem.x_full)
            end
            
            mul!(mem.temp_m̃, prob.G11, mem.ξ)
            mem.temp_m̃ .+= prob.t .* prob.G12
            mem.temp_m̃ .^= 2
            mem.temp_m̃ .-= prob.observations
            mem.temp_m̃ .= abs.(mem.temp_m̃)
            sum_errors = sum(mem.temp_m̃)

            mem.sum_errors_hist[1:end-1] .= mem.sum_errors_hist[2:end]
            mem.sum_errors_hist[end] = sum_errors

            distance_bound = sqrt(2 * subopt_bound / μ)

            if (sum_errors <= termination_criteria)
                flag = 1
                break
            elseif (sum_errors > (2 * distance_bound + distance_bound^2) * (prob.Gtop_norm^2) ) || (mem.sum_errors_hist[1] <= mem.sum_errors_hist[10])
                flag = 0
                break
            end
        end
    end

    if flag == -1
        println("Cautious AGD reached max iterations")
    end

    !isnothing(iterate_info) && push_p!(iterate_info, nothing)

    ret .= mem.x
    return flag
end

# Let Gγ = ∑ γᵢ (Aᵢξ + bᵢ) and 𝒢ᵀ its adjoint
function 𝒢_rmul!(prob::QMMP, ξ::Vector{Float64},
    xout::Vector{Float64}, γin::Vector{Float64}, temp::Vector{Float64})
    # ∑ γᵢ (Aᵢξ + bᵢ)
    # = -2(G_11' Diag(γ) (G_11 ξ + t G_12))

    mul!(temp, prob.G11, ξ)
    temp .= γin .* (temp .+ prob.t .* prob.G12)
    mul!(xout, prob.G11', temp)
    xout .*= -2
end
function 𝒢ᵀ_rmul!(prob::QMMP, ξ::Vector{Float64}, γout::Vector{Float64},
    xin::Vector{Float64}, temp::Vector{Float64})
    # <Aᵢξ,x> + <bᵢ, x> = -2 (diag((G_11 x)(t G_12 + G_11 ξ)')

    mul!(temp, prob.G11, ξ)
    temp .+= prob.t .* prob.G12

    mul!(γout, prob.G11, xin)
    γout .*= -2 .* temp
end
