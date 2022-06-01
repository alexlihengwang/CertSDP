function certSDP(qmp::Problem, penalty::Float64, R::Float64, G::Float64;
    maxiter::Int=100000, verbose::Bool=false,
    iterate_info::Union{Iterate_info,Nothing}=nothing,
    savehist::Union{Vector{Int},Nothing}=nothing,
    maxtime::Union{Float64,Nothing}=nothing)
    #= 
    Apply Accelegrad to solve
    
    min_{γ, T} tr(T) + penalty min(0, λ₁(slack(γ)))

    where slack(γ) = M₀ + ∑ᵢ γᵢ Mᵢ - 0 ⊕ T

    at each iteration in savehist, attempt to construct strongly convex qmmp
    and, if successful, run Cautious AGD

    Assume:
        penalty > (‖ X★ ‖² + k)
        R ≥ ‖ γ★ ‖
        G ≥ Lipschitz constant of objective
    =#

    n, k, m = qmp.n, qmp.k, qmp.m

    # initalize Accelegrad quantities
    γx, Tx = zeros(m), zeros(k, k)
    γy, Ty = zeros(m), zeros(k, k)
    γz, Tz = zeros(m), zeros(k, k)
    γ∇, T∇ = zeros(m), zeros(k, k)
    γout, Tout = zeros(m), zeros(k, k)
    α_running = 0
    η_denominator = G^2

    # default setting for savehist is a mixture of guess-and-double and linear
    if isnothing(savehist)
        savehist = [0, 1]
        while savehist[end] < maxiter
            if savehist[end] < 256
                push!(savehist, 2 * savehist[end])
            else
                push!(savehist, savehist[end] + 256)
            end
        end
        push!(savehist, maxiter)
    end

    # pre-allocate memory for all subroutines
    X = zeros(n, k)
    fo_mem = Fo_mem(n, k, m)
    cq_mem = Cq_mem(n, k, m)
    ca_mem = Ca_mem(n, k, m)
    pm_mem = Pm_mem(n, k, m)

    # run accelegrad
    verbose && println("dual iterations")
    for tt in 0:maxiter
        if !isnothing(maxtime) && !isnothing(iterate_info) &&
           (length(iterate_info.d_time) >= 1) && (iterate_info.d_time[end] >= maxtime)
            break
        end

        α = (tt <= 2) ? 1.0 : (tt + 1.0) / 4.0
        τ = 1.0 / α

        γx .= τ .* γz .+ (1 - τ) .* γy
        Tx .= τ .* Tz .+ (1 - τ) .* Ty

        first_order_info(qmp, penalty, γx, Tx, γ∇, T∇, fo_mem)

        η_denominator += α^2 * (norm(γ∇)^2 + norm(T∇)^2)
        η = 4 * R / sqrt(η_denominator)

        γz .+= (α * η) .* γ∇
        Tz .+= (α * η) .* T∇

        z_iter_norm = sqrt(norm(γz)^2 + norm(Tz)^2)
        if z_iter_norm >= R
            γz .*= (R / z_iter_norm)
            Tz .*= (R / z_iter_norm)
        end

        γy .= γx .+ η .* γ∇
        Ty .= Tx .+ η .* T∇

        γout .*= (α_running / (α_running + α))
        Tout .*= (α_running / (α_running + α))
        γout .+= (α / (α_running + α)) .* γy
        Tout .+= (α / (α_running + α)) .* Ty

        sym!(Tx)
        sym!(Ty)
        sym!(Tz)
        sym!(Tout)

        α_running += α

        dual_val = first_order_info(qmp, penalty, γout, Tout, γ∇, T∇, fo_mem)
        !isnothing(iterate_info) && push_d!(iterate_info, dual_val)

        if tt in savehist
            # attempt to construct strongly convex QMMP with bounded condition number
            qmmp_flag, qmmp_data... = construct_qmmp(qmp, γout, X, cq_mem)

            # if successful, solve strongly convex QMMP
            if qmmp_flag
                !isnothing(iterate_info) && push_d!(iterate_info, nothing)

                verbose && println("primal iterations")

                cautious_agd(qmp, γout, X, qmmp_data..., ca_mem, pm_mem;
                    iterate_info=iterate_info) && break

                verbose && println("dual iterations")
            end
        end
    end

    return X
end