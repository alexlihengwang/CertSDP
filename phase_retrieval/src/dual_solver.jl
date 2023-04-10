function dualSolver(problem::PhaseRetProblem, penalty::Float64, R::Float64, lipschitz::Float64, μ::Float64; maxiter::Int=100000, verbose::Bool=false,
    iterate_info::Union{Iterate_info,Nothing}=nothing,
    savehist::Union{Vector{Int},Nothing}=nothing,
    maxtime::Union{Float64,Nothing}=nothing, termination_criteria::Float64=1e-8)
    #= 
    Apply Accelegrad to solve
    
    max_{γ} dot(obs,γ) + penalty max(0, λ₁(slack(γ))) + max(0, λ₁+λ₂(slack(γ)) - μ̂)

    where slack(γ) = I - G' Diag(γ) G

    at each iteration in savehist, attempt to construct strongly convex qmmp
    and, if successful, run Cautious AGD

    Assume:
        penalty > (‖ X★ ‖²)
        R ≥ ‖ γ★ ‖
        lipschitz ≥ Lipschitz constant of objective
    =#
    n, m = problem.n, problem.m

    time_0 = time()

    γx = zeros(m)
    γy = copy(γx)
    γz = copy(γx)
    γout = copy(γx)
    γ∇ = zeros(m)

    γupper = zeros(m - 1)

    α_running = 0
    η_denominator = lipschitz^2

    # initialize memory
    x = zeros(n - 1)
    fo_mem = Fo_mem(n, m)
    ca_mem = Ca_mem(n, m)
    pm_mem = Pm_mem(n, m)
    cq_mem = Cq_mem(n, m)
    qmmp = QMMP(problem)
    
    # default setting for savehist is guess-and-double
    if isnothing(savehist)
        savehist = [0, 1]
        while savehist[end] < maxiter
            if savehist[end] < 2048
                push!(savehist, 2 * savehist[end])
            else
                push!(savehist, savehist[end] + 2048)
            end
        end
        push!(savehist, maxiter)
    end

    verbose && println(".... dual iterations")
    for tt in 0:maxiter
        time_limit_exceed = (!isnothing(maxtime) && (time() - time_0 >= maxtime))

        α = (tt <= 2) ? 1.0 : (tt + 1.0) / 4.0
        τ = 1.0 / α

        γx .= τ .* γz .+ (1 - τ) .* γy

        first_order_info(problem, penalty, μ, γx, γ∇, fo_mem)

        η_denominator += α^2 * norm(γ∇)^2
        η = 4 * R / sqrt(η_denominator)

        γz .+= (α * η) .* γ∇

        γy .= γx .+ η .* γ∇

        γout .= (α / (α_running + α)) .* γy .+
            (α_running / (α_running + α)) .* γout

        α_running += α

        (tt % 100 == 0 || tt in savehist) && !isnothing(iterate_info) && push_d!(problem, iterate_info, γout)

        if tt in savehist || time_limit_exceed
            # attempt to construct strongly convex QMMP with bounded condition number
            γupper .= γout[1:m - 1]
            qmmp_flag, qmmp_data... = construct_qmmp(γupper, x, qmmp, cq_mem)

            # if successful, solve strongly convex QMMP
            if qmmp_flag
                !isnothing(iterate_info) && push_d!(problem, iterate_info, nothing)

                verbose && println(".... primal iterations")

                ca_flag = cautious_agd(qmmp, γupper, x, qmmp_data..., ca_mem, pm_mem; iterate_info=iterate_info, termination_criteria=termination_criteria)

                if ca_flag == 1
                    break
                end

                verbose && println(".... dual iterations")
            end
        end

        time_limit_exceed && break
    end
    return γout
end

struct Fo_mem
    temp_n::Vector{Float64}
    temp_m::Vector{Float64}
end

function Fo_mem(n::Int64,m::Int64)
    Fo_mem(zeros(n),zeros(m))
end

function first_order_info(problem::PhaseRetProblem, penalty::Float64, μ::Float64, γ::Vector{Float64}, γ∇::Vector{Float64}, fo_mem::Fo_mem)
    # <b, gamma> + penalty lambda_min(I - G' diag(gamma) G)_- + (lambda_1+2 (I - G' diag(gamma) G) - \mu)_-

    val = dot(problem.observations, γ)
    γ∇ .= problem.observations

    function slack_rmul!(y, x)
        mul!(fo_mem.temp_m, problem.G, x)
        fo_mem.temp_m .*= γ
        mul!(y, problem.G', fo_mem.temp_m)
        y .= x .- y
    end

    slack_vals, slack_vecs = eig_from_rmul(slack_rmul!, problem.n, 2, :SR)

    if slack_vals[1] < 0 
        val += penalty * slack_vals[1]

        fo_mem.temp_n .= slack_vecs[:,1]
        normalize!(fo_mem.temp_n)
        mul!(fo_mem.temp_m, problem.G, fo_mem.temp_n)
        fo_mem.temp_m .^= 2
        γ∇ .-= penalty .* fo_mem.temp_m
    end

    if slack_vals[1] + slack_vals[2] < μ
        val += slack_vals[1] + slack_vals[2] - μ

        fo_mem.temp_n .= slack_vecs[:,1]
        normalize!(fo_mem.temp_n)
        mul!(fo_mem.temp_m, problem.G, fo_mem.temp_n)
        fo_mem.temp_m .^= 2
        γ∇ .-= fo_mem.temp_m

        fo_mem.temp_n .= slack_vecs[:,2]
        normalize!(fo_mem.temp_n)
        mul!(fo_mem.temp_m, problem.G, fo_mem.temp_n)
        fo_mem.temp_m .^= 2
        γ∇ .-= fo_mem.temp_m
    end

    return val
end


