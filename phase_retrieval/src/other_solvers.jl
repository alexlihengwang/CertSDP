function cssdp(problem::PhaseRetProblem, penalty::Float64, R::Float64, lipschitz::Float64, μ::Float64; maxiter::Int=1000000, verbose::Bool=false,
    iterate_info::Union{Iterate_info,Nothing}=nothing,
    savehist::Union{Vector{Int},Nothing}=nothing,
    maxtime::Union{Float64,Nothing}=nothing, termination_criteria::Float64=1e-8)

    time_0 = time()
    n, m = problem.n, problem.m

    # initalize quantities
    γx = zeros(m)
    γy = copy(γx)
    γz = copy(γx)
    γout = copy(γx)
    γ∇ = zeros(m)

    temp_m = zeros(m)
    x = zeros(n)

    α_running = 0
    η_denominator = lipschitz^2

    if isnothing(savehist)
        savehist = [0, 1]
        while savehist[end] < maxiter
            if savehist[end] < 512
                push!(savehist, 2 * savehist[end])
            else
                push!(savehist, savehist[end] + 512)
            end
        end
        push!(savehist, maxiter)
    end 

    # allocate extra working memory
    fo_mem = Fo_mem(n, m)

    # run accelegrad
    verbose && println(".... dual iterations")
    for tt in 0:maxiter
        time_limit_exceeded = (!isnothing(maxtime) && (time() - time_0 >= maxtime))

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

        if tt in savehist || time_limit_exceeded
            !isnothing(iterate_info) && push_d!(problem, iterate_info, nothing)

            function slack_rmul!(y, x)
                mul!(temp_m, problem.G, x)
                temp_m .*= γout
                mul!(y, problem.G', temp_m)
                y .= x .- y
            end
            _, slack_vecs = eig_from_rmul(slack_rmul!, n, 1, :SR)

            x .= slack_vecs[:,1]
            mul!(temp_m, problem.G, x)
            temp_m .^= 2
            α = sqrt(max(0, dot(problem.observations, temp_m)/(norm(temp_m)^2)))
            x .*= α
            normalize!(x)
            if x[end] < 0
                x .*= -1
            end

            if !isnothing(iterate_info)
                push_p!(iterate_info, x)
                push_p!(iterate_info, nothing)
            end

            mul!(temp_m, problem.G, x)
            temp_m .^= 2
            temp_m .= abs.(problem.observations .- temp_m)
            sum(temp_m) <= termination_criteria && break

            verbose && println(".... dual iterations")
        end

        time_limit_exceeded && break
    end

    return x
end

function sketchy_cgal(prob::PhaseRetProblem, α::Float64, 𝒜_norm::Float64;
    β₀::Float64=1.0, 
    maxiter::Int=1000000, maxtime::Union{Float64,Nothing}=nothing,
    iterate_info::Union{Iterate_info,Nothing}=nothing,
    savehist::Union{Vector{Int},Nothing}=nothing, verbose::Bool=false, termination_criteria::Float64=1e-8)
    
    time_0 = time()
    n, m = prob.n, prob.m

    if isnothing(savehist)
        savehist = [0, 1]
        while savehist[end] < maxiter
            if savehist[end] < 512
                push!(savehist, 2 * savehist[end])
            else
                push!(savehist, savehist[end] + 512)
            end
        end
        push!(savehist, maxiter)
    end 

    # allocate all memory
    γz = zeros(m)
    γy = zeros(m)

    sensing_matrix = randn(n,5)
    sketch = zeros(n,5)
    temp_5 = zeros(5)
    temp_sketch = zeros(n,5)
    temp_55 = zeros(5,5)
    # sketch = zeros(n,n)

    temp_n = zeros(n)
    temp_m_1 = zeros(m)
    temp_m_2 = zeros(m)

    x = zeros(n)

    for tt=1:maxiter
        time_limit_exceeded = (!isnothing(maxtime) && (time() - time_0 >= maxtime))
 
        β = β₀ * sqrt(tt + 1.0)
        η = 2.0 / (tt + 1.0)

        temp_m_1 .= γy .+ β .* (γz .+ prob.observations)
        function slack_rmul!(y, x)
            mul!(temp_m_2, prob.G, x)
            temp_m_2 .*= temp_m_1
            mul!(y, prob.G', temp_m_2) 
            y .= x .- y
        end

        slack_vals, slack_vecs = eig_from_rmul(slack_rmul!, n, 1, :SR)

        if slack_vals[1] < 0
            temp_n .= slack_vecs[:, 1]
            normalize!(temp_n)
            mul!(temp_m_1, prob.G, temp_n)
            temp_m_1 .^= 2
            γz .= (1 - η) .* γz .- (α * η) .* temp_m_1

            mul!(temp_5, sensing_matrix', temp_n) 
            for i=1:n
                for j=1:5
                    temp_sketch[i,j] = temp_n[i] * temp_5[j]
                end
            end
            sketch .= (1 - η) .* sketch + (α * η) .* temp_sketch
            # sketch .= (1-η) .* sketch + (α * η) .* (temp_n * temp_n')
        else
            γz .= (1 - η) .* γz
            sketch .*= (1-η)
        end

        step_length = β₀

        γy .= γy .+ step_length .* (γz .+ prob.observations)

        if tt in savehist || time_limit_exceeded
            σ = sqrt(n) * eps(norm(sketch))
            temp_sketch .= sketch .+ σ .* sensing_matrix
            mul!(temp_55, sensing_matrix', temp_sketch)
            sym!(temp_55)
            L = cholesky(temp_55).U
            temp_sketch .= temp_sketch / L
            U, Σ, _ = svd(temp_sketch)
            Λ = max.(0, Σ.^2 .- σ)
            x .= Λ[1] .* U[:,1]

            # vals, vecs = eigs(sketch; nev = 1, which=:LR)
            # x .= vecs[:,1]
            # normalize!(x)
            # x .*= vals
            if x[end] < 0
                x .*= -1
            end

            !isnothing(iterate_info) && push_p!(iterate_info, x)

            mul!(temp_m_1, prob.G, x)
            temp_m_1 .^= 2
            temp_m_1 .= abs.(prob.observations .- temp_m_1)

            sum(temp_m_1) <= termination_criteria && break

        end

        time_limit_exceeded && break
    end

    !isnothing(iterate_info) && push_p!(iterate_info, nothing)

    return x
end


function scs_solve(prob::PhaseRetProblem; verbose::Bool=false, maxtime::Union{Float64,Nothing}=nothing, iterate_info::Union{Iterate_info,Nothing}=nothing)
    options = Vector{Any}()

    push!(options, "verbose" => verbose)
    push!(options, "eps_abs" => 1e-10)
    push!(options, "eps_rel" => 1e-10)
    push!(options, "eps_infeas" => 1e-10)
    !isnothing(maxtime) && push!(options, "time_limit_secs" => maxtime)

    optimizer = optimizer_with_attributes(
        SCS.Optimizer, options...)

    return convexjl_solve(prob, optimizer; iterate_info=iterate_info)
end


function proxSDP_solve(prob::PhaseRetProblem; verbose::Bool=false, maxtime::Union{Float64,Nothing}=nothing, iterate_info::Union{Iterate_info,Nothing}=nothing)
    options = Vector{Any}()

    push!(options, "log_verbose" => verbose)

    push!(options, "tol_gap" => 1e-10)
    push!(options, "tol_feasibility" => 1e-10)
    push!(options, "tol_feasibility_dual" => 1e-10)
    push!(options, "tol_primal" => 1e-10)
    push!(options, "tol_dual" => 1e-10)
    push!(options, "tol_psd" => 1e-10)
    push!(options, "tol_soc" => 1e-10)

    !isnothing(maxtime) && push!(options, "time_limit" => maxtime)

    optimizer = optimizer_with_attributes(
        ProxSDP.Optimizer, options...)

    return convexjl_solve(prob, optimizer; iterate_info=iterate_info)
end


function convexjl_solve(prob::PhaseRetProblem, optimizer::MOI.OptimizerWithAttributes; iterate_info::Union{Iterate_info,Nothing}=nothing)
    n, m = prob.n, prob.m

    model = Model(optimizer)

    @variable(model, Y[i=1:n, j=1:n], PSD)
    @objective(model, Min, tr(Y))

    @constraint(model, [i = 1:m], LinearAlgebra.dot(prob.G[i,:]', Y * prob.G[i,:]) == prob.observations[i])

    optimize!(model)

    Y_val = value.(Y)
    sym!(Y_val)
    vals, vecs = eigs(Y_val; nev = 1, which=:LR)
    x = vecs[:,1]
    normalize!(x)
    x .*= vals[1]
    if x[end] < 0
        x .*= -1
    end

    !isnothing(iterate_info) && push_p!(iterate_info, x)
    !isnothing(iterate_info) && push_p!(iterate_info, nothing)

    return x
end

function burer_monteiro(prob::PhaseRetProblem, γ::Float64, η::Float64, σ::Float64;
    maxiter::Int=100000, iterate_info::Union{Iterate_info,Nothing}=nothing,
    savehist::Union{Vector{Int},Nothing}=nothing, maxtime::Union{Float64,Nothing}=nothing, σ_max = 1e5, termination_criteria::Float64=1e-8)

    time_0 = time()
    n, m = prob.n, prob.m

    if isnothing(savehist)
        savehist=1:maxiter
    end

    errors = zeros(m)
    temp_m = zeros(m)

    function compute_err(x, errors)
        mul!(temp_m, prob.G, x)
        errors .= (temp_m .^ 2) .- prob.observations
    end

    x = randn(n)
    normalize!(x)

    compute_err(x, errors)
    v = norm(errors)^2

    y = zeros(m)
    y_prev = zeros(m)

    for tt in 0:maxiter
        time_limit_exceeded = (!isnothing(maxtime) && (time() - time_0 >= maxtime))

        function f(x̂)
            compute_err(x̂, errors)
            return norm(x̂)^2 - dot(y, errors) +  (σ / 2) * norm(errors)^2
        end

        function g!(g, x̂)
            compute_err(x̂, errors)
            
            mul!(temp_m, prob.G, x̂)
            temp_m .*= (σ .* errors) .- y
            mul!(g, prob.G', temp_m)

            g .= 2 .* (x̂ .+ g)
        end

        # println("lbfgs")
        res = optimize(f, g!, x, LBFGS(), Optim.Options(iterations = 1000000))
        x .= res.minimizer
        compute_err(x, errors)
        v_new = norm(errors)^2

        # println("updating y, v, σ")
        if v_new < η * v || σ > σ_max
            y_prev .= y
            y .-= σ .* errors
            v = v_new
        else
            σ *= γ
        end

        if tt in savehist || time_limit_exceeded
            if x[end] < 0
                x .*= -1
            end

            if !isnothing(iterate_info)
                push_p!(iterate_info, x)
            end

            mul!(temp_m, prob.G, x)
            temp_m .^= 2
            temp_m .= abs.(prob.observations .- temp_m)

            sum(temp_m) <= termination_criteria && break
        end

        time_limit_exceeded && break
    end
end
