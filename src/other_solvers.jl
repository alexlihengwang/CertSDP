function cssdp(qmp::Problem, penalty::Float64, R::Float64, G::Float64;
    maxiter::Int=100000, verbose::Bool=false, iterate_info::Union{Iterate_info,Nothing}=nothing,
    savehist::Union{Vector{Int},Nothing}=nothing, maxtime::Union{Float64,Nothing}=nothing)

    n, k, m = qmp.n, qmp.k, qmp.m

    # initalize quantities
    γx, Tx = zeros(m), zeros(k, k)
    γy, Ty = zeros(m), zeros(k, k)
    γz, Tz = zeros(m), zeros(k, k)
    γ∇, T∇ = zeros(m), zeros(k, k)

    γout, Tout = zeros(m), zeros(k, k)

    compressed_X = Semidefinite(k)
    X_out = zeros(n, k)

    α_running = 0
    η_denominator = G^2

    infeas_Ms = []
    infeas_rhs = []
    for i = 1:m
        push!(infeas_Ms, qmp.Ms[i])
        push!(infeas_rhs, 0)
    end
    for i = 1:k
        push!(infeas_Ms, sparse([n + i], [n + i], [1], n + k, n + k))
        push!(infeas_rhs, 1)
        for j = i+1:k
            push!(infeas_Ms, sparse([n + i, n + j], [n + j, n + i], [1, 1], n + k, n + k))
            push!(infeas_rhs, 0)
        end
    end
    function compressed_infeas(V, X)
        vcat([dot((V' * infeas_M * V), X) for infeas_M in infeas_Ms]...) - infeas_rhs
    end

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

    # allocate extra working memory
    temp_npk = zeros(n + k)
    temp_nk = zeros(n, k)
    temp_kk = zeros(k, k)
    temp_k = zeros(k)

    # preallocate memory for all subroutines
    fo_mem = Fo_mem(n, k, m)

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
            !isnothing(iterate_info) && push_d!(iterate_info, nothing)

            verbose && println("solving compressed SDP")

            function slack_rmul!(y, x)
                mul!(y, qmp.M₀, x)
                for i = 1:m
                    mul!(temp_npk, qmp.Ms[i], x)
                    y .+= γout[i] .* temp_npk
                end
                mul!(temp_k, Tout, x[n+1:end])
                y[n+1:end] .-= temp_k
            end
            _, slack_vecs = eig_from_rmul(slack_rmul!, n + k, k, :SR)

            problem = minimize(sumsquares(compressed_infeas(slack_vecs, compressed_X)))
            solve!(problem, MOI.OptimizerWithAttributes(SCS.Optimizer,
                "verbose" => verbose,
                "time_limit_secs" => 60,
                "eps_abs" => 1e-13,
                "eps_rel" => 1e-13,
                "eps_infeas" => 1e-13))

            mul!(temp_kk, evaluate(compressed_X), slack_vecs[n+1:end, :]')
            mul!(X_out, slack_vecs[1:n, :], temp_kk)

            if !isnothing(iterate_info)
                push_p!(iterate_info, X_out)
                push_p!(iterate_info, nothing)
            end

            maxqi = 0
            for i = 1:m
                mul!(temp_nk, qmp.As[i], X_out)
                qi = dot(X_out, temp_nk) / 2 + dot(qmp.Bs[i], X_out) + qmp.cs[i]
                maxqi = max(maxqi, abs(qi))
            end
            maxqi <= 1e-13 && break

            verbose && println("dual iterations")
        end
    end

    return X_out
end

function scs_solve(qmp::Problem; verbose::Bool=false, maxtime::Union{Float64,Nothing}=nothing, iterate_info::Union{Iterate_info,Nothing}=nothing)
    options = Vector{Any}()

    push!(options, "verbose" => verbose)
    push!(options, "eps_abs" => 1e-13)
    push!(options, "eps_rel" => 1e-13)
    push!(options, "eps_infeas" => 1e-13)
    !isnothing(maxtime) && push!(options, "time_limit_secs" => maxtime)

    optimizer = optimizer_with_attributes(
        SCS.Optimizer, options...)

    return convexjl_solve(qmp, optimizer; iterate_info=iterate_info)
end


function proxSDP_solve(qmp::Problem; verbose::Bool=false, maxtime::Union{Float64,Nothing}=nothing, iterate_info::Union{Iterate_info,Nothing}=nothing)
    options = Vector{Any}()

    push!(options, "log_verbose" => verbose)

    push!(options, "tol_gap" => 1e-13)
    push!(options, "tol_feasibility" => 1e-13)
    push!(options, "tol_feasibility_dual" => 1e-13)
    push!(options, "tol_primal" => 1e-13)
    push!(options, "tol_dual" => 1e-13)
    push!(options, "tol_psd" => 1e-13)
    push!(options, "tol_soc" => 1e-13)

    !isnothing(maxtime) && push!(options, "time_limit" => maxtime)

    optimizer = optimizer_with_attributes(
        ProxSDP.Optimizer, options...)

    return convexjl_solve(qmp, optimizer; iterate_info=iterate_info)
end


function convexjl_solve(qmp::Problem, optimizer::MOI.OptimizerWithAttributes; iterate_info::Union{Iterate_info,Nothing}=nothing)
    n, k, m = qmp.n, qmp.k, qmp.m

    model = Model(optimizer)

    @variable(model, Y[i=1:n+k, j=1:n+k], PSD)
    @objective(model, Min, LinearAlgebra.dot(qmp.M₀, Y))

    @constraint(model, [i = 1:m], LinearAlgebra.dot(qmp.Ms[i], Y) == 0)
    @constraint(model, [i = 1:k, j = 1:k], Y[n+i, n+j] == (i == j ? 1.0 : 0.0))

    optimize!(model)

    Y_val = value.(Y)
    X_val = Y_val[1:n, n+1:end]

    !isnothing(iterate_info) && push_p!(iterate_info, X_val)
    !isnothing(iterate_info) && push_p!(iterate_info, nothing)

    return X_val
end

function sketchy_cgal(qmp::Problem, α::Float64;
    β₀::Float64=1.0, 
    maxiter::Int=100000, maxtime::Union{Float64,Nothing}=nothing,
    iterate_info::Union{Iterate_info,Nothing}=nothing,
    savehist::Union{Vector{Int},Nothing}=nothing)
    # NB: This is _not_ the algorithm suggested in the SketchyCGAL paper
    # This function implements CGAL
    # but tracks the projection of the matrix iterate onto its top-right corner
    # instead of using the Nystrom sketch
    # This is simpler and allows us to directly reconstruct the QMP solution
    # as opposed to sketching and reconstructing from the sketch
    
    n, m, k = qmp.n, qmp.m, qmp.k
    
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

    # Compute lower bound on 𝒜_norm
    temp = k
    for M in qmp.Ms
        temp += tr(M)^2
    end
    𝒜_norm = sqrt(temp / n)

    # allocate all memory
    γz = zeros(m)
    γy = zeros(m)
    Tz = zeros(k,k)
    Ty = zeros(k,k)

    X = zeros(n, k)

    temp_npk_1 = zeros(n + k)
    temp_npk_2 = zeros(n + k)
    temp_k_1 = zeros(k)
    temp_k_2 = zeros(k)
    temp_kk = zeros(k,k)
    temp_nk = zeros(n, k)

    for tt=1:maxiter
        if !isnothing(maxtime) && !isnothing(iterate_info) &&
            (length(iterate_info.p_time) >= 1) && (iterate_info.p_time[end] >= maxtime)
            break
         end
 
        β = β₀ * sqrt(tt + 1.0)
        η = 2.0 / (tt + 1.0)

        function slack_rmul!(y, x)
            mul!(y, qmp.M₀, x)
            for i = 1:m
                γi = γy[i] + β * γz[i]
                mul!(temp_npk_1, qmp.Ms[i], x)
                y .+= γi .* temp_npk_1
            end
            temp_k_2 .= @view x[n+1:end]
            temp_kk .= Ty .+ β .* (Tz .- sparse(I,k,k))
            mul!(temp_k_1, temp_kk, temp_k_2)
            @view(y[n+1:end]) .+= temp_k_1
        end

        slack_vals, slack_vecs = eig_from_rmul(slack_rmul!, n + k, 1, :SR)

        if slack_vals[1] < 0
            temp_npk_1 .= @view slack_vecs[:, 1]
            normalize!(temp_npk_1)

            for i=1:n
                for j=1:k
                    X[i,j] = (1 - η) * X[i,j] + (α * η) * temp_npk_1[i] * temp_npk_1[n + j]
                end
            end
            for i=1:m
                mul!(temp_npk_2, qmp.Ms[i], temp_npk_1)
                γz[i] = (1 - η) * γz[i] + (α * η) * dot(temp_npk_1, temp_npk_2)
            end
            for i=1:k
                for j=1:k
                    Tz[i,j] = (1 - η) * Tz[i,j] + (α * η) * temp_npk_1[n + i] *  temp_npk_1[n + j]
                end
            end
        else
            X  .= (1 - η) .* X
            γz .= (1 - η) .* γz
            Tz .= (1 - η) .* Tz
        end

        temp_kk .= Tz .- sparse(I, k, k)
        error = norm(γz)^2 + norm(temp_kk)^2
        step_length = min(
            β₀,
            (4 * α^2 * β₀ * 𝒜_norm^2) / ((tt + 1.0)^(1.5) * error)
        )

        γy .= γy .+ step_length .* γz
        Ty .= Ty .+ step_length .* (Tz .- sparse(I, k, k))

        if tt in savehist
            if !isnothing(iterate_info)
                push_p!(iterate_info, X)
            end

            maxqi = 0
            for i = 1:m
                mul!(temp_nk, qmp.As[i], X)
                qi = dot(X, temp_nk) / 2 + dot(qmp.Bs[i], X) + qmp.cs[i]
                maxqi = max(maxqi, abs(qi))
            end
            maxqi <= 1e-13 && break
        end
    end
end