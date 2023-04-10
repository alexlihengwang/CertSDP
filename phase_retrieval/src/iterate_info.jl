struct Iterate_info
    x★::Vector{Float64}
    opt::Float64
    p_time::Vector{Float64}
    p_sqdist::Vector{Union{Float64,Nothing}}
    p_iter::Vector{Int}
    d_time::Vector{Float64}
    d_subopt::Vector{Union{Float64,Nothing}}
    d_iter::Vector{Int}
    start_time::Float64
    curr_iter::Vector{Int}
    temp_n::Vector{Float64}
    temp_m::Vector{Float64}
end

Iterate_info(x★::Vector{Float64}, n::Int64, m::Int64) = Iterate_info(
    x★, norm(x★)^2, [], [], [], [], [], [], time(), [1], zeros(n), zeros(m))

function push_p!(iter_info::Iterate_info, x::Union{Vector{Float64},Nothing})
    push!(iter_info.p_time, time() - iter_info.start_time)
    push!(iter_info.p_iter, iter_info.curr_iter[1])
    iter_info.curr_iter[1] += 1

    if !isnothing(x)
        iter_info.temp_n .= x .- iter_info.x★
        push!(iter_info.p_sqdist, norm(iter_info.temp_n)^2)
    else
        push!(iter_info.p_sqdist, nothing)
    end
end

function push_d!(prob::PhaseRetProblem, iter_info::Iterate_info, γ::Union{Vector{Float64},Nothing})
    push!(iter_info.d_time, time() - iter_info.start_time)
    push!(iter_info.d_iter, iter_info.curr_iter[1])
    iter_info.curr_iter[1] += 1

    if !isnothing(γ)
        function slack_rmul!(y, x)
            mul!(iter_info.temp_m, prob.G, x)
            iter_info.temp_m .*= γ
            mul!(y, prob.G', iter_info.temp_m)
            y .= x .- y
        end

        slack_vals, _ = eig_from_rmul(slack_rmul!, prob.n, 1, :LR)
        
        push!(iter_info.d_subopt, iter_info.opt -
            dot(γ, prob.observations) -
            iter_info.opt * min(0, slack_vals[1])
            )
    else
        push!(iter_info.d_subopt, nothing)
    end
end
