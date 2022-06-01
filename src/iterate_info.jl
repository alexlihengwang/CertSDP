struct Iterate_info
    X★::Matrix{Float64}
    opt::Float64
    p_time::Vector{Float64}
    p_sqdist::Vector{Union{Float64,Nothing}}
    p_iter::Vector{Int}
    d_time::Vector{Float64}
    d_subopt::Vector{Union{Float64,Nothing}}
    d_iter::Vector{Int}
    start_time::Float64
    curr_iter::Vector{Int}
    temp_nk::Matrix{Float64}
end

Iterate_info(X★::Matrix{Float64}, opt::Float64, n::Int64, k::Int64) = Iterate_info(
    X★, opt, [], [], [], [], [], [], time(), [1], zeros(n, k))

function push_p!(iter_info::Iterate_info, X::Union{Matrix{Float64},Nothing})
    push!(iter_info.p_time, time() - iter_info.start_time)
    push!(iter_info.p_iter, iter_info.curr_iter[1])
    iter_info.curr_iter[1] += 1

    if !isnothing(X)
        iter_info.temp_nk .= X .- iter_info.X★
        push!(iter_info.p_sqdist, norm(iter_info.temp_nk)^2)
    else
        push!(iter_info.p_sqdist, nothing)
    end
end

function push_d!(iter_info::Iterate_info, val::Union{Float64,Nothing})
    push!(iter_info.d_time, time() - iter_info.start_time)
    push!(iter_info.d_iter, iter_info.curr_iter[1])
    iter_info.curr_iter[1] += 1
    if !isnothing(val)
        push!(iter_info.d_subopt, iter_info.opt - val)
    else
        push!(iter_info.d_subopt, nothing)
    end
end
