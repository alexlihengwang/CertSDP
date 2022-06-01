struct Fo_mem
    npk_1::Vector{Float64}
    npk_2::Vector{Float64}
    k_1::Vector{Float64}
    k_2::Vector{Float64}
end

function Fo_mem(n::Int64, k::Int64, m::Int64)
    Fo_mem(
        zeros(n + k),
        zeros(n + k),
        zeros(k),
        zeros(k))
end

function first_order_info(qmp::Problem, penalty::Float64,
    γ::Vector{Float64}, T::Matrix{Float64}, γ∇::Vector{Float64},
    T∇::Matrix{Float64}, mem::Fo_mem)
    #= 
        return tr(T) + penalty min(0, λ₁(slack(γ)))
        store gradient in (γ, T) at (γ∇, T∇)
    =#
    
    
    val = tr(T)
    γ∇ .= 0
    T∇ .= 0
    for i = 1:qmp.k
        T∇[i, i] = 1.0
    end

    function slack_rmul!(y, x)
        mul!(y, qmp.M₀, x)
        for i = 1:qmp.m
            mul!(mem.npk_1, qmp.Ms[i], x)
            y .+= γ[i] .* mem.npk_1
        end
        mem.k_2 .= @view x[qmp.n+1:end]
        mul!(mem.k_1, T, mem.k_2)
        @view(y[qmp.n+1:end]) .-= mem.k_1
    end

    slack_vals, slack_vecs = eig_from_rmul(slack_rmul!, qmp.n + qmp.k, 1, :SR)

    if slack_vals[1] < 0
        mem.npk_1 .= @view slack_vecs[:, 1]

        normalize!(mem.npk_1)

        mem.k_1 .= @view mem.npk_1[qmp.n+1:end]

        for i = 1:qmp.m::Int
            mul!(mem.npk_2, qmp.Ms[i], mem.npk_1)
            γ∇[i] += penalty * dot(mem.npk_1, mem.npk_2)
        end

        for i = 1:qmp.k
            for j = 1:qmp.k
                T∇[i, j] -= penalty * mem.k_1[i] * mem.k_1[j]
            end
        end

        val += penalty * slack_vals[1]
    end

    return val
end