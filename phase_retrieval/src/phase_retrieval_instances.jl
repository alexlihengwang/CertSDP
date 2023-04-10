struct PhaseRetProblem
    n::Int
    m::Int
    G::Matrix{Float64}
    observations::Vector{Float64}
end

function randPhaseRetProblem(n::Int, m::Int)
    xtop = randn(n-1)
    normalize!(xtop)
    x★ = zeros(n)
    x★[1:n-1] .= sqrt(1 - 0.1^2) .* xtop
    x★[end] = 0.1 # generate random instance with one highly correlated observation
    
    G = randn(m,n) ./ sqrt(m)

    # the following normalization can be done as a preprocessing step after seeing the absolute value of an (m+1)th observation
    Gend_norm = norm(G[end,:])
    G[end,:] .= 0
    G[end,end] = Gend_norm

    observations = zeros(m)
    mul!(observations, G, x★)
    observations .^= 2

    return PhaseRetProblem(n, m, G, observations), x★
end