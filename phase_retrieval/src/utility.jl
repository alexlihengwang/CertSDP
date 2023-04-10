function eig_from_rmul(rmul!::Function, n::Int, howmany::Int, which::Symbol)
    op = LinearMap{Float64}(rmul!, n; issymmetric=true, ismutating=true)
    vals, vecs, _, _, _, _ = eigs(op; nev=howmany, which=which, ncv=100, maxiter=3000)
    return vals, vecs
end

function sym!(T::Matrix{Float64})
    n = size(T)[1]
    for i = 1:n-1
        for j = i+1:n
            T[i, j] = T[j, i]
        end
    end
end
