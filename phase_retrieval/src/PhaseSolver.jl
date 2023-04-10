module PhaseSolver

export PhaseRetProblem, Iterate_info
export randPhaseRetProblem
export dualSolver
export cssdp, sketchy_cgal, scs_solve, proxSDP_solve, burer_monteiro

using LinearAlgebra, LinearMaps, Arpack, KrylovKit
using Convex, MosekTools, SCS
using MathOptInterface
using ProxSDP
const MOI = MathOptInterface
using JuMP
using Optim

include("./phase_retrieval_instances.jl")
include("./iterate_info.jl")
include("./construct_qmmp.jl")
include("./prox_map.jl")
include("./cautious_agd.jl")
include("./dual_solver.jl")

include("./utility.jl")

include("./other_solvers.jl")



end