using LinearAlgebra, Arpack
using Dates
using JLD2
using LaTeXStrings
using LaTeXTabulars
using Statistics
using Printf

include("./src/PhaseSolver.jl")
using .PhaseSolver

function generate_instances(n::Int, m::Int, num_tests::Int, testname::String)
    for test_num = 1:num_tests
        println(". Generating instance ", test_num)
        prob, x★ = randPhaseRetProblem(n, m)
        save_object("./results/instances_" * testname * "/instance_" * string(test_num) * ".jld2", Dict("prob" => prob, "x★" => x★))
    end
end

function load_instance(testname::String, test_num::Int)
    problem_data = load_object("./results/instances_" * testname * "/instance_" * string(test_num) * ".jld2")
    prob = problem_data["prob"]::PhaseSolver.PhaseRetProblem
    x★ = problem_data["x★"]::Vector{Float64}
    return prob, x★
end

function test_instance(testname::String, prob::PhaseSolver.PhaseRetProblem, penalty::Float64, R::Float64, lipschitz::Float64,
    x★::Vector{Float64},
    μ::Float64;
    run_certSDP::Bool=true, run_cssdp::Bool=true, run_scs::Bool=true, run_proxSDP::Bool=true, run_sketchy_cgal::Bool=true, run_bm::Bool=true,
    savename::Union{Nothing,String}=nothing, maxiter::Int=1000000, maxtime::Float64=86400.0, verbose::Bool=false, termination_criteria=1e-8)

    pid = getpid()
    cmd = `bash -c "sh track_mem.sh $pid track_mem.tmp"`

    n, m = prob.n, prob.m

    if run_certSDP
        println("... CertSDP")
        certSDP_results = Iterate_info(x★, n, m)
        GC.gc()
        run_cmd = run(cmd, wait=false)
        dualSolver(prob, penalty, R, lipschitz, μ; maxtime=maxtime, iterate_info=certSDP_results, verbose=verbose, maxiter=maxiter, termination_criteria=termination_criteria)
        kill(run_cmd)
        certSDP_mem = parse(Int, read("track_mem.tmp", String))
        if !isnothing(savename)
            save_object("./results/tests_" * testname * "/certSDP_results_" * savename * ".jld2", certSDP_results)
            open("./results/tests_" * testname * "/certSDP_mem_" * savename * ".txt", "w") do io
                write(io, "$certSDP_mem")
            end
        end
    end

    if run_sketchy_cgal
        println("... SketchyCGAL")
        sketchy_cgal_results = Iterate_info(x★, n, m)
        GC.gc()
        run_cmd = run(cmd, wait=false)
        sketchy_cgal(prob, penalty, 0.0; maxtime=maxtime, maxiter=maxiter, iterate_info=sketchy_cgal_results, verbose=verbose, termination_criteria=termination_criteria)
        kill(run_cmd)
        sketchy_cgal_mem = parse(Int, read("track_mem.tmp", String))
        if !isnothing(savename)
            save_object("./results/tests_" * testname * "/sketchy_cgal_results_" * savename * ".jld2", sketchy_cgal_results)
            open("./results/tests_" * testname * "/sketchy_cgal_mem_" * savename * ".txt", "w") do io
                write(io, "$sketchy_cgal_mem")
            end
        end
    end

    if run_cssdp
        println("... CSSDP")
        cssdp_results = Iterate_info(x★, n, m)
        GC.gc()
        run_cmd = run(cmd, wait=false)
        cssdp(prob, penalty, R, lipschitz, μ; maxiter=maxiter, iterate_info=cssdp_results, verbose=verbose, maxtime=maxtime, termination_criteria=termination_criteria)
        kill(run_cmd)
        cssdp_mem = parse(Int, read("track_mem.tmp", String))
        if !isnothing(savename)
            save_object("./results/tests_" * testname * "/cssdp_results_" * savename * ".jld2", cssdp_results)
            open("./results/tests_" * testname * "/cssdp_mem_" * savename * ".txt", "w") do io
                write(io, "$cssdp_mem")
            end
        end
    end

    if run_scs
        println("... SCS")
        scs_results = Iterate_info(x★, n, m)
        GC.gc()
        run_cmd = run(cmd, wait=false)
        scs_solve(prob;
            verbose=verbose, iterate_info=scs_results, maxtime=maxtime)
        kill(run_cmd)
        scs_mem = parse(Int, read("track_mem.tmp", String))
        if !isnothing(savename)
            save_object("./results/tests_" * testname * "/scs_results_" * savename * ".jld2", scs_results)
            open("./results/tests_" * testname * "/scs_mem_" * savename * ".txt", "w") do io
                write(io, "$scs_mem")
            end
        end
    end

    if run_proxSDP
        println("... ProxSDP")
        proxSDP_results = Iterate_info(x★, n, m)
        GC.gc()
        run_cmd = run(cmd, wait=false)
        proxSDP_solve(prob;
            verbose=verbose, iterate_info=proxSDP_results, maxtime=maxtime)
        kill(run_cmd)
        proxSDP_mem = parse(Int, read("track_mem.tmp", String))
        if !isnothing(savename)
            save_object("./results/tests_" * testname * "/proxSDP_results_" * savename * ".jld2", proxSDP_results)
            open("./results/tests_" * testname * "/proxSDP_mem_" * savename * ".txt", "w") do io
                write(io, "$proxSDP_mem")
            end
        end
    end

    if run_bm
        println("... Burer--Monteiro")
        bm_results = Iterate_info(x★, n, m)
        GC.gc()
        run_cmd = run(cmd, wait=false)
        burer_monteiro(prob, sqrt(10), 0.9, 0.25;
            iterate_info=bm_results, maxtime=maxtime, termination_criteria=termination_criteria)
        kill(run_cmd)
        bm_mem = parse(Int, read("track_mem.tmp", String))
        if !isnothing(savename)
            save_object("./results/tests_" * testname * "/bm_results_" * savename * ".jld2", bm_results)
            open("./results/tests_" * testname * "/bm_mem_" * savename * ".txt", "w") do io
                write(io, "$bm_mem")
            end
        end
    end
end

function test_manager(testname::String, num_tests::Int, maxtime::Float64;
    run_certSDP::Bool=true, run_cssdp::Bool=true, run_scs::Bool=true, run_proxSDP::Bool=true, run_sketchy_cgal::Bool=true, run_bm::Bool=true, termination_criteria=1e-8)
    println(". Warming up")
    prob, x★ = load_instance(testname, 1)
    penalty = 10.0 # ≥ norm(x^*)^2
    R = 10.0 * sqrt(n) # ≥ norm(γ^*)
    μ = 0.1

    Gop = svds(prob.G)[1].S[1]
    lipschitz = norm(prob.observations) + (penalty + 2) * Gop^2

    test_instance(testname, prob, penalty, R, lipschitz, x★, μ;
        savename=nothing, maxiter=10, maxtime=10.0,
        run_certSDP=run_certSDP, run_cssdp=run_cssdp, run_scs=run_scs, run_proxSDP=run_proxSDP, run_sketchy_cgal=run_sketchy_cgal, run_bm=run_bm, termination_criteria=termination_criteria)

    println(". Actual tests")
    for test_num = 1:num_tests
        println(". Loading instance ", test_num)
        prob, x★ = load_instance(testname, test_num)
        penalty = 10.0 # ≥ norm(x^*)^2
        R = 10.0 * sqrt(n) # ≥ norm(γ^*)
        μ = 0.1
        Gop = svds(prob.G)[1].S[1]
        lipschitz = norm(prob.observations) + (penalty + 2) * Gop^2

        test_instance(testname, prob, penalty, R, lipschitz, x★, μ;
            savename=string(test_num), maxtime=maxtime,
            run_certSDP=run_certSDP, run_cssdp=run_cssdp, run_scs=run_scs, run_proxSDP=run_proxSDP,run_sketchy_cgal=run_sketchy_cgal, run_bm=run_bm,termination_criteria=termination_criteria)
    end
end

function make_table(test_name, run_cssdp, run_scs, run_proxSDP, run_sketchy_cgal, run_bm)
    certSDP_p_times = []
    certSDP_p_sqdists = []
    certSDP_mems = []

    cssdp_p_times = []
    cssdp_p_sqdists = []
    cssdp_mems = []

    scs_p_times = []
    scs_p_sqdists = []
    scs_mems = []

    proxSDP_p_times = []
    proxSDP_p_sqdists = []
    proxSDP_mems = []

    sketchy_cgal_p_times = []
    sketchy_cgal_p_sqdists = []
    sketchy_cgal_mems = []

    bm_p_times = []
    bm_p_sqdists = []
    bm_mems = []

    test_num = 1
    while true
        filename = "./results/tests_" * test_name * "/certSDP_mem_" * string(test_num) * ".txt"
        isfile(filename) || break
        certSDP_mem = parse(Int, read(filename, String))
        push!(certSDP_mems, certSDP_mem * 1e-3)
        
        filename = "./results/tests_" * test_name * "/certSDP_results_" * string(test_num) * ".jld2"
        isfile(filename) || break
        certSDP_results = load_object(filename)
        push!(certSDP_p_times, max(certSDP_results.p_time[end], certSDP_results.d_time[end]))
        push!(certSDP_p_sqdists, certSDP_results.p_sqdist[end - 1])

        if run_cssdp
            filename = "./results/tests_" * test_name * "/cssdp_mem_" * string(test_num) * ".txt"
            isfile(filename) || break
            cssdp_mem = parse(Int, read(filename, String))
            push!(cssdp_mems, cssdp_mem * 1e-3)
            
            filename = "./results/tests_" * test_name * "/cssdp_results_" * string(test_num) * ".jld2"
            isfile(filename) || break
            cssdp_results = load_object(filename)
            push!(cssdp_p_times, max(cssdp_results.p_time[end], cssdp_results.d_time[end]))
            push!(cssdp_p_sqdists, cssdp_results.p_sqdist[end - 1])
        end

        if run_scs
            filename = "./results/tests_" * test_name * "/scs_mem_" * string(test_num) * ".txt"
            isfile(filename) || break
            scs_mem = parse(Int, read(filename, String))
            push!(scs_mems, scs_mem * 1e-3)
            
            filename = "./results/tests_" * test_name * "/scs_results_" * string(test_num) * ".jld2"
            isfile(filename) || break
            scs_results = load_object(filename)
            push!(scs_p_times, scs_results.p_time[end])
            push!(scs_p_sqdists, scs_results.p_sqdist[end - 1])
        end

        if run_proxSDP
            filename = "./results/tests_" * test_name * "/proxSDP_mem_" * string(test_num) * ".txt"
            isfile(filename) || break
            proxSDP_mem = parse(Int, read(filename, String))
            push!(proxSDP_mems, proxSDP_mem * 1e-3)
            
            filename = "./results/tests_" * test_name * "/proxSDP_results_" * string(test_num) * ".jld2"
            isfile(filename) || break
            proxSDP_results = load_object(filename)
            push!(proxSDP_p_times, proxSDP_results.p_time[end])
            push!(proxSDP_p_sqdists, proxSDP_results.p_sqdist[end - 1])
        end

        if run_sketchy_cgal
            filename = "./results/tests_" * test_name * "/sketchy_cgal_mem_" * string(test_num) * ".txt"
            isfile(filename) || break
            sketchy_cgal_mem = parse(Int, read(filename, String))
            push!(sketchy_cgal_mems, sketchy_cgal_mem * 1e-3)
            
            filename = "./results/tests_" * test_name * "/sketchy_cgal_results_" * string(test_num) * ".jld2"
            isfile(filename) || break
            sketchy_cgal_results = load_object(filename)
            push!(sketchy_cgal_p_times, sketchy_cgal_results.p_time[end])
            push!(sketchy_cgal_p_sqdists, sketchy_cgal_results.p_sqdist[end - 1])
        end
    

        if run_bm
            filename = "./results/tests_" * test_name * "/bm_mem_" * string(test_num) * ".txt"
            isfile(filename) || break
            bm_mem = parse(Int, read(filename, String))
            push!(bm_mems, bm_mem * 1e-3)
            
            filename = "./results/tests_" * test_name * "/bm_results_" * string(test_num) * ".jld2"
            isfile(filename) || break
            bm_results = load_object(filename)
            push!(bm_p_times, bm_results.p_time[end])
            push!(bm_p_sqdists, bm_results.p_sqdist[end - 1])
        end

        test_num += 1
    end


    table_rows = []

    function num_or_dash(a)
        if isnan(a)
            return "-"
        else
            return @sprintf("\\num{%.1e}",a)
        end
    end

    push!(table_rows, ["CertSDP",
                num_or_dash(mean(certSDP_p_times)),
                num_or_dash(std(certSDP_p_times)),
                num_or_dash(mean(certSDP_p_sqdists)),
                num_or_dash(std(certSDP_p_sqdists)),
                num_or_dash(mean(certSDP_mems)),
                num_or_dash(std(certSDP_mems))])
    run_cssdp && push!(table_rows, ["CSSDP",
                num_or_dash(mean(cssdp_p_times)),
                num_or_dash(std(cssdp_p_times)),
                num_or_dash(mean(cssdp_p_sqdists)),
                num_or_dash(std(cssdp_p_sqdists)),
                num_or_dash(mean(cssdp_mems)),
                num_or_dash(std(cssdp_mems))])
    run_sketchy_cgal && push!(table_rows, ["SketchyCGAL",
                num_or_dash(mean(sketchy_cgal_p_times)),
                num_or_dash(std(sketchy_cgal_p_times)),
                num_or_dash(mean(sketchy_cgal_p_sqdists)),
                num_or_dash(std(sketchy_cgal_p_sqdists)),
                num_or_dash(mean(sketchy_cgal_mems)),
                num_or_dash(std(sketchy_cgal_mems))])
    run_proxSDP && push!(table_rows, ["ProxSDP",
                num_or_dash(mean(proxSDP_p_times)),
                num_or_dash(std(proxSDP_p_times)),
                num_or_dash(mean(proxSDP_p_sqdists)),
                num_or_dash(std(proxSDP_p_sqdists)),
                num_or_dash(mean(proxSDP_mems)),
                num_or_dash(std(proxSDP_mems))])
    run_scs && push!(table_rows, ["SCS",
                num_or_dash(mean(scs_p_times)),
                num_or_dash(std(scs_p_times)),
                num_or_dash(mean(scs_p_sqdists)),
                num_or_dash(std(scs_p_sqdists)),
                num_or_dash(mean(scs_mems)),
                num_or_dash(std(scs_mems))])
    run_bm && push!(table_rows, ["Burer--Monteiro",
                num_or_dash(mean(bm_p_times)),
                num_or_dash(std(bm_p_times)),
                num_or_dash(mean(bm_p_sqdists)),
                num_or_dash(std(bm_p_sqdists)),
                num_or_dash(mean(bm_mems)),
                num_or_dash(std(bm_mems))])


    latex_tabular("./results/tests_" * test_name * "/table.tex",
        Tabular("lllllll"),
        [Rule(:top),
            ["Algorithm", "time (s)", "std.", "\$\\norm{x - x^*}_2^2\$", "std.", "memory (MB)", "std."],
            Rule(:mid),
            table_rows...,
            Rule(:bottom)
        ])
end

# ======

testname = "example"

n = 100
m = 5 * n
termination_criteria=1e-7
num_tests = 5

run_certSDP = true 
run_cssdp = true
run_scs = true
run_proxSDP = true
run_sketchy_cgal = true
run_bm = true

maxtime = 500.0

println("\nGenerating instances " * testname)
Base.Filesystem.mkdir("./results/tests_" * testname)
Base.Filesystem.mkdir("./results/instances_" * testname)
open("./results/tests_" * testname * "/notes.txt", "w") do io
    write(io, "n = $n, m = $m, num_tests = $num_tests")
end;

generate_instances(n, m, num_tests, testname)

println("\nTesting instances " * testname)
test_manager(testname, num_tests, maxtime;
    run_certSDP=run_certSDP, run_cssdp=run_cssdp, run_scs=run_scs, run_proxSDP=run_proxSDP, run_sketchy_cgal=run_sketchy_cgal, run_bm=run_bm, termination_criteria=termination_criteria)

println("Generating table")
make_table(testname, run_cssdp, run_scs, run_proxSDP, run_sketchy_cgal, run_bm)