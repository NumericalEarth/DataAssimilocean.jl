# Workaround for Julia bug in Base.iterate(::EnvDict)
#
# On multi-node MPI runs (e.g., srun on Perlmutter), Cray MPICH injects
# malformed environment entries (no '=' separator) into remote processes.
# Julia's ENV iterator has a bug where it does `continue` without incrementing
# the index on malformed entries, causing an infinite loop of warnings.
#
# This fix overrides the iterator to skip malformed entries correctly.
# MUST be included before any package that accesses ENV (including MPI.jl).

function Base.iterate(::Base.EnvDict, i=0)
    while true
        env = ccall(:jl_environ, Any, (Int32,), i)
        env === nothing && return nothing
        env = env::String
        m = findfirst('=', env)
        if m === nothing
            i += 1
            continue
        end
        return (Pair{String,String}(env[1:prevind(env, m)], env[nextind(env, m):end]), i+1)
    end
end
