## NCCL-Tests Framework for SCC Connect '26

This is a benchmarking framework for NCCL, built for the SCC Connect '26 Hackathon.

It is built to benchmark cross-gpu communication on NVIDEA (NCCL-Test) and AMD (RCCL-Test) systems.

Testing on:
- AMD Developer Cloud (aac6 & aac7)
- BlueBEAR
- NVIDEA Launchpad instance

### Instructions for use
- Clone the repo
- ./build.sh [SYSTEM]
    - Clones the benchmark source set in `config/[SYSTEM].env` into external/ and builds it
    - NVIDIA systems use the [jtc469/nccl-tests](https://github.com/jtc469/nccl-tests/tree/csv-output) `csv-output` branch, as stock nccl-tests can't write csv
    - AMD systems use [ROCm/rccl-tests](https://github.com/ROCm/rccl-tests)
    - Extra args are passed to make, e.g. `./build.sh scc NVCC_GENCODE="-gencode=arch=compute_80,code=sm_80"`
    - Build details (repo, commit, compiler) are saved to `external/build/build-info.txt`

#### Running on slurm
- ./job.sh [SYSTEM] 
- python3 analysis/plot_charts.py [SYSTEM] [JOB_ID]


## NCCL Docs (NVIDEA)
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html

## TODO

#### The Story
Does latency increase with ngpus? -- Why?
Do all tests scale the same way? -- Why/Why not?

#### In-place vs Out-of-place
I currently use out-of-place for all analysis for fair tests between ngpus. Chosen for no particular reason. 
Investigate differences and decide on a way to include them in the analysis
- In-place: the input buffer is reused to hold the result, overwriting its original contents.
- Out-of-place: input and output use separate buffers. The original input remains available.

#### Multi-node benchmark
- Allocate multiple nodes.
- Use a test build with MPI support.
- Launch cooperating benchmark processes on those nodes using the cluster's MPI launcher.

Just changing --nodes=1 to --nodes=2 will just launch two copies, one on each node

Will be interesting to compare communication within one node vs communication between nodes across the network
