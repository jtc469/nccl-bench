## NCCL-Tests Framework for SCC Connect '26

This is a benchmarking framework for NCCL, built for the SCC Connect '26 Hackathon.

It is built to benchmark cross-gpu communication on NVIDEA (NCCL-Test) and AMD (RCCL-Test) systems.

Testing on:
- AMD Developer Cloud
- BlueBEAR
- ... 

### Instructions for use
- Clone the repo
- Clone nccl-tests (or rccl-tests for rocm systems) into external/
- run `cd external && make`
- ...

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

Just changing --nodes=1 to --nodes=2 will launch two copies

Will be interesting to compare communication within one node vs communication between nodes across the network
