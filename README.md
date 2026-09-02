**NCCL-Tests Framework for SCC Connect '26**

This is a benchmarking framework for NCCL/RCCL - tests, built for the SCC Connect '26 Hackathon.

**Instructions for use**
- Clone the repo
- Clone nccl-tests (or rccl-tests for rocm systems) into external/
- run `cd external && make`
- ...


**In-place vs Out-of-place**
- In-place: the input buffer is reused to hold the result, overwriting its original contents.
- Out-of-place: input and output use separate buffers. The original input remains available.

**NCCL Docs (NVIDEA)**
https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html

**TODO**


2. *The Story*
Does latency increase with ngpus? -- Why?
Do all tests scale the same way? -- Why/Why not?

1. *Multi-node benchmark*
- Allocate multiple nodes.
- Use a test build with MPI support.
- Launch cooperating benchmark processes on those nodes using the cluster's MPI launcher.

Just changing --nodes=1 to --nodes=2 will launch two copies

Will be interesting to compare communication within one node vs communication between nodes across the network