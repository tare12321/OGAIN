# OGAIN

**OGAIN** is an open, vendor-independent platform for GPU-resident execution of AI and numerical workloads.

The project explores how tensor operations, computation graphs, memory management, and execution backends can be separated into a portable software architecture. The main goal is to reduce unnecessary CPU-GPU data transfers and enable reproducible performance testing across different hardware and backend implementations.

## Main idea

Many AI and HPC workflows depend heavily on specific vendor ecosystems. OGAIN is designed to allow the same computation graph to be executed through different backends, such as:

- CPU reference backend
- Vulkan Compute
- CUDA / cuBLAS / cuDNN
- ROCm / Vulkan paths 

## Current status

Parts of the software architecture and prototype are already under development, including tensor abstractions, computation graph concepts, backend separation, and benchmark planning.

The next important step is hardware validation. 
.
