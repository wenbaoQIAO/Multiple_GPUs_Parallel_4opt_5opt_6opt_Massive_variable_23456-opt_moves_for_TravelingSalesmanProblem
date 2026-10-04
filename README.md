# MultiGPU-k-opt-TSP
Multi-GPU parallel complete k-opt local search for the Travelling Salesman Problem (TSP)

This repository hosts the open-source implementation accompanying the paper:
> **Breaking TSP local search barriers: scalable multi-GPU parallelisation of 4-opt, 5-opt, 6-opt and hybrid variable K-opt**

## Overview
The Travelling Salesman Problem (TSP) is a canonical NP-hard combinatorial optimisation problem. The k-opt family (2-opt, 3-opt, 4-opt, 5-opt, 6-opt and variable λ-opt) forms the core local-search mechanism for tour improvement and is the fundamental building block of state-of-the-art LKH heuristic solvers.

High-order complete k-opt (4-opt, 5-opt, 6-opt) suffers from factorial computational complexity. Serial complete k-opt may take tens of hours or multiple days for moderate-scale TSP instances. Existing GPU-accelerated TSP local search tools are mostly limited to 2-opt and 3-opt, while open-source implementations for **complete higher-order k-opt and variable λ-opt running on multi-GPU hardware** remain scarce.

MultiGPU-k-opt-TSP is a C/C++ + CUDA toolkit designed for brute-force complete k-opt local search. It targets a single workstation with up to four GPUs and does **not require MPI for task partitioning**. This implementation omits nearest-neighbour pruning: all edge combinations are exhaustively evaluated over the global tour to provide a ground-truth complete k-opt baseline.

## Key Features
- CUDA kernels for complete 2-opt, 3-opt, 4-opt, 5-opt and 6-opt local search, covering all valid edge-reconnection schemes
- Multi-GPU workload partitioning on a single node, aggregating improving candidate moves from all available GPU devices
- Linear-time non-conflicting move selection to avoid tour corruption when applying multiple simultaneous λ-opt moves
- Two optimisation modes: fixed-k iterative k-opt; hybrid increasing variable λ-opt (λ from 2 to 6)
- Native reader for standard TSPLIB `.tsp` instance files; exports optimised tour records and runtime statistics
- Fully modular design: TSP IO module, GPU kernel library, multi-GPU scheduler, conflict resolver and main search controller

## Software Architecture
The toolkit is organised into five core modules:
1. **TSP IO module**: Load standard TSPLIB `.tsp` instances, initialise tour data structures, export final optimised tour and timing logs.
2. **Single-GPU k-opt kernel module**: CUDA kernels for full enumeration of k-edge combinations and tour gain evaluation.
3. **Multi-GPU task partition module**: Distribute edge checking workload evenly across available GPUs and collect candidate improving moves.
4. **Non-conflicting move selection module**: Filter compatible simultaneous moves to prevent tour damage caused by overlapping edge modifications.
5. **Iterative local-search controller**: Execute fixed-k optimisation or hybrid variable λ-opt until the tour reaches λ-optimal convergence.

### Main Workflow
1. Load TSPLIB instance and initial tour
2. While improving k-opt / variable λ-opt moves exist:
   - Distribute edge-combination checking tasks to available GPUs
   - Collect all candidate improving moves
   - Filter non-conflicting moves in linear time
   - Update global tour with the selected compatible moves
3. Write optimised tour, iteration statistics and wall-clock timing metrics to output files

## System Requirements
- NVIDIA GPU(s) with CUDA compute capability ≥ 7.0
- CUDA Toolkit (tested with 11.x / 12.x)
- C++ compiler supporting C++17
- Linux / Windows (tested on Windows with CUDA + MSVC or MinGW)
- Single node only, up to 4 GPUs (MPI is not required)

## Compilation
This project is developed with Qt Creator, requiring prebuilt static Qt libraries, Boost library and NVIDIA CUDA Toolkit. The compilation workflow contains two stages: first build the static library, then compile the test executable.

### Stage 1: Build static library `libCalculateur_KOPT.lib`
1. Open Qt Creator. Load the project file `MultiGPU_2_3_4_5_6_optTsp/lib/lib.pro`.
2. Configure the Qt kit with **statically compiled Qt**, Boost headers/libraries and CUDA environment.
3. Build the project. This generates the static library `libCalculateur_KOPT.lib`.

### Stage 2: Build test executable
1. In Qt Creator, open the test project `MultiGPU_2_3_4_5_6_optTsp/test/test.pro`.
2. Ensure the build target links against the previously compiled `libCalculateur_KOPT.lib`, Boost and CUDA runtime libraries.
3. Build the test project to get the executable binary.

> **Working directory configuration**:
> Set the program run working directory in Qt Creator to:
> `MultiGPU_2_3_4_5_6_optTsp/test_worldTsp/dataNational`

## Usage & Demo Example
Launch the compiled test executable from the configured working directory.
The executable accepts four command-line arguments:
`[executable] input.tsp output_tour output_stat config.cfg`

Algorithm mode, k-opt variant and GPU count are defined in `config.cfg`.

```cfg
# config.cfg example
[coalition_global_param]
# choix du mode de fonctionnement
# 2:2-opt; 3:3-opt; 4:4-opt; 5:5-opt; 6:6-opt; 7:variable k-opt(k=2,3,4,5,6); 8:iterative increasing k-optimal from 2-6
functionModeChoice = 8  //iterative increasing k-optimal from 2-6

### Demo Run on `mu1979.tsp` (TSPLIB Benchmark)
calculateur.exe mu1979.tsp output output config.cfg
```

- Working directory: `MultiGPU_2_3_4_5_6_optTsp/test_worldTsp/dataNational`
- Input: `lu980.tsp` (standard TSPLIB instance)
- Output 1: `output` — final optimised tour file
- Output 2: `output` — iteration logs, wall-clock time and speedup metrics
- Configuration: `config.cfg` controls algorithm parameters and GPU resources

After execution, the output files record iteration count, runtime and speedup relative to serial complete k-opt.

## Research Impact
This toolkit fills a critical gap in open-source combinatorial optimisation software for TSP local search research and lays a foundation toward fully multi-GPU parallel LKH solvers.
- **Research benchmarking**: Reproducible baseline for complete k-opt without pruning, enabling quantitative comparison against heuristic pruned k-opt and LKH variants.
- **Algorithm prototyping**: Users can modify CUDA kernels, edge-reconnection rules and conflict resolution logic to explore novel multi-move k-opt heuristics.
- **Education**: Suitable for graduate students studying metaheuristics, to understand factorial complexity scaling and GPU acceleration for high-order local search.
- **Hybrid solver polishing**: Integrate as a post-processing refinement module. Fast heuristic solvers generate near-optimal tours; MultiGPU-k-opt-TSP then applies high-order complete k-opt polishing to obtain λ-optimal solutions within feasible wall-clock time.

Practical application scenarios include logistics routing, PCB drilling path planning and symmetric production scheduling problems reducible to symmetric TSP.

## Citation
If you use this code in your research, please cite our publication:
```bibtex
@article{qiao2026breaking,
	title={Breaking TSP local search barriers: scalable multi-GPU parallelisation of 4-opt, 5-opt, 6-opt and hybrid variable $\lambda$-opt},
	author={Qiao, Wen-Bao and Cr{\'e}put, Jean-Charles and Wang, Nan and Meng, Kun and Mansouri, Abdelkhalek},
	journal={Expert Systems with Applications},
	volume={296},
	pages={129110},
	year={2026},
	publisher={Elsevier}
}
```
> Update the `journal` field after formal acceptance.

## License
This project is released under the Attribution-NonCommercial 4.0 International(LICENSE) file available in this repository.

## Notes and Limitations
- This implementation performs brute-force complete k-opt **without nearest-neighbour filtering**.
- Only symmetric TSP instances from TSPLIB are supported.
- The software is designed for single-node workstations with at most four GPUs.
- The repository includes demo TSPLIB instances (e.g., `mu1979.tsp`) for validation and benchmarking.
```
