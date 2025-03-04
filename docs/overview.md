### Core Purpose
The project aims to solve a significant challenge in zkML (zero-knowledge machine learning) by introducing a distributed approach to proof computation.
### Key Technical Innovation
The main innovation is the concept of "vertical layer circuitization" where:
1. Instead of generating proofs for an entire neural network at once
2. The system splits the neural network into vertical layers
3. Each layer (or group of layers) can be processed independently

### Primary Goals
1. **Distributed Computation**
    - Break down large ML models into manageable pieces
    - Enable parallel processing across multiple machines
    - Support both GPU and non-GPU nodes

2. **Resource Optimization**
    - Reduce RAM requirements through layer splitting
    - Implement caching mechanisms
    - Better manage compute resources

3. **System Flexibility**
    - Configurable number of layers per circuit
    - Adjustable compute allocation
    - Adaptable to different miner capabilities

4. **Performance Enhancement**
    - Reduce overall proof generation time
    - Balance circuit size with distributed computation
    - Optimize for the Bittensor network environment

### Implementation Framework
- Built on top of existing tools:
    - ezkl for circuit generation
    - Halo 2 as the underlying proving system

- Designed to work within Bittensor network constraints
- Focuses on weight and bias matrices, making it applicable to various neural network architectures
