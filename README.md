<div align="center">

# **kubz: A Distributed zkML Project**

[![GitHub issues](https://img.shields.io/github/issues/yourusername/kubz.svg)](https://github.com/yourusername/kubz/issues) [![License](https://img.shields.io/github/license/yourusername/kubz.svg)](LICENSE)

### Zero-Knowledge Machine Learning for Distributed Networks

[Documentation](https://yourdocs.link/) • [Twitter](https://twitter.com/yourproject) • [Demo](https://demo.yourproject.link) • [Stats](https://stats.yourproject.link)

</div>

kubz is a project designed to harness distributed zero-knowledge machine learning (DzkML) to ensure secure, verifiable AI across decentralized networks. Our system converts AI models into unique digital fingerprints—proofs that validate the model's inference process, ensuring both integrity and performance.

---

## Architecture Overview

For more detailed information about the architecture and approach, please refer to the [architecture overview](docs/overview.md) file.

### Miners and Validators

Miners and validators are two key roles in kubz's DzkML architecture, acting in tandem to maintain the integrity and
efficiency of the system.

Miners are responsible for processing the machine learning model's inference process to generate zk-proofs. They utilize
the input data provided by validators, run the necessary computations, and return both the witness (model output) and
its corresponding proof. These tasks are essential to ensuring a distributed, scalable, and decentralized inference
system.

Validators, on the other hand, play a crucial role in overseeing the miners' work. They validate the zk-proofs submitted
by miners, verifying their accuracy and cryptographic soundness. Additionally, they are tasked with distributing input
data, balancing workloads across miners, and scoring their performance based on specific metrics, such as computation
efficiency and proof size.

For an in-depth look into these roles and their workflows, refer to the [architecture](docs/arc42) file.

#### Incentive and Reward System

kubz introduces an innovative reward mechanism that:
- **Scores Predictions**: Rewards miners based on the cryptographic integrity of the generated zk-proofs and the efficiency of inference.
- **Validates Inference**: Uses validators to confirm that each miner’s zk-proof accurately corresponds to the AI model’s output.
- **Balances Workloads**: Allows participation from both GPU and non-GPU nodes while pushing for GPU-optimized proving systems over time.

#### Roles

- **Miners**
  - Receive input data from validators.
  - Run zkML models to generate a witness and produce zero-knowledge proofs.
  - Return both the witness and its verification proof for validation.

- **Validators**
  - Distribute data and requests to miners.
  - Verify the authenticity and integrity of each miner’s output through zk-proofs.
  - Score miners based on performance metrics such as proof size and response time.

---

## Quickstart

Install kubz and its dependencies with the following command:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/yourusername/kubz/main/setup.sh)"
