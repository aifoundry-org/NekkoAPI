# LLM Performance Testing Suite

A comprehensive benchmarking framework to simulate real-world concurrent loads on Large Language Model (LLM) API endpoints. This suite not only measures request latencies and throughput but also tracks system resource usage (CPU and memory) to help you compare performance across multiple API implementations.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Test Results & Monitoring](#test-results--monitoring)
- [Error Handling & Cleanup](#error-handling--cleanup)
- [Project Structure](#project-structure)
- [Contributing](#contributing)

---

## Overview

The performance testing suite is designed to:

- **Simulate Concurrent Load:** Run concurrent request tests against LLM API endpoints.
- **Measure Performance Metrics:** Track key metrics like throughput, request latencies (time to first token and complete response), and system resource utilization.
- **Compare Different Implementations:** Benchmark various API implementations (e.g., Nekko, vLLM, Ollama) side by side.
- **Flexible Test Scenarios:** Configure multiple scenarios with varying prompts, output lengths, and concurrency levels.

---

## Features

- **Docker & Docker Compose:** Orchestrate multi-container environments for isolated testing.
- **Makefile Automation:** Easily run all or individual test suites.
- **Health Checks & Dependency Management:** Ensure services are ready before tests begin.
- **Aggregated Results:** Automatically combine metrics and generate CSV reports.
- **Retry Mechanism:** Automatically retries failed test scenarios (up to 3 attempts).

---

## Prerequisites

Before getting started, make sure you have the following installed:

- **Docker** and **Docker Compose**
- **Make**
- **Git**
- **Curl** (required for downloading models)

---

## Installation & Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your_username/llm-performance-testing.git
   cd llm-performance-testing
   ```

2. **Prepare the Environment:**

   - The **Makefile** automates cloning necessary subprojects (e.g., `llmperf` and `vllm`).
   - Models required for certain APIs (e.g., Ollama) will be downloaded automatically via Make targets.
   - Docker Compose profiles are set up for different test modes (e.g., `nekko_mode`, `vllm_mode`, `ollama_mode`).

3. **Build Docker Images & Setup Dependencies:**

   The Makefile targets take care of building the images with the correct Dockerfiles and setting up the required services.

---

## Configuration

Test parameters and API endpoints are defined in the `config/specs.yaml` file. This file includes:

- **Global Settings:** Timeout, iterations, and results directory.
- **API Endpoints:** Settings for each API (e.g., `nekko`, `vllm`, `ollama`), including the model, API key, base URL, and extra sampling parameters.
- **Scenarios:** A list of test scenarios (e.g., short prompt, medium prompt, long prompt, high concurrency) with their specific parameters.

Other configuration files include:

- `config/nekko_settings.json` – Settings for the Nekko API.
- `config/otel-collector-config.yaml` – Configuration for the OpenTelemetry collector used in monitoring.

---

## Usage

### Running Tests

You can run all tests or target specific API modes using the provided Makefile commands.

- **Run All Tests:**

  ```bash
  make perf_test_all
  ```

- **Run Specific API Tests:**

  - **vLLM:**

    ```bash
    make perf_test_vllm
    ```

  - **Nekko:**

    ```bash
    make perf_test_nekko
    ```

  - **Ollama:**

    ```bash
    make perf_test_ollama
    ```

Each target performs the following steps:
1. Clones necessary subprojects (if not already present).
2. Downloads required models.
3. Builds the appropriate Docker image (with your host’s Docker group ID for permissions).
4. Runs the Docker Compose stack for the chosen profile.
5. Waits for dependent services (using health checks) and then executes the performance benchmark.
6. Cleans up the containers once tests are complete.

### Cleaning Up

To remove generated directories and built files, run:

```bash
make cleanup
```

---

## Test Results & Monitoring

- **Results Location:** All test outputs and metrics are stored under the `results` directory.
- **Metrics Captured:**
  - **Performance Metrics:** Throughput (tokens/sec) and latency (time to first token and complete response).
  - **System Metrics:** CPU usage and memory consumption.
- **Aggregated Report:** The `scripts/combine_results.py` script aggregates individual test results and generates a CSV report (`results/aggregated_benchmarks.csv`).

The suite also uses an OpenTelemetry collector (configured in `config/otel-collector-config.yaml`) to gather and output detailed logs for troubleshooting and analysis.

---

## Error Handling & Cleanup

- **Automatic Retries:** Each test scenario automatically retries up to 3 times if a run fails.
- **Graceful Shutdown:** The Docker Compose configuration along with health checks ensures that containers are properly stopped after tests.
- **Logs & Metrics:** Detailed logs (including system resource usage) are captured to aid in diagnosing any issues.
