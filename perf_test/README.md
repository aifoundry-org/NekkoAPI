# LLM Performance Testing

This tool benchmark and measure the LLMs across different llms

## Overview

The performance testing tool:
- Run concurrent request load tests against LLM endpoints
- Measure system resource utilization (CPU, Memory)
- Compare performance across different API implementations
- Configure multiple test scenarios with different parameters

## Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Make
- Git
- Curl (for downloading models)

## Configuration

Test parameters are defined in `config/specs.yaml` with the following structure:

## Usage

1. Clone the repository and navigate to the perf_test directory
2. Run the performance tests:
`make`

To clean up all generated files and directories:
`make clean`

## Test Results

Results are stored in the `results` directory, organized by API endpoint and scenario name. Each test run generates:
- Performance metrics
- System resource utilization data (CPU, Memory)
- Request/response statistics

## Monitoring

The framework monitors and records:
- System-wide CPU usage
- System-wide memory utilization
- Request latencies
- Throughput metrics

## Error Handling

- Tests automatically retry failed scenarios up to 3 times
- Graceful cleanup of resources on test completion or failure
