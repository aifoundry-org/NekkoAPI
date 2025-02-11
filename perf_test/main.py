import json
import logging
import os
import shutil
import subprocess
import time
import yaml
import psutil

from typing import TypedDict


class ScenarioConfig(TypedDict):
    name: str
    prompt: str
    output_length: int
    mean_input_tokens: int
    stddev_input_tokens: int
    mean_output_tokens: int
    stddev_output_tokens: int
    num_concurrent_requests: int
    max_num_completed_requests: int


class APIEndpointConfig(TypedDict):
    llm_api: str
    api_key: str
    api_base: str
    additional_sampling_params: str


class GlobalConfig(TypedDict):
    model: str
    timeout: int
    iterations: int
    results_dir: str


class TestParams(TypedDict):
    global_config: GlobalConfig
    api_endpoints: dict[str, APIEndpointConfig]
    scenarios: list[ScenarioConfig]


def setup_logging() -> None:
    """Configure logging based on environment variable."""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logging.debug(f"Log level set to: {log_level}")


def load_specs(params_file: str) -> TestParams:
    """Load and return test parameters from YAML file."""
    with open(params_file, "r") as file:
        return yaml.safe_load(file)


def setup_results_directory(results_dir: str) -> None:
    """Ensure results directory exists and is empty."""
    if os.path.exists(results_dir):
        for item in os.listdir(results_dir):
            item_path = os.path.join(results_dir, item)
            if os.path.isfile(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    else:
        logging.info(f"Creating results directory: {results_dir}")
        os.makedirs(results_dir)


def run_scenario(
    scenario: ScenarioConfig,
    api_config: APIEndpointConfig,
    global_config: GlobalConfig,
    results_dir: str,
) -> bool:
    """Run a single test scenario with system resource monitoring."""
    scenario_name = scenario["name"]
    prompt = scenario["prompt"]
    output_length = scenario["output_length"]
    num_concurrent_requests = scenario["num_concurrent_requests"]
    max_num_completed_requests = scenario["max_num_completed_requests"]

    logging.info(f"Running scenario: {scenario_name}")

    # Merge JSON parameters
    merged_params = json.loads(api_config["additional_sampling_params"])
    merged_params.update({"prompt": prompt, "max_tokens": output_length})
    merged_params_json = json.dumps(merged_params)

    results_dir_run = os.path.join(results_dir, f"{api_config['name']}_{scenario_name}")
    os.makedirs(results_dir_run, exist_ok=True)

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        # Get baseline measurements before starting the test
        baseline_cpu = psutil.cpu_percent(interval=0.5)
        baseline_memory = psutil.virtual_memory().used / (1024 * 1024)

        # Initialize metrics collection
        cpu_measurements: list[float] = []
        memory_measurements: list[float] = []

        # For delta calculations
        cpu_delta_measurements: list[float] = []
        memory_delta_measurements: list[float] = []

        prev_time = time.time()

        cmd = [
            "python",
            "llmperf/token_benchmark_ray.py",
            "--model",
            "dummy_model",
            "--num-concurrent-requests",
            str(num_concurrent_requests),
            "--results-dir",
            results_dir_run,
            "--llm-api",
            api_config["llm_api"],
            "--additional-sampling-params",
            merged_params_json,
            "--timeout",
            str(global_config["timeout"]),
            "--max-num-completed-requests",
            str(max_num_completed_requests),
        ]

        process = subprocess.Popen(cmd)

        # Collect metrics while the process is running
        while process.poll() is None:
            try:
                timeout = 0.5
                current_time = time.time()
                time_delta = current_time - prev_time

                # System-wide CPU usage
                cpu_percent = psutil.cpu_percent(interval=timeout)
                cpu_delta = (cpu_percent - baseline_cpu) / time_delta

                # System-wide memory usage
                memory = psutil.virtual_memory()
                memory_used_mb = (memory.total - memory.available) / (1024 * 1024)
                memory_delta = (memory_used_mb - baseline_memory) / time_delta

                # Store measurements
                cpu_measurements.append(cpu_percent)
                memory_measurements.append(memory_used_mb)
                cpu_delta_measurements.append(cpu_delta)
                memory_delta_measurements.append(memory_delta)

                # Update previous values
                baseline_cpu = cpu_percent
                baseline_memory = memory_used_mb
                prev_time = current_time

                time.sleep(timeout)  # Add small delay to prevent excessive sampling

            except Exception as e:
                logging.error(f"Error collecting system metrics: {e}")
                break

        if process.returncode == 0:
            if cpu_measurements:  # If we collected any measurements
                metrics = {
                    "system_cpu_usage": {
                        "baseline": baseline_cpu,
                        "max_during_test": max(cpu_measurements),
                        "max_increase": max(cpu_measurements) - baseline_cpu,
                    },
                    "system_memory_usage_mb": {
                        "baseline": baseline_memory,
                        "max_during_test": max(memory_measurements),
                        "max_increase": max(memory_measurements) - baseline_memory,
                    },
                }

                metrics_file = os.path.join(results_dir_run, "system_metrics.json")
                with open(metrics_file, "w") as f:
                    json.dump(metrics, f, indent=2)

            logging.info(f"Scenario '{scenario_name}' completed successfully.")
            return True
        else:
            logging.warning(f"Attempt {attempt} for scenario '{scenario_name}' failed.")

            if attempt == max_attempts:
                logging.error(
                    f"Scenario '{scenario_name}' failed after {attempt} attempts. Exiting."
                )
                return False
            else:
                logging.info("Retrying in 1 second...")
                time.sleep(1)


def run_api_tests(
    api_name: str,
    api_config: APIEndpointConfig,
    scenarios: list[ScenarioConfig],
    global_config: GlobalConfig,
    results_dir: str,
) -> bool:
    """Run all scenarios for a specific API endpoint."""
    logging.info(f"Starting tests for API: {api_name} ({api_config['llm_api']})")

    os.environ["OPENAI_API_KEY"] = api_config["api_key"]
    os.environ["OPENAI_API_BASE"] = api_config["api_base"]

    for scenario in scenarios:
        if not run_scenario(
            scenario, {**api_config, "name": api_name}, global_config, results_dir
        ):
            return False
        time.sleep(5)
    logging.info(f"Testing for API '{api_name}' completed.")
    return True


def main() -> None:
    """Main function to orchestrate the performance tests."""
    # Setup logging
    setup_logging()

    # Load test parameters
    params = load_specs("config/specs.yaml")

    # Extract global parameters
    global_config = params["global"]

    # Setup results directory
    setup_results_directory(global_config["results_dir"])

    # Process API endpoints
    for api_name, api_config in params["api_endpoints"].items():
        if not run_api_tests(
            api_name,
            api_config,
            params["scenarios"],
            global_config,
            global_config["results_dir"],
        ):
            logging.error("Test execution failed. Exiting.")
            exit(1)

    logging.info("All tests completed successfully.")


if __name__ == "__main__":
    main()
