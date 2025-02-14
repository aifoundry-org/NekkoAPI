import json
import logging
import os
import shutil
import subprocess
import time
import yaml
import docker
from typing import TypedDict, Dict, Any
import argparse
import glob


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


def setup_api_results_directory(results_dir: str, api_name: str) -> None:
    """Ensure results directory exists and clean specific API results."""
    api_results_pattern = os.path.join(results_dir, f"{api_name}_*")
    
    # Create results directory if it doesn't exist
    os.makedirs(results_dir, exist_ok=True)
    
    # Clean existing results for this API
    for item in glob.glob(api_results_pattern):
        if os.path.isfile(item):
            os.unlink(item)
        elif os.path.isdir(item):
            shutil.rmtree(item)
    
    logging.info(f"Cleaned results directory for API: {api_name}")


def get_container_stats(client: docker.DockerClient, container_name: str) -> Dict[Any, Any]:
    """Get stats for a specific container"""
    try:
        container = client.containers.get(container_name)
        stats = container.stats(stream=False)
        
        # Calculate CPU percentage
        cpu_percent = 0.0
        try:
            cpu_stats = stats.get('cpu_stats', {})
            precpu_stats = stats.get('precpu_stats', {})
            
            cpu_usage = cpu_stats.get('cpu_usage', {})
            precpu_usage = precpu_stats.get('cpu_usage', {})
            
            cpu_delta = cpu_usage.get('total_usage', 0) - precpu_usage.get('total_usage', 0)
            system_delta = cpu_stats.get('system_cpu_usage', 0) - precpu_stats.get('system_cpu_usage', 0)
            
            if system_delta > 0:
                num_cpus = len(cpu_usage.get('percpu_usage', [1]))  # Default to 1 if not available
                cpu_percent = (cpu_delta / system_delta) * num_cpus * 100.0
        except Exception as e:
            logging.warning(f"Error calculating CPU stats: {e}")

        # Calculate memory usage in MB
        try:
            memory_stats = stats.get('memory_stats', {})
            memory_usage_mb = memory_stats.get('usage', 0) / (1024 * 1024)
        except Exception as e:
            logging.warning(f"Error calculating memory stats: {e}")
            memory_usage_mb = 0

        return {
            'cpu_percent': cpu_percent,
            'memory_mb': memory_usage_mb
        }
    except Exception as e:
        logging.error(f"Error getting stats for container {container_name}: {e}")
        return None


def run_scenario(
    scenario: ScenarioConfig,
    api_config: APIEndpointConfig,
    global_config: GlobalConfig,
    results_dir: str,
) -> bool:
    """Run a single test scenario with Docker container resource monitoring."""
    scenario_name = scenario["name"]
    prompt = scenario["prompt"]
    output_length = scenario["output_length"]
    num_concurrent_requests = scenario["num_concurrent_requests"]
    max_num_completed_requests = scenario["max_num_completed_requests"]

    logging.info(f"Running scenario: {scenario_name}")

    # Initialize Docker client
    docker_client = docker.from_env()

    # Merge JSON parameters
    merged_params = json.loads(api_config["additional_sampling_params"])
    merged_params.update({"prompt": prompt, "max_tokens": output_length})
    merged_params_json = json.dumps(merged_params)

    results_dir_run = os.path.join(results_dir, f"{api_config['name']}_{scenario_name}")
    os.makedirs(results_dir_run, exist_ok=True)

    max_attempts = 3
    target_container = f"perf_test-{api_config['container_name']}-1"

    for attempt in range(1, max_attempts + 1):
        # Initialize max metrics
        max_cpu_percent = 0.0
        max_memory_mb = 0.0

        cmd = [
            "python",
            "llmperf/token_benchmark_ray.py",
            "--model",
            scenario_name,
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
                stats = get_container_stats(docker_client, target_container)
                if stats:
                    max_cpu_percent = max(max_cpu_percent, stats['cpu_percent'])
                    max_memory_mb = max(max_memory_mb, stats['memory_mb'])
                
                time.sleep(0.5)  # Add small delay to prevent excessive sampling

            except Exception as e:
                logging.error(f"Error collecting container metrics: {e}")
                break

        if process.returncode == 0:
            metrics = {
                "max_cpu_percent": max_cpu_percent,
                "max_memory_mb": max_memory_mb
            }

            metrics_file = os.path.join(results_dir_run, "container_metrics.json")
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
    logging.info(f"Testing for API '{api_name}' completed.")
    return True


def run_perf_test(api_names: list[str] = []) -> None:
    """Main function to orchestrate the performance tests."""
    setup_logging()

    params = load_specs("config/specs.yaml")

    global_config = params["global"]

    os.makedirs(global_config["results_dir"], exist_ok=True)

    for api_name, api_config in params["api_endpoints"].items():
        if api_names and api_name not in api_names:
            logging.info(f"Skipping API '{api_name}' as it was not specified in command line arguments")
            continue

        # Clean results directory for this API before running tests
        setup_api_results_directory(global_config["results_dir"], api_name)

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
    parser = argparse.ArgumentParser(description='Run performance tests for specified APIs')
    parser.add_argument('--api-names', nargs='*', help='List of API names to test. If not specified, all APIs will be tested.')
    args = parser.parse_args()
    run_perf_test(args.api_names)
