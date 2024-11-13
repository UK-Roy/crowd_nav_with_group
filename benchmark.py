import re
import subprocess
import itertools
from benchmark_config import config_updates

def update_config_parameter(config_file, param_name, new_value):
    """
    Update a specific parameter in the config file.
    """
    with open(config_file, 'r') as file:
        lines = file.readlines()
        
    # Ensure the new value is a string, wrapping in quotes if necessary
    if isinstance(new_value, str):
        new_value = f"'{new_value}'"  # Wrap string values in quotes
    
    # Loop through each line to find and update the specified parameter
    for i, line in enumerate(lines):
        if re.match(rf'\s*{param_name}\s*=', line):  # Match the line with the parameter
            lines[i] = f"    {param_name} = {new_value}\n"
            break
    
    # Write the updated lines back to the config file
    with open(config_file, 'w') as file:
        file.writelines(lines)

def benchmark(config_file, test_script):
    """
    Iterate over all config parameters in the benchmark list and update the config file.
    """
    keys = list(config_updates.keys())
    value_combinations = list(itertools.product(*config_updates.values()))

    # Loop through all combinations and update the config file accordingly
    for combination in value_combinations:
        # Update the config file for each combination
        for idx, param_name in enumerate(keys):
            value = combination[idx]
            print(f"Updating {param_name} to {value}...")
            update_config_parameter(config_file, param_name, value)
            print(f"{param_name} set to {value}")
            
        # Use the first value in the list as the log file name prefix
        log_filename = f"logs/{param_name}_{value}.log"
        
        # Run the test and save the results in the log file
        # run_test_and_save_log(test_script, log_filename)
        # print(f"Results saved in {log_filename}")

def run_test_and_save_log(test_script, log_filename):
    """
    Run the test script and save the output to a log file.
    """
    with open(log_filename, 'w') as log_file:
        # Run the test.py script and capture the output
        process = subprocess.Popen(
            ['python', test_script],
            stdout=log_file,
            stderr=log_file
        )
        process.communicate()  # Wait for the process to finish

# Example usage
if __name__ == "__main__":
    config_file = 'config.py'  # Path to your config.py file
    test_script = 'test.py'  # Path to your test script
    benchmark(config_file, test_script)
