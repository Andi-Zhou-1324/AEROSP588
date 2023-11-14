import numpy as np
import time
from truss import tenbartruss

# Define the area and the perturbation step size
A = np.ones(10, dtype="complex_")
h_values = [1E-7, 1E-200]

# Number of times each method should be run
num_runs = 1

# Initialize dictionary to store results
results = {}

# Define the gradient methods to be tested
grad_methods = ['FD', 'CS', 'DT_CS', 'AJ_CS']

# Loop through the gradient methods
for grad_method in grad_methods:
    # Initialize list to store run times for each method
    run_times = []

    # Set h according to the grad_method
    h = h_values[0] if 'FD' in grad_method else h_values[1]

    # Run the method multiple times
    for _ in range(num_runs):
        start_time = time.time()
        result = tenbartruss(A, h, grad_method=grad_method, aggregate=False)
        elapsed_time = time.time() - start_time
        run_times.append(elapsed_time)

    # Calculate average run time
    average_run_time = np.mean(run_times)

    # Store results and average elapsed time
    results[grad_method] = {
        'result': result,
        'average_run_time': average_run_time
    }

    # Print the average elapsed time nicely
    print(f"Average elapsed time for {grad_method}: {average_run_time:.6f} seconds")

# Calculate and print the relative errors
AJ_CS_solution = results['AJ_CS']['result'][2]
for grad_method, data in results.items():
    if grad_method != 'AJ_CS':  # Avoid comparing the method to itself
        rel_error = np.linalg.norm(data['result'][2] - AJ_CS_solution,'fro')/np.linalg.norm(AJ_CS_solution,'fro')
        print(f"Relative error for {grad_method} compared to AJ_CS: {rel_error:.10e}")
