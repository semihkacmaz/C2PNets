import os
import re

# Directory containing the .out files
log_dir = "./logs"

# List to store extracted numbers
times = []

# Regex to extract the elapsed time in seconds
time_pattern = re.compile(r"Elapsed time: (\d+\.\d+) s")

# Loop through .out files in the directory
for filename in os.listdir(log_dir):
    if filename.endswith(".out"):
        with open(os.path.join(log_dir, filename), 'r') as file:
            for line in file:
                match = time_pattern.search(line)
                if match:
                    times.append(float(match.group(1)))

# Sort and get the fastest 10
fastest_times = sorted(times)[:10]

# Output the result
print(f"Fastest 10 times: {fastest_times}")
