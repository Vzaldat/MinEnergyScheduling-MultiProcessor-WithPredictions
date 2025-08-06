from scheduling_functions import densest_interval_BKP

# Test with simple case
jobs = {1: (10, 0, 5), 2: (15, 2, 8)}
print("Testing densest_interval_BKP...")
try:
    result = densest_interval_BKP(jobs, 1.0, 0.1)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}") 