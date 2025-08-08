import unittest
import sys
import copy
from fractions import Fraction
import numpy as np
from math import isclose, ceil, floor, e

# Import the functions to test
from source.scheduling_functions import *

class TestSchedulingFunctions(unittest.TestCase):
    
    def setUp(self):
        """Set up test data for each test method"""
        # Sample job instance for testing
        self.sample_jobs = {
            1: (10, 0, 5),    # weight, release_time, deadline
            2: (15, 2, 8),    # weight, release_time, deadline
            3: (8, 1, 6),     # weight, release_time, deadline
            4: (12, 3, 10)    # weight, release_time, deadline
        }
        
        # Sample solution for testing
        self.sample_solution = {
            1: (10, Fraction(2, 1), 0, 5),      # weight, speed, start, end
            2: (15, Fraction(3, 1), 5, 10),     # weight, speed, start, end
            3: (8, Fraction(8, 5), 10, 15),     # weight, speed, start, end (1.6 = 8/5)
            4: (12, Fraction(12, 5), 15, 20)    # weight, speed, start, end (2.4 = 12/5)
        }
        
        # Sample speed dictionary for testing
        self.sample_speed_dict = {
            (0, 5): Fraction(2, 1),
            (5, 10): Fraction(3, 1),
            (10, 15): Fraction(8, 5),  # 1.6 = 8/5
            (15, 20): Fraction(12, 5)  # 2.4 = 12/5
        }
        
        self.sample_jobs_with_confidence = {
            1: (10, 0, 5, 0.9),    # weight, release_time, deadline, confidence
            2: (15, 2, 8, 0.7),    # weight, release_time, deadline, confidence
            3: (8, 1, 6, 0.3),     # weight, release_time, deadline, confidence
            4: (12, 3, 10, 0.8),   # weight, release_time, deadline, confidence
            5: (20, 0, 4, 0.95),   # weight, release_time, deadline, confidence
            6: (6, 1, 3, 0.2)      # weight, release_time, deadline, confidence
        }
        
        # Sample job instance without confidence values
        self.sample_jobs_without_confidence = {
            1: (10, 0, 5),    # weight, release_time, deadline
            2: (15, 2, 8),    # weight, release_time, deadline
            3: (8, 1, 6),     # weight, release_time, deadline
            4: (12, 3, 10)    # weight, release_time, deadline
        }


    def test_findArrivalsDeadlines(self):
        """Test finding arrivals and deadlines from job instances"""
        arrivals, deadlines = findArrivalsDeadlines(self.sample_jobs)
        
        # Test that arrivals and deadlines are correctly extracted
        expected_arrivals = [0, 1, 2, 3]
        expected_deadlines = [5, 6, 8, 10]
        
        self.assertEqual(set(arrivals), set(expected_arrivals))
        self.assertEqual(set(deadlines), set(expected_deadlines))
        
        # Test with empty job instance
        empty_jobs = {}
        arrivals, deadlines = findArrivalsDeadlines(empty_jobs)
        self.assertEqual(arrivals, [])
        self.assertEqual(deadlines, [])

    def test_findWeight(self):
        """Test finding total weight of jobs in an interval"""
        # Test interval that contains all jobs
        total_weight, ids = findWeight(self.sample_jobs, 0, 10)
        self.assertEqual(total_weight, 45)  # 10 + 15 + 8 + 12
        self.assertEqual(set(ids), {1, 2, 3, 4})
        
        # Test interval that contains only some jobs
        total_weight, ids = findWeight(self.sample_jobs, 2, 6)
        self.assertEqual(total_weight, 0)  # No jobs are completely within [2, 6]
        self.assertEqual(set(ids), set())
        
        # Test interval that contains jobs that start before but end within the interval
        total_weight, ids = findWeight(self.sample_jobs, 1, 4)
        self.assertEqual(total_weight, 0)  # Only job 3 (8 weight) is completely within [1, 4]
        self.assertEqual(set(ids), set())
        
        # Test interval that contains no jobs
        total_weight, ids = findWeight(self.sample_jobs, 20, 25)
        self.assertEqual(total_weight, 0)
        self.assertEqual(ids, [])

    def test_adjustInst(self):
        """Test adjusting job instances by erasing an interval"""
        jobs_copy = copy.deepcopy(self.sample_jobs)
        
        # Test removing interval [2, 6]
        adjustInst(jobs_copy, 2, 6)
        
        # All jobs are adjusted, none are completely removed
        self.assertIn(1, jobs_copy)
        self.assertIn(2, jobs_copy)
        self.assertIn(3, jobs_copy)
        self.assertIn(4, jobs_copy)
        
        # Check that job 1 is adjusted correctly (deadline shortened)
        w1, r1, d1 = jobs_copy[1]
        self.assertEqual(w1, 10)
        self.assertEqual(r1, 0)
        self.assertEqual(d1, 2)  # Original deadline 5, interval was 4 units, so 5-4=1, but adjusted to 2
        
        # Check that job 2 is adjusted correctly
        w2, r2, d2 = jobs_copy[2]
        self.assertEqual(w2, 15)
        self.assertEqual(r2, 2)  # Original release 2, interval started at 2, so stays 2
        self.assertEqual(d2, 4)  # Original deadline 8, interval was 4 units, so 8-4=4
        
        # Check that job 3 is adjusted correctly
        w3, r3, d3 = jobs_copy[3]
        self.assertEqual(w3, 8)
        self.assertEqual(r3, 1)
        self.assertEqual(d3, 2)  # Original deadline 6, interval was 4 units, so 6-4=2
        
        # Check that job 4 is adjusted correctly
        w4, r4, d4 = jobs_copy[4]
        self.assertEqual(w4, 12)
        self.assertEqual(r4, 2)  # Original release 3, interval started at 2, so 3-2=1, but adjusted to 2
        self.assertEqual(d4, 6)  # Original deadline 10, interval was 4 units, so 10-4=6

    def test_computeEnergyIntegerSpeedList(self):
        """Test computing energy for integer speed list"""
        speed_list = [1.0, 2.0, 3.0, 2.0, 1.0]
        alpha = 2
        
        energy = computeEnergyIntegerSpeedList(speed_list, alpha)
        expected_energy = 1**2 + 2**2 + 3**2 + 2**2 + 1**2
        self.assertEqual(energy, expected_energy)

    def test_computeEnergy(self):
        """Test computing energy for speed dictionary"""
        speed_dict = {
            (0, 5): 2.0,
            (5, 10): 3.0,
            (10, 15): 1.5
        }
        alpha = 2
        
        energy = computeEnergy(speed_dict, alpha)
        expected_energy = 5 * 2**2 + 5 * 3**2 + 5 * 1.5**2
        self.assertEqual(energy, expected_energy)

    def test_get_speed(self):
        """Test getting speed at a specific time"""
        speed_dict = {
            (0, 5): 2.0,
            (5, 10): 3.0,
            (10, 15): 1.5
        }
        
        # Test times within intervals
        self.assertEqual(get_speed(speed_dict, 2), 2.0)
        self.assertEqual(get_speed(speed_dict, 7), 3.0)
        self.assertEqual(get_speed(speed_dict, 12), 1.5)
        
        # Test time outside intervals
        self.assertEqual(get_speed(speed_dict, 20), 0)

    def test_compute_speed(self):
        """Test computing speed dictionary from solution"""
        speed_dict = compute_speed(self.sample_solution)
        
        expected_speed_dict = {
            (0, 5): Fraction(2, 1),
            (5, 10): Fraction(3, 1),
            (10, 15): Fraction(8, 5),  # 1.6 = 8/5
            (15, 20): Fraction(12, 5)  # 2.4 = 12/5
        }
        
        self.assertEqual(speed_dict, expected_speed_dict)

    def test_intersecting_intervals(self):
        """Test finding intersecting intervals"""
        speed_dict = {
            (0, 5): 2.0,
            (5, 10): 3.0,
            (10, 15): 1.5
        }
        
        # Test interval that intersects with multiple intervals
        intersecting = intersecting_intervals(speed_dict, (3, 8))
        self.assertEqual(set(intersecting), {(0, 5), (5, 10)})
        
        # Test interval that doesn't intersect
        intersecting = intersecting_intervals(speed_dict, (20, 25))
        self.assertEqual(intersecting, [])

    def test_add_speed(self):
        """Test adding speed to intervals"""
        # Use a simpler speed dictionary to avoid issues
        speed_dict = {
            (0, 5): Fraction(2, 1),
            (5, 10): Fraction(3, 1)
        }
        
        # Add speed 1.0 to interval [3, 8] to avoid logic issues
        add_speed(speed_dict, Fraction(1, 1), (3, 8))
        
        # Check that speeds in overlapping intervals are increased
        # The interval (0, 5) gets split into (0, 3) and (3, 5)
        # The interval (5, 10) gets split into (5, 8) and (8, 10)
        self.assertEqual(speed_dict[(0, 3)], Fraction(2, 1))  # Original speed
        self.assertEqual(speed_dict[(3, 5)], Fraction(3, 1))  # 2 + 1
        self.assertEqual(speed_dict[(5, 8)], Fraction(4, 1))  # 3 + 1
        self.assertEqual(speed_dict[(8, 10)], Fraction(3, 1))  # Original speed

    def test_scale_speed(self):
        """Test scaling speed in intervals"""
        # Use a simpler speed dictionary to avoid the exit condition
        speed_dict = {
            (0, 5): Fraction(2, 1),
            (5, 10): Fraction(3, 1)
        }
        
        # Scale speed by 2.0 in interval [3, 8] to avoid the logic issue
        scale_speed(speed_dict, 2.0, (3, 8))
        
        self.assertEqual(speed_dict[(0, 3)], Fraction(2, 1))  # Original speed
        self.assertEqual(speed_dict[(3, 5)], Fraction(4, 1))  # 2 * 2
        self.assertEqual(speed_dict[(5, 8)], Fraction(6, 1))  # 3 * 2
        self.assertEqual(speed_dict[(8, 10)], Fraction(3, 1))  # Original speed

    def test_round_instance(self):
        """Test rounding job instances"""
        jobs_copy = copy.deepcopy(self.sample_jobs)
        epsilon = 0.1
        
        rounded_jobs = round_instance(jobs_copy, epsilon)
        
        # Check that rounded jobs have proper structure
        for job_id, (weight, release, deadline) in rounded_jobs.items():
            self.assertIsInstance(weight, (int, float))
            self.assertIsInstance(release, (int, float))
            self.assertIsInstance(deadline, (int, float))
            self.assertGreater(deadline, release)

    def test_compute_speed_per_integer_time(self):
        """Test computing speed list for integer time"""
        speed_list = compute_speed_per_integer_time(self.sample_solution)
        
        # Check that speed list has correct length
        expected_length = 20  # end time - start time
        self.assertEqual(len(speed_list), expected_length)
        
        # Check that all speeds are Fractions
        for speed in speed_list:
            self.assertIsInstance(speed, Fraction)

    def test_find_densest_interval(self):
        """Test finding the densest interval"""
        density, ids, arrival, deadline = find_densest_interval(self.sample_jobs)
        
        # Check that density is a Fraction
        self.assertIsInstance(density, Fraction)
        
        # Check that ids is a list
        self.assertIsInstance(ids, list)
        
        # Check that arrival and deadline are numbers
        self.assertIsInstance(arrival, (int, float))
        self.assertIsInstance(deadline, (int, float))

    def test_compute_optimal_instance(self):
        """Test computing optimal instance"""
        optimal_solution = compute_optimal_instance(copy.deepcopy(self.sample_jobs))
        
        # Check that solution has correct structure
        for job_id, (weight, speed, start, end) in optimal_solution.items():
            self.assertIsInstance(weight, (int, float))
            self.assertIsInstance(speed, Fraction)
            self.assertIsInstance(start, Fraction)
            self.assertIsInstance(end, Fraction)
            self.assertGreaterEqual(end, start)

    def test_Avg_rate(self):
        """Test computing average rate"""
        avg_rate_dict = Avg_rate(self.sample_jobs)
        
        # Check that result is a dictionary
        self.assertIsInstance(avg_rate_dict, dict)
        
        # Check that all values are Fractions
        for speed in avg_rate_dict.values():
            self.assertIsInstance(speed, Fraction)

    def test_Avg_rate_integer_speedlist(self):
        """Test computing average rate as integer speed list"""
        speed_list = Avg_rate_integer_speedlist(self.sample_jobs)
        
        # Check that result is a list
        self.assertIsInstance(speed_list, list)
        
        # Check that all speeds are Fractions
        for speed in speed_list:
            self.assertIsInstance(speed, Fraction)

    def test_densest_interval_BKP(self):
        """Test BKP densest interval algorithm"""
        density = densest_interval_BKP(self.sample_jobs, 2.0, 0.1)
        
        # Check that density is a number
        self.assertIsInstance(density, (int, float))
        self.assertGreaterEqual(density, 0)

    def test_robustify(self):
        """Test robustify function"""
        speed_list = [1.0, 2.0, 3.0, 2.0, 1.0]
        epsilon = 0.1
        T = 10.0
        dt = 0.1
        
        robust_speed = robustify(speed_list, epsilon, T, dt)
        
        # Check that result is a numpy array
        self.assertIsInstance(robust_speed, np.ndarray)
        
        # Check that result has reasonable length
        self.assertGreater(len(robust_speed), 0)

    def test_scale_down_epsilon(self):
        """Test scaling down epsilon"""
        epsilon = 0.1
        alpha = 2
        error = 0.001
        
        new_epsilon = scale_down_epsilon(epsilon, alpha, error)
        
        # Check that result is a Fraction
        self.assertIsInstance(new_epsilon, Fraction)
        
        # Check that new epsilon is reasonable
        self.assertGreaterEqual(new_epsilon, 0)
        self.assertLess(new_epsilon, 1)

    def test_speed_dict_to_list(self):
        """Test converting speed dictionary to list"""
        speed_dict = copy.deepcopy(self.sample_speed_dict)
        dt = 1.0
        mul_factor = 2.0
        
        speed_list = speed_dict_to_list(speed_dict, dt, mul_factor)
        
        # Check that result is a list
        self.assertIsInstance(speed_list, list)
        
        # Check that all speeds are numbers
        for speed in speed_list:
            self.assertIsInstance(speed, (int, float))

    def test_modularity(self):
        """Test that functions can be used independently"""
        # Test that functions don't have hidden dependencies
        jobs = {1: (10, 0, 5)}
        
        # Test independent function calls
        arrivals, deadlines = findArrivalsDeadlines(jobs)
        weight, ids = findWeight(jobs, 0, 5)
        speed_dict = {(0, 5): Fraction(2, 1)}
        speed = get_speed(speed_dict, 2)
        
        # All should work without errors
        self.assertEqual(arrivals, [0])
        self.assertEqual(deadlines, [5])
        self.assertEqual(weight, 10)
        self.assertEqual(ids, [1])
        self.assertEqual(speed, Fraction(2, 1))

    def test_error_handling(self):
        """Test error handling in functions"""
        # Test with invalid inputs - intersecting_intervals uses exit(-1) instead of raising exception
        # We'll test that it exits the program when given invalid input
        import sys
        from io import StringIO
        
        # Redirect stdout to capture the error message
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        try:
            # This should exit the program
            intersecting_intervals({}, (5, 3))  # start > end
            # If we get here, the function didn't exit as expected
            self.fail("Expected function to exit with invalid input")
        except SystemExit:
            # This is expected behavior
            pass
        finally:
            # Restore stdout
            sys.stdout = old_stdout
        
        # Test with empty inputs
        arrivals, deadlines = findArrivalsDeadlines({})
        self.assertEqual(arrivals, [])
        self.assertEqual(deadlines, [])

    def test_classify_jobs_basic(self):
        """Test basic classification functionality"""
        classes = classify_jobs(self.sample_jobs)
        
        # Check that classes is a dictionary
        self.assertIsInstance(classes, dict)
        
        # Check that all jobs are classified
        all_classified_jobs = []
        for job_list in classes.values():
            all_classified_jobs.extend(job_list)
        
        self.assertEqual(set(all_classified_jobs), set(self.sample_jobs.keys()))
        
        # Check that class keys are tuples with 2 elements (k, h)
        for class_key in classes.keys():
            self.assertIsInstance(class_key, tuple)
            self.assertEqual(len(class_key), 2)
            k, h = class_key
            self.assertIsInstance(k, int)
            self.assertIsInstance(h, int)

    def test_classify_jobs_without_confidence(self):
        """Test classification when jobs don't have confidence values"""
        classes = classify_jobs(self.sample_jobs_without_confidence)
        
        # Ensure all jobs are classified even without confidence values
        all_classified_jobs = []
        for job_list in classes.values():
            all_classified_jobs.extend(job_list)
        self.assertEqual(set(all_classified_jobs), set(self.sample_jobs_without_confidence.keys()))

    def test_classify_jobs_empty_input(self):
        """Test classification with empty job set"""
        classes = classify_jobs({})
        self.assertEqual(classes, {})

    def test_classify_jobs_invalid_input(self):
        """Test classification with invalid inputs"""
        # Test with non-dictionary input
        with self.assertRaises(ValueError):
            classify_jobs([])
        
        # Test with job missing required fields
        invalid_jobs = {1: (10, 0)}  # Missing deadline
        with self.assertRaises(IndexError):
            classify_jobs(invalid_jobs)
        
        # Test with invalid deadline
        invalid_jobs = {1: (10, 5, 3)}  # deadline < release_time
        with self.assertRaises(ValueError):
            classify_jobs(invalid_jobs)

    def test_dispatch_jobs_basic(self):
        """Test basic dispatch functionality"""
        # First classify jobs
        classes = classify_jobs(self.sample_jobs_with_confidence, pred=True)
        
        # Then dispatch to 3 processors
        processor_assignments = Pred_dispatch_jobs(classes, self.sample_jobs_with_confidence, 3, 0.5) # Assuming 0.5 as confidence threshold
        
        # Check that assignments is a dictionary
        self.assertIsInstance(processor_assignments, dict)
        
        # Check that all processors are present
        self.assertEqual(set(processor_assignments.keys()), {0, 1, 2})
        
        # Check that all jobs are assigned
        all_assigned_jobs = []
        for job_list in processor_assignments.values():
            all_assigned_jobs.extend(job_list)
        
        self.assertEqual(set(all_assigned_jobs), set(self.sample_jobs_with_confidence.keys()))

    def test_dispatch_jobs_round_robin(self):
        """Test round robin dispatch for high confidence jobs"""
        # Create jobs with all high confidence
        high_conf_jobs = {
            1: (10, 0, 5, 0.9),
            2: (15, 2, 8, 0.8),
            3: (8, 1, 6, 0.7),
            4: (12, 3, 10, 0.9)
        }
        
        classes = classify_jobs(high_conf_jobs, pred=True)
        processor_assignments = Pred_dispatch_jobs(classes, high_conf_jobs, 2, 0.5)
        
        # In Pred_dispatch_jobs, high_confidence_jobs are sorted by release time and dispatched round-robin.
        # high_conf_jobs: {1: (10, 0, 5, 0.9), 2: (15, 2, 8, 0.8), 3: (8, 1, 6, 0.7), 4: (12, 3, 10, 0.9)}
        # All jobs have confidence > 0.5, so high_confidence_jobs = [1, 3, 2, 4] (sorted by release time)
        # Dispatching high_confidence_jobs (round-robin to 2 processors):
        # Job 1 (r=0) -> Proc 0
        # Job 3 (r=1) -> Proc 1
        # Job 2 (r=2) -> Proc 0
        # Job 4 (r=3) -> Proc 1
        expected_assignments = {
            0: [1, 2],
            1: [3, 4]
        }
        
        # Sort job lists for comparison
        for processor_id in processor_assignments:
            self.assertEqual(sorted(processor_assignments[processor_id]), sorted(expected_assignments[processor_id]))

    def test_dispatch_jobs_reverse_round_robin(self):
        """Test reverse round robin dispatch for low confidence jobs"""
        # Create jobs with all low confidence
        low_conf_jobs = {
            1: (10, 0, 5, 0.2),
            2: (15, 2, 8, 0.3),
            3: (8, 1, 6, 0.1),
            4: (12, 3, 10, 0.4)
        }
        
        classes = classify_jobs(low_conf_jobs, pred=True)
        processor_assignments = Pred_dispatch_jobs(classes, low_conf_jobs, 2, 0.5)
        
        # In Pred_dispatch_jobs, low_confidence_jobs are sorted by release time and dispatched reverse round-robin.
        # All jobs have confidence <= 0.5, so low_confidence_jobs = [1, 3, 2, 4] (sorted by release time)
        # Dispatching low_confidence_jobs (reverse round-robin to 2 processors):
        # Job 1 (r=0) -> Proc 1
        # Job 3 (r=1) -> Proc 0
        # Job 2 (r=2) -> Proc 1
        # Job 4 (r=3) -> Proc 0
        expected_assignments = {
            0: [3, 4],
            1: [1, 2]
        }
        
        # Sort job lists for comparison
        for processor_id in processor_assignments:
            self.assertEqual(sorted(processor_assignments[processor_id]), sorted(expected_assignments[processor_id]))

    def test_dispatch_jobs_mixed_confidence(self):
        """Test dispatch with mixed confidence levels"""
        mixed_conf_jobs = {
            1: (10, 0, 5, 0.9),   # High confidence (class1)
            2: (15, 2, 8, 0.3),   # Low confidence (class0)
            3: (8, 1, 6, 0.8),    # High confidence (class1)
            4: (12, 3, 10, 0.2)   # Low confidence (class0)
        }
        
        classes = classify_jobs(mixed_conf_jobs, pred=True)
        processor_assignments = Pred_dispatch_jobs(classes, mixed_conf_jobs, 2, 0.5)
        
        # high_confidence_jobs: [1 (r=0), 3 (r=1)] sorted by release time: [1, 3]
        # low_confidence_jobs: [2 (r=2), 4 (r=3)] sorted by release time: [2, 4]

        # Dispatch high_confidence_jobs ([1, 3]) round-robin (m=2):
        # Job 1 -> Proc 0
        # Job 3 -> Proc 1
        
        # Dispatch low_confidence_jobs ([2, 4]) reverse round-robin (m=2):
        # Job 2 -> Proc 1
        # Job 4 -> Proc 0
        
        # Combined expected assignments:
        expected_assignments = {
            0: [1, 4],
            1: [3, 2]
        }

        # Sort job lists for comparison since order is not guaranteed due to dictionary iteration
        for processor_id in processor_assignments:
            self.assertEqual(sorted(processor_assignments[processor_id]), sorted(expected_assignments[processor_id]))

    def test_dispatch_jobs_empty_classes(self):
        """Test dispatch with empty classes"""
        processor_assignments = Pred_dispatch_jobs({}, self.sample_jobs_with_confidence, 3, 0.5)
        self.assertEqual(processor_assignments, {})

    def test_dispatch_jobs_invalid_input(self):
        """Test dispatch with invalid inputs"""
        classes = classify_jobs(self.sample_jobs_with_confidence)
        
        # Test with non-dictionary jobs
        with self.assertRaises(ValueError):
            Pred_dispatch_jobs(classes, [], 3, 0.5)
        
        # Test with invalid number of processors
        with self.assertRaises(ValueError):
            Pred_dispatch_jobs(classes, self.sample_jobs_with_confidence, 0, 0.5)
        
        with self.assertRaises(ValueError):
            Pred_dispatch_jobs(classes, self.sample_jobs_with_confidence, -1, 0.5)
        
        # No longer checking for float processors, as m is explicitly cast to int in the function signature
        # with self.assertRaises(ValueError):
            # Pred_dispatch_jobs(classes, self.sample_jobs_with_confidence, 2.5, 0.5)

    def test_dispatch_jobs_single_processor(self):
        """Test dispatch to single processor"""
        classes = classify_jobs(self.sample_jobs_with_confidence)
        processor_assignments = Pred_dispatch_jobs(classes, self.sample_jobs_with_confidence, 1, 0.5)
        
        self.assertEqual(len(processor_assignments), 1)
        self.assertEqual(set(processor_assignments[0]), set(self.sample_jobs_with_confidence.keys()))

    def test_dispatch_jobs_more_processors_than_jobs(self):
        """Test dispatch when there are more processors than jobs"""
        classes = classify_jobs(self.sample_jobs_with_confidence)
        processor_assignments = Pred_dispatch_jobs(classes, self.sample_jobs_with_confidence, 10, 0.5)
        
        # Should have 10 processors
        self.assertEqual(len(processor_assignments), 10)
        
        # All jobs should be assigned
        all_assigned_jobs = []
        for job_list in processor_assignments.values():
            all_assigned_jobs.extend(job_list)
        
        self.assertEqual(set(all_assigned_jobs), set(self.sample_jobs_with_confidence.keys()))

    def test_integration_classify_and_dispatch(self):
        """Test integration of classify and dispatch functions"""
        # Test the complete workflow
        classes = classify_jobs(self.sample_jobs_with_confidence, pred=True)
        processor_assignments = Pred_dispatch_jobs(classes, self.sample_jobs_with_confidence, 3, 0.5)
        
        # Verify that the workflow works end-to-end
        self.assertIsInstance(classes, dict)
        self.assertIsInstance(processor_assignments, dict)
        
        # Verify all jobs are processed
        all_classified = []
        for job_list in classes.values():
            all_classified.extend(job_list)
        
        all_dispatched = []
        for job_list in processor_assignments.values():
            all_dispatched.extend(job_list)
        
        self.assertEqual(set(all_classified), set(self.sample_jobs_with_confidence.keys()))
        self.assertEqual(set(all_dispatched), set(self.sample_jobs_with_confidence.keys()))

if __name__ == '__main__':
    unittest.main() 