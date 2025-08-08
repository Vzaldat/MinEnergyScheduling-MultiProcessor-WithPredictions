from fractions import Fraction
from typing import Dict, Any, Tuple, Union
from scheduling_functions import *

import numpy as np
import sys
import copy
from random import sample, randint, seed
from math import isclose, ceil, floor
from decimal import *
from fractions import *

def BKP_alg(J, dt, alpha):
    key = sorted(J.keys())
    min_time = J[key[0]][1]
    max_time = J[key[-1]][2]

    energy = 0
    t = min_time
    while t < max_time:
        density = densest_interval_BKP(J, t, dt)
        energy += dt * ((e * density) ** alpha)
        t += dt
    
    return energy

def OptimalOnline(_J):
    all_idxs = sorted(_J.keys())
    if not all_idxs:
        return []

    # Determine the overall time range
    min_release_time = min(job[1] for job in _J.values())
    max_deadline = max(job[2] for job in _J.values())

    current_time = min_release_time
    speeds_OO = []
    current_active_jobs = {}

    # Create a copy of jobs for tracking remaining work
    remaining_jobs_orig = copy.deepcopy(_J)

    while current_time < max_deadline or current_active_jobs:
        # 1. Add jobs whose release time is current_time
        jobs_to_add = []
        for job_id, job_data in list(remaining_jobs_orig.items()):
            w, r, d = job_data
            if r <= current_time and job_id not in current_active_jobs:
                current_active_jobs[job_id] = [w, r, d] # Store as mutable list for remaining weight
                del remaining_jobs_orig[job_id]
        
        # 2. If no active jobs, append 0 speed and move to next release time
        if not current_active_jobs:
            next_release_time = float('inf')
            for job_id, job_data in remaining_jobs_orig.items():
                next_release_time = min(next_release_time, job_data[1])
            
            if next_release_time == float('inf'): # No more jobs to release
                break
            
            while current_time < next_release_time and current_time < max_deadline:
                speeds_OO.append(Fraction(0, 1))
                current_time += 1
            continue

        # 3. Compute optimal instance for current active jobs
        # Create a temporary job dictionary for compute_optimal_instance
        temp_J = {}
        for job_id, job_data in current_active_jobs.items():
            temp_J[job_id] = tuple(job_data) # Convert to tuple for consistent input

        J_sol_active = compute_optimal_instance(temp_J)
        speed_list_active = compute_speed_per_integer_time(J_sol_active)

        # Get the speed for the current time slot (duration 1 unit)
        speed_for_slot = Fraction(0, 1)
        if speed_list_active: # speed_list_active could be empty if compute_optimal_instance returns empty sol
            # We need to consider how compute_speed_per_integer_time works. 
            # It returns a list where index i represents speed for time [i, i+1)
            # We are currently at time `current_time`, so we need the speed at the beginning of the optimal solution's timeline
            speed_for_slot = speed_list_active[0]

        speeds_OO.append(speed_for_slot)
        work_done = speed_for_slot * 1  # Work done in this unit time slot

        # 4. Deduct work from active jobs (greedy by earliest deadline first)
        # Convert to list of tuples for sorting, then back to dict for modification
        sorted_active_jobs = sorted(current_active_jobs.items(), key=lambda item: item[1][2]) # Sort by deadline
        
        jobs_to_remove = []
        for job_id, job_data in sorted_active_jobs:
            remaining_weight = job_data[0]
            if work_done >= remaining_weight:
                work_done -= remaining_weight
                job_data[0] = Fraction(0, 1) # Mark as completed
                jobs_to_remove.append(job_id)
            else:
                job_data[0] -= work_done
                work_done = Fraction(0, 1) # All work done for this slot consumed
                break
        
        for job_id in jobs_to_remove:
            del current_active_jobs[job_id]

        current_time += 1 # Move to the next time slot
    
    return speeds_OO


def Avg_rate(J):
    # input: receives an instance J which is represented as a dictionary:
    #                                                           key--> job id
    #                                                           value--> (job weight, release time, deadline) as a tuple
    # output: the speed_list dictionary of this algorithm
    ids = sorted(J.keys())
    _, _start, _ = J[ids[0]]
    _, _, _end = J[ids[-1]]
    start = _start
    end = _end
    speed_list_dictionary = {}
    speed_list_dictionary[(start, end)] = Fraction(0, 1)
    for id in ids:
        w_id, r, d = J[id]
        speed = Fraction(w_id, d - r)
        r_id = Fraction(r, 1)
        d_id = Fraction(d, 1)
        interval = (r_id, d_id)
        add_speed(speed_list_dictionary, speed, interval)
    return speed_list_dictionary


def Optimal_Alg(J):
    # input: receives an instance J which is represented as a dictionary:
    #                                                           key--> job id
    #                                                           value--> (job weight, release time, deadline) as a tuple
    # output: the speed_list dictionary of this algorithm, thus the optimal speed function
    #                           key --> interval as a tuple (r,d)
    #                           value--> speed in the aforementioned interval
    #         J_sol the solution instance as a dictionary with:
    #                                           key--> the id of the job
    #                                           value--> (weight, speed, start processing, end processing)

    J_sol = compute_optimal_instance(copy.deepcopy(J))
    speed_list_dictionary = compute_speed(J_sol)
    return speed_list_dictionary, J_sol


def LAS(_J_prediction, _J_true, _epsilon, dt, alpha):
    # input: (1)receives the prediction and real instances as dictionaries:
    #                                                       key --> job id
    #                                                       value--> (job weight, release time, deadline) as a tuple
    #        (2) the robustness parameter _epsilon
    #        (3) dt --> the time granularity of the output list
    #        (4) alpha --> the convexity parameter of the problem
    # output: a speed list which is the actual speed that our processor runs
    #         element i of the output speed represents time = i*dt
    
    
    
    
    # find epsilon such that ((1+epsilon)/(1 - epsilon))^alpha = 1 + _epsilon
    epsilon = scale_down_epsilon(_epsilon, alpha , 0.01)

    # Calculate overall time span T from prediction jobs
    min_r_pred = min(job[1] for job in _J_prediction.values())
    max_d_pred = max(job[2] for job in _J_prediction.values())
    T = max_d_pred - min_r_pred

    T_prime = Fraction(1-epsilon, 1)*T
    J_true = copy.deepcopy(_J_true)
    
    #preprocess the prediction so as to have T_prime = (1-epsilon)*T
    J_prediction = {}
    ids = sorted(_J_prediction.keys())
    for id in ids:
        w_id, start, end = _J_prediction[id]
        end_prime = start + T_prime
        J_prediction[id] = (w_id, start, end_prime)    
      
    #Optimal_Alg returns a speed list and a solution instance 
    _, J_prediction_opt = Optimal_Alg(J_prediction)

    speed_not_robust = {}
    speed_to_smear_out = {}
    ids = sorted(J_prediction_opt.keys())
    for id in ids:
        w_pred, speed_pred, start_processing, end_processing = J_prediction_opt[id]
        w_true, start, end = J_true[id]
        _, start_prime, end_prime = J_prediction[id]
        # just to check if everything is ok
        if start_prime != start:
            print("Houston we have a problem")
             
        processing_interval = (start_processing, end_processing)
        feasible_interval = (start_prime, end_prime)
        if w_true <= w_pred:
            speed_to_finish_between_start_and_end = Fraction(w_true, end_processing-start_processing)
            speed_not_robust[processing_interval] = speed_to_finish_between_start_and_end
        else:
            speed_not_robust[processing_interval] = speed_pred
            w_exceeding = w_true - w_pred
            speed_to_smear_out[feasible_interval] = Fraction(w_exceeding, end_prime-start_prime)
        
    
    # at this point we will use speed_to_smear out in order to 
    # augment the speed in the speed_not_robust dictionary and
    # produce a feasible schedule
    
    intervals = speed_to_smear_out.keys()
    for interval in intervals:
        speed_to_add = speed_to_smear_out[interval]
        add_speed(speed_not_robust, speed_to_add, interval)
    
    
    #now it remains to robustify the speed through convolution
    
    # we turn our dictionary to a list with granularity dt
    # mul_factor is a parameter iwhich is set more than 1 in the noise robust version of LAS
    speed_to_make_robust = speed_dict_to_list(speed_not_robust, dt, mul_factor = 1)
    
    # we use the robustify function to perform the convolution
    speed_robust = robustify(speed_to_make_robust, epsilon, T, dt)
    
    return speed_robust

def DCRR(J, m):
    classes = classify_jobs(J)
    processor_assignments = dispatch_jobs(classes, J, m)
    speed = {}
    for i, job_list in processor_assignments.items():
        # For each job list, do AVR
        # The problem would be 
        sub_J = {}
        for job_id in job_list:
            sub_J[job_id] = J[job_id]
        speed_i = Avg_rate(sub_J)
        speed.update(speed_i)
    return speed

def swp_m(J_True, J_Predicted, m, _epsilon, dt, alpha, confThreshold):
    classes = classify_jobs(J_Predicted, pred=True)
    processor_assignments = Pred_dispatch_jobs(classes, J_Predicted, m, confThreshold)

    # Determine the maximum time horizon across all jobs from J_True and J_Predicted
    max_overall_deadline = 0
    if J_True:
        max_overall_deadline = max(max_overall_deadline, max(job[2] for job in J_True.values()))
    if J_Predicted:
        max_overall_deadline = max(max_overall_deadline, max(job[2] for job in J_Predicted.values()))
    
    # Initialize combined speed list to zeros
    combined_speed_list_length = int(ceil(float(max_overall_deadline) / dt))
    combined_speed = np.zeros(combined_speed_list_length)

    for i, job_list in processor_assignments.items():
        if not job_list:
            continue

        sub_J_prediction = {}
        # Assuming J contains the prediction with confidence, and we want to use it as true for LAS for now
        sub_J_true = {}
        for job_id in job_list:
            # LAS expects (weight, release_time, deadline), so exclude confidence for prediction part
            sub_J_prediction[job_id] = (J_Predicted[job_id][0], J_Predicted[job_id][1], J_Predicted[job_id][2])
            # For simplicity, using predicted jobs as true jobs for LAS here. 
            # A proper implementation might require a separate J_true input to swp_m.
            sub_J_true[job_id] = (J_True[job_id][0], J_True[job_id][1], J_True[job_id][2])

        # Call LAS for the subset of jobs assigned to this processor
        # Note: LAS internally calculates T based on its _J_prediction input
        processor_speed_list = LAS(sub_J_prediction, sub_J_true, _epsilon, dt, alpha)

        # Accumulate speeds into the combined speed list
        # Ensure the combined_speed array is large enough
        if len(processor_speed_list) > len(combined_speed):
            # Resize combined_speed if a processor's schedule extends further
            temp_combined_speed = np.zeros(len(processor_speed_list))
            temp_combined_speed[:len(combined_speed)] = combined_speed
            combined_speed = temp_combined_speed
        
        combined_speed[:len(processor_speed_list)] += processor_speed_list
        
    return combined_speed

"""
Comparisons can be done
DCRR + LAS
Your Disptach + OPT OF SINGLE PROCESSOR against all processors
Your Dispatch + LAS
DCRR + YDS
"""