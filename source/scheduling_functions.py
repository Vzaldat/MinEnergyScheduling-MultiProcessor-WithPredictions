import numpy as np
import sys
import copy
from random import sample, randint, seed
from math import isclose, ceil, floor, e, log2
from decimal import *
from fractions import *

def findArrivalsDeadlines(J):
    """
    Input - dictionary which represents the job instances, (job_id : (release time, deadline))
    Output - list of arrivals and deadlines
    """

    arrivals = []
    deadlines = []
    for k in J.keys():
        _, r, d = J[k]
        arrivals.append(r)
        deadlines.append(d)

    arrivals = list(set(arrivals))
    deadlines = list(set(deadlines))
    return arrivals, deadlines

def findWeight(J, r, d):
    """
    Input - Job instance is given and an interval [r, d]
    Output - total weight of jobs in the interval
    """
    totWeight = 0
    ids = []
    for k in J.keys():
        w, rk, dk = J[k]
        if rk >= r and dk <= d:
            totWeight += w
            ids.append(k)
    return totWeight, ids

def adjustInst(J, r, d):
    """
    Input - Job instance is given and an interval [r, d]
    Output - Change job instance J erasing the interval [r, d] and modifying all the jobs that interfere
    """
    for k in list(J.keys()):
        wk,rk,dk = J[k]
        
        new_rk = rk
        new_dk = dk

        if rk <= r and dk >= d:
            new_dk = dk - (d - r)
        elif rk <= r and dk >= r and dk <= d:
            new_dk = r
        elif rk >= r and rk <= d and dk >= d:
            new_rk = r
            new_dk = dk - (d - r)
        elif rk >= d and dk >= d:
            new_rk = rk - (d - r)
            new_dk = dk - (d - r)

        if new_dk <= new_rk:
            del J[k] # Remove job if its interval becomes invalid or zero
        else:
            J[k] = (wk, new_rk, new_dk)

def computeEnergyIntegerSpeedList(speed_list, alpha):

    energy = sum([(speed**alpha) for speed in speed_list])
    return energy

def computeEnergy(speed_list, alpha):
    
    energy = sum([ (interval[1] - interval[0]) * (speed_list[interval])**alpha for interval in speed_list.keys()])
    return energy

def get_speed(speed_dict, t):
    speed = 0
    intervals = speed_dict.keys()
    for interval in intervals:
        start, end = interval
        if (t >= start) and (t <= end):
            speed = speed_dict[interval]
            break
    return speed

def compute_speed(J_Sol):
    speed_lst = {}
    for id in J_Sol.keys():
        _, speed, start, end = J_Sol[id]
        speed_lst[(start, end)] = speed
    return speed_lst

def intersecting_intervals(speed_list, interval):
    interval_start = interval[0]
    interval_end = interval[1]
    if interval_start >= interval_end:
        print("There is a problem in input of the function update_intervals")
        exit(-1)
    
    intrevals_to_update_list = []

    for interval_to_update in speed_list.keys():
        start = interval_to_update[0]
        end = interval_to_update[1]
        intersects = not((start >= interval_end) or (end <= interval_start))
        if intersects: 
            intrevals_to_update_list.append(interval_to_update)
        
    return intrevals_to_update_list

def modify_speedlist(speed_list, intersectings, interval):
    start_interval = interval[0]
    end_interval = interval[1]

    for intersecting in intersectings:
        speed = speed_list[intersecting]
        start_intersecting = intersecting[0]
        end_intersecting = intersecting[1]
        if start_interval <= start_intersecting and end_interval >= end_intersecting:
            continue
        
        elif start_interval >= start_intersecting and end_interval <= end_intersecting:
            interval1 = (start_intersecting, start_interval)
            interval2 = (start_interval, end_interval)
            interval3 = (end_interval, end_intersecting)
            del speed_list[intersecting]
            if start_intersecting < start_interval:
                speed_list[interval1] = speed
            if start_interval < end_interval:
                speed_list[interval2] = speed
            if end_interval < end_intersecting:
                speed_list[interval3] = speed
        
        elif start_interval < start_intersecting and end_interval > start_intersecting and end_interval < end_intersecting:
            interval1 = (start_intersecting, end_interval)
            interval2 = (end_interval, end_intersecting)
            del speed_list[intersecting]
            if start_intersecting < end_interval:
                speed_list[interval1] = speed
            if end_interval < end_intersecting:
                speed_list[interval2] = speed
            
        elif start_intersecting < start_interval and start_interval < end_intersecting and end_intersecting < end_interval:
            interval1 = (start_intersecting, start_interval)
            interval2 = (start_interval, end_intersecting)
            del speed_list[intersecting]
            if start_intersecting < start_interval:
                speed_list[interval1] = speed
            if start_interval < end_intersecting:
                speed_list[interval2] = speed
            
def add_speed(speed_list, speed, interval):
    start_of_update = interval[0]
    end_of_update = interval[1]

    intersecting_intervals_list = intersecting_intervals(speed_list, interval)

    modify_speedlist(speed_list, intersecting_intervals_list, interval)

    speed_list_keys = sorted(speed_list.keys(), key=lambda x: x[0])

    for interval_to_update in speed_list_keys:
        start = interval_to_update[0]
        end = interval_to_update[1]
        if start_of_update <= start and end <= end_of_update:
            speed_to_increase = speed_list[interval_to_update]
            del speed_list[interval_to_update]
            new_speed = speed_to_increase + speed
            speed_list[interval_to_update] = new_speed
        elif end_of_update <= start or start_of_update >= end:
            continue
        else:
            print("This is a problem")
            print("start of update  = ", start_of_update, "---", "end of update = ", end_of_update)
            print("start = ", start, "---", "end = ", end)
            raise ValueError

def scale_speed(speed_list, mul_factor, interval):
    start_of_update = interval[0]
    end_of_update = interval[1]

    intersecting_intervals_list = intersecting_intervals(speed_list, interval)

    modify_speedlist(speed_list, intersecting_intervals_list, interval)

    speed_list_keys = sorted(speed_list.keys(), key = lambda x: x[0])

    for interval_to_update in speed_list_keys:
        start = interval_to_update[0]
        end = interval_to_update[1]

        if start_of_update <= start and end <= end_of_update:
            speed_to_increase = speed_list[interval_to_update]
            del speed_list[interval_to_update]
            new_speed = speed_to_increase * mul_factor
            speed_list[interval_to_update] = new_speed
        elif end_of_update <= start or start_of_update >= end:
            continue
        
        else:
            print("This is a problem")
            print("start of update  = ", start_of_update, "---", "end of update = ", end_of_update)
            print("start = ", start, "---", "end = ", end)
            raise ValueError

def round_instance(J, epsilon):
    w, r, d = J[1]
    T = d - r

    ids = sorted(J.keys())
    intervals = []
    for id in ids:
        w, r, d = J[id]
        r_new = epsilon * T * ceil(float(r) / (epsilon * T))
        d_new = r_new  + (1 - epsilon) * T
        del J[id]
        J[id] = (w, r_new, d_new)
        interval = (r_new, d_new)
        intervals.append(interval)

    intervals = sorted(list(set(intervals)), key = lambda x: x[0])

    J_new = {}

    for interval in intervals:
        J_new[interval] = 0
    for id in ids:
        w, r, d = J[id]            
        interval  = (r, d)
        J_new[interval] += w
    del J

    J = {}
    for id in range(0, len(intervals)):
        interval = intervals[id]
        r, d = interval
        w = J_new[interval]
        J[id + 1] = (w, r, d)
    
    return J

def print_speed_list(speed_list):
    speed_list_keys = sorted(speed_list.keys(), key = lambda x: x[0])
    for interval in speed_list_keys:
        start = interval[0]
        end = interval[1]
        speed = speed_list[interval]
        print(start, "---", end, "--->speed = ", speed)

def compute_speed_per_integer_time(J_Sol):
    ids = sorted(J_Sol.keys())

    _, speed, start_of_time, _ = J_Sol[ids[0]]
    _, _, _, end_of_time = J_Sol[ids[1]] if len(J_Sol.keys()) > 2 else J_Sol[ids[-1]]

    start_of_time = int(start_of_time)
    end_of_time = int(end_of_time)

    start = start_of_time 
    speed_list = [Fraction(1, 1)]*int(end_of_time - start_of_time)
    for id in ids:
        w_id, s_id, start_id, end_id = J_Sol[id]
        floored_start_of_interval = floor(start_id)
        ceiled_end_of_interval = ceil(end_id)
        speed_list[floored_start_of_interval:ceiled_end_of_interval] = [s_id] * (ceiled_end_of_interval - floored_start_of_interval)
    
    return speed_list

def is_sol_correct(J_sol, J):
    ids = sorted(J.keys())
    speed_list = compute_speed_per_integer_time(J_sol)
    prev_processing_end = 0.0
    for id in ids:
        w, r, d = J[id]
        w_sol, speed_sol, start, end = J_sol[id]
        validity = (w_sol == w) and (r <= start) and (end <= d) and (start < end) and speed_sol == Fraction(w, (end - start)) and (start >= prev_processing_end)
        if not validity:
            print(J[id])
            print(J_sol[id])
            print("start = ", start, "should be more than previous processing ends = ", prev_processing_end)
            exit()
        prev_processing_end = end

        r = int(r)
        d = int(d)

        optimality = all([(speed_list[t] >= speed_sol) for t in range(r, d)])
        if not optimality:
            print(J_sol[id])
            print(r, d)
            print(optimality)
            exit()
        
    print("congratulations, your algorithm is great")
    return 0

def find_densest_interval(J):
    arrivals, deadlines = findArrivalsDeadlines(J)

    maximum_density = Fraction(0, 100)
    best_ids = []
    best_arrival = 0
    best_deadline = 0
    for arrival in arrivals:
        for deadline in deadlines:
            if arrival == deadline:
                continue
            total_weight, ids = findWeight(J, arrival, deadline)

            density = Fraction(total_weight, deadline - arrival)
            if density >= maximum_density: 
                best_ids = ids
                maximum_density = density
                best_arrival = arrival
                best_deadline = deadline
    return maximum_density, best_ids, best_arrival, best_deadline

def compute_optimal_instance(J):
    J_save = copy.deepcopy(J)
    J_sol = {}
    for id in J.keys():
        w, r, d = J[id]
        r_id = Fraction(r, 1)
        d_id = Fraction(d, 1)
        speed_id = Fraction(0, 1)
        J_sol[id] = (w, speed_id, r_id, d_id)

    while J:
        speed, ids, r, d = find_densest_interval(J)
        for id in ids:
            speed_id = speed
            w_id, _, r_id, d_id = J_sol[id]
            J_sol[id] = (w_id, speed_id, r_id, d_id)
        adjustInst(J, r, d)
    
    ids = sorted(J_sol.keys())
    previous_deadline = Fraction(0, 1)
    for id in ids:
        w_id, speed_id, r_id, d_id = J_sol[id]
        start = previous_deadline
        if w_id == 0 and speed_id == 0:
            end = start
        else:
            end = Fraction(w_id, speed_id) + start
        previous_deadline = end
        J_sol[id] = (w_id, speed_id, start, end)
    return J_sol

def print_solution(J_sol):
    ids = sorted(J_sol.keys())
    for id in ids:
        w_id, speed_id, r_id, d_id = J_sol[id]
        srt_to_print = str(id) + "/weight=" + str(w_id) + "/speed=" + str(float(speed_id)) + "/start=" + str(float(r_id)) + "/end=" + str(float(d_id))
        print(srt_to_print)
    
def Avg_rate(J):
    ids = sorted(J.keys())
    _, _start, _ = J[ids[0]]
    _, _, _end = J[ids[-1]]
    start = _start
    end = _end
    speed_list_dict = {}
    speed_list_dict[(start, end)] = Fraction(0,1)
    for id in ids:
        w_id, r, d = J[id]
        speed = Fraction(w_id, d - r)
        r_id = Fraction(r, 1)
        d_id = Fraction(d, 1)
        interval = (r_id, d_id)
        add_speed(speed_list_dict, speed, interval)
    return speed_list_dict

def Avg_rate_integer_speedlist(J):
    ids = sorted(J.keys())
    _, start, _ = J[ids[0]]
    _, _, end = J[ids[-1]]
    start = int(start)
    end = int(end)

    speed_list = [Fraction(0,1)]*(end - start)
    for id in ids:
        w_id, r, d = J[id]
        r = int(r)
        d = int(d)
        speed = Fraction(w_id, d - r)
        for t in range(r, d):
            speed_list[t] += speed
    return speed_list

def densest_interval_BKP(J, t, dt):
    key = sorted(J.keys())
    min_time = J[key[0]][1]
    max_time = J[key[-1]][2]
    if t < min_time: return 0
    if t > max_time: return 0
    time_arg = t + dt
    max_density = 0
    bound = (e/(e-1)) * float(t) - float(min_time)/(e - 1)
    bound = max(bound , float(max_time))
    while float(time_arg) < 2 * bound:
        a = e*t - ((e - 1) * time_arg)
        b = time_arg
        total_weight_released = 0
        i = key[0]
        while i <= key[-1]:
            if ((J[i][1] <= t) and (J[i][1] >= a) and (J[i][2] <= b)):
                total_weight_released += J[i][0]
            
            i += 1
        
        density = total_weight_released / (e * (time_arg - t))

        if(density > max_density):
            max_density = density
        time_arg += dt
    return max_density

def robustify(speed_to_make_robust, epsilon, T, dt):
    dim = int(float(epsilon * T) / float(dt)) + 1
    mask_val = 1.0 / float(dim)
    mask = [mask_val]*dim
    length_of_the_solution = len(speed_to_make_robust)
    speed_to_make_robust = np.array(speed_to_make_robust)
    mask = np.array(mask)
    speed = np.convolve(mask, speed_to_make_robust)

    return speed

def scale_down_epsilon(epsilon, alpha, error):
    search_granularity = 1000000
    epsilons = np.linspace(0, 0.9, search_granularity)
    for new_epsilon in epsilons:
        diff = ((1 + new_epsilon) / (1 - new_epsilon))**alpha - (1 + epsilon)
        if abs(diff) < error:
            break
    new_epsilon = Fraction.from_float(new_epsilon).limit_denominator()
    return new_epsilon

def speed_dict_to_list(speed_dict, dt, mul_factor):
    intervals = list(speed_dict.keys())
    intervals.sort(key = lambda x: x[1])

    for interval in intervals:
        old_speed = speed_dict[interval]
        new_speed = old_speed * mul_factor
        del speed_dict[interval]
        speed_dict[interval] = new_speed

    s = []
    t = 0
    for interval in intervals:
        start, end = interval
        speed = speed_dict[interval]
        while t < end:
            s.append(float(speed)) # Convert Fraction to float here
            t+=dt
    return s

# Classification function to set the job in their respective classes
# Dispatch function to assign a processor to the job

def classify_jobs(J, pred=False):
    """
    This function classifies all jobs in jobset J, with respect to density(weight / interval length), weight and confidence of the prediction
    Assume we already know the maximum weight(w_max), and maximum density(d_max) (Can be calculated in a different function as well)
    Job j belongs to class C_k,h if 
        w_j /in (w_max / 2^(k+1), w_max / 2^(k)]
        density_j /in [density_max / 2 ^ h, density_max / 2 ^ (h + 1))
    The expected output can be any ds but make sure to use it in order to dispatch jobs from these classes
    
    Args:
        J: Dictionary of jobs {job_id: (weight, release_time, deadline, confidence)}
        confThreshold: Confidence threshold for classification (default: None)
    
    Returns:
        Dictionary mapping class keys (k,h) to lists of job IDs
    """
    # Input validation
    if not isinstance(J, dict):
        raise ValueError("J must be a dictionary")
    
    if not J:
        return {}
    
    # Calculate maximum weight and density
    max_weight = max(job[0] for job in J.values())
    max_density = 0
    
    # Calculate density for each job and find maximum
    job_densities = {}
    for job_id, job_data in J.items():
        # if not isinstance(job_data, (tuple, list)) or len(job_data) < 3:
            # raise ValueError(f"Job {job_id} must have at least 3 values (weight, release_time, deadline)")
        if pred:
            weight, release_time, deadline, _ = job_data[0], job_data[1], job_data[2], job_data[3]
        else:
            weight, release_time, deadline = job_data[0], job_data[1], job_data[2]
        
        if deadline <= release_time:
            raise ValueError(f"Job {job_id}: deadline must be greater than release time")
        
        density = weight / (deadline - release_time)
        job_densities[job_id] = density
        max_density = max(max_density, density)
    
    if max_density == 0:
        raise ValueError("All jobs have zero density")
    
    # Initialize classification dictionary
    classes = {}
    
    # Classify each job
    for job_id, job_data in J.items():
        if pred:
            weight, release_time, deadline, _ = job_data[0], job_data[1], job_data[2], job_data[3]
        else:
            weight, release_time, deadline = job_data[0], job_data[1], job_data[2]

        density = job_densities[job_id]
        
        # Determine confidence level (l)
        # confidence = job_data[3] if len(job_data) > 3 else 1.0  # Default confidence is 1.0
        # l = 1 if confThreshold is None or confidence > confThreshold else 0
        
        # Find k such that w_j /in (w_max / 2^(k+1), w_max / 2^(k)]
        k = 0
        kmax =  int(log2(max_weight))
        while k < kmax:  # Limit to prevent infinite loop
            lower_bound = max_weight / (2 ** (k + 1))
            upper_bound = max_weight / (2 ** k)
            if lower_bound < weight <= upper_bound:
                break
            k += 1
        
        # Find h such that density_j /in [density_max / 2 ^ h, density_max / 2 ^ (h + 1))
        h = 0
        hmax = int(log2(max_density))
        while h < hmax:  # Limit to prevent infinite loop
            lower_bound = max_density / (2 ** h)
            upper_bound = max_density / (2 ** (h + 1))
            if lower_bound <= density < upper_bound:
                break
            h += 1
        
        # Add job to appropriate class
        class_key = (k, h)
        if class_key not in classes:
            classes[class_key] = []
        classes[class_key].append(job_id)
    
    return classes

def dispatch_jobs(classes_kh, J, m):
    """
    Here, all jobs belonging to class C_k,h,1 from above will be dispatched in a round robin manner
    and C_k,h,0 will be dispatched in a reverse round robin manner, first arrive, first dispatch manner
    What this function does is bind that job to this processor
    m is the number of processors
    """
    if not classes_kh:
        return
    
    if not isinstance(classes_kh, dict):
        raise ValueError("J must be a dictionary")
    
    if not isinstance(m ,int) or m <= 0:
        raise ValueError("m must be positive integer")
    
    processor_assignments = {i: [] for i in range(m)}

    for _, job_ids in classes_kh.items():

        sorted_ids = sorted(job_ids, key = lambda job_id : J[job_id][1])
        
        for i, job_id in enumerate(sorted_ids):
            processor_id = i % m
            processor_assignments[processor_id].append(job_id)
        
    return processor_assignments

def Pred_dispatch_jobs(Classes_kh, J, m, confThreshold):
    """
    Here, all jobs belonging to class C_k,h,1 from above will be dispatched in a round robin manner
    and C_k,h,0 will be dispatched in a reverse round robin manner, first arrive, first dispatch manner
    What this function does is bind that job to this processor
    m is the number of processors
    
    Args:
        Classes_khl: Dictionary mapping class keys (k,h,l) to lists of job IDs
        J: Dictionary of jobs {job_id: (weight, release_time, deadline, confidence)}
        m: Number of processors
    
    Returns:
        Dictionary mapping processor IDs to lists of assigned job IDs
    """
    if not Classes_kh:
        return {}
    
    # Input validation
    if not isinstance(Classes_kh, dict):
        raise ValueError("Classes_kh must be a dictionary")
    
    if not isinstance(J, dict):
        raise ValueError("J must be a dictionary")
    
    if not isinstance(m, int) or m <= 0:
        raise ValueError("m must be a positive integer")
    
    # Initialize processor assignments
    processor_assignments = {i: [] for i in range(m)}
    
    # Process each class
    # For loop ordering for tuples - can be a problem
    for class_key, job_ids in Classes_kh.items():
        
        if not job_ids:
            continue
        
        high_confidence_jobs = []
        low_confidence_jobs = []

        for job_id in job_ids:
            if J[job_id][3] > confThreshold:
                high_confidence_jobs.append(job_id)
            else:
                low_confidence_jobs.append(job_id)
        
        # Sort high and low confidence jobs by release time for consistent dispatch order
        high_confidence_jobs = sorted(high_confidence_jobs, key=lambda job_id: J[job_id][1])
        low_confidence_jobs = sorted(low_confidence_jobs, key=lambda job_id: J[job_id][1])

        for i, job_id in enumerate(high_confidence_jobs):
            processor_id = i % m
            processor_assignments[processor_id].append(job_id)
        
        for i, job_id in enumerate(low_confidence_jobs):
            processor_id = m - 1 - (i % m)
            processor_assignments[processor_id].append(job_id)

    return processor_assignments