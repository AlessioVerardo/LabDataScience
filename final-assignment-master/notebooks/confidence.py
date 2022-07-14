import numpy as np
from scipy.stats import norm
from datetime import datetime


def prob_to_arrive_before_exp(mean, time):
    """
    Probabiltiy that an exponential random variable is smaller than time when it has the given mean
    
    :param time: the time to compute the probability
    :param mean: the estimated mean from the data
    :return: the probability to arrive before the given time
    """
    if mean == 0.0:
        return 1.0
    else:
        return 1.0 - np.exp(- float(time) / float(mean))
    

def prob_to_arrive_before_gauss(mean, std, time):
    """
    Probabiltiy that an Gaussian(mean, std^2) random variable is smaller than time when it has the given mean and std
    
    :param time: the time to compute the probability
    :param mean: the estimated mean from the data
    :param std: the estimated std from the data
    :return: the probability to arrive before the given time
    """
    return norm.cdf(time, loc=mean, scale=std)
    


def path_confidence(df_edges, free_time):
    """
    Compute the probability that the user can take all the correspondences
    on the path given by the sequence of edges.
    
    :param df_edges: df pandas of sorted edges (start to end) taken by a path
    :param T: arrival time limit
    :return the probability that the user arrives at destination on time
    """
    # Probability of success (can take all the correspondences) for the path
    p = 1.0
    
    # Free time to spend on potential delays
    free = free_time
    
    # Current line number of the transport type (same curr_line for same train)
    curr_line = ''
    
    # Start from the destination
    edges_list = list(df_edges.iterrows())
    
    for _, e in edges_list:
        # If this edge is a walking or waiting edge
        if e['is_trip'] == 0:
            
            # If this edge is a connection to another train/bus/etc
            if e['line_number'] == '-1':
                
                free += e['waiting_time']
                #print(f"waiting time : {e['waiting_time']}")
                curr_line = ''
        
        # If this edge is a transport edge
        else:
            # We changed of transport or went from walking and hop in a transport
            if e['line_number'] != curr_line:
                # Proba to get the correspondence
                #p = p * prob_to_arrive_before_exp(e['mean_delay'], free)
                p = p * prob_to_arrive_before_gauss(e['mean_delay'], e['std_delay'], free)
                
                # No more free time to spend on delay
                free = 0
                
                # 
                curr_line = e['line_number']
                
    return p