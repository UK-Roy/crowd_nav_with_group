import numpy as np

def update_cluster(human_num, index_dict):
    """
    Creates and updates a 1D numpy array based on a dictionary of indices and values.
    Indices not specified in the dictionary are set to -1.
    
    Parameters:
        array_size (int): The size of the 1D numpy array to be created.
        index_dict (dict): A dictionary where keys are the values to place in the array,
                           and the values are lists of indices to update with the key.
    
    Returns:
        np.ndarray: The updated array.
    """
    # Initialize the array with -1
    array = np.full(human_num, -1, dtype=np.int32)
    
    # Update the array with the values from the dictionary
    for key, indices in index_dict.items():
        array[indices] = key

    return array