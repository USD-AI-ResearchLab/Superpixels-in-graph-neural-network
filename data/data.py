"""
    File to load dataset based on user control from main file
"""
from data.superpixels import SuperPixDataset


def LoadData(DATASET_NAME):
    """
        This function is called in the main.py file 
        returns:
        ; dataset object
    """
   
    # handling for MNIST or CIFAR Superpixels
    if DATASET_NAME in ['LUNGS_75','LUNGS_200','LUNGS_300','LUNGS_400']:
        return SuperPixDataset(DATASET_NAME)
    
    else:
        return "No Data as Such"
    
    