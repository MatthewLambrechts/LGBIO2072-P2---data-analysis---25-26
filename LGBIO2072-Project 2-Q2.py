import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import scipy.io as spio
import csv


"""
Loading the different files and storing them in a dictionnary. See the project 
instruction file available on moodle for a description of the data dictionaries. 

The folder dataNeuron and the folder dataMuscle must be in the same folder as 
this script.
dataP2 : ___ loadDataP2.py
        |___ dataNeuron : *.mat
        |___ dataMuscle : *.mat
               
"""

def loadMuscle():
    """
    Loading the different files and storing them in a dictionnary
    """
    
    
    HandX = spio.loadmat("dataMuscle/HandX.mat")
    HandXVel = spio.loadmat("dataMuscle/HandXVel.mat")
    HandXForce = spio.loadmat("dataMuscle/HandXForce.mat")
    
    HandY = spio.loadmat("dataMuscle/HandY.mat")
    HandYVel = spio.loadmat("dataMuscle/HandYVel.mat")
    HandYForce = spio.loadmat("dataMuscle/HandYForce.mat")
    
    Pectoralis = spio.loadmat("dataMuscle/Pectoralis.mat")
    Deltoid = spio.loadmat("dataMuscle/Deltoid.mat")
    
    extracted = spio.loadmat("dataMuscle/extracted.mat")
    descriptions = spio.loadmat("dataMuscle/descriptions.mat")
    
    
    """ 
    Creation of the first dictionnary - strMuscles
    """
    
    dictMuscles = {"HandX": HandX["HandX"],
                  "HandXVel": HandXVel["HandXVel"],
                  "HandXForce": HandXForce["HandXForce"],
                  "HandY": HandY["HandY"],
                  "HandYVel": HandYVel["HandYVel"],
                  "HandYForce": HandYForce["HandYForce"],
                  "Pectoralis": Pectoralis["Pectoralis"],
                  "Deltoid": Deltoid["Deltoid"],
                  "extracted": extracted["extracted"],
                  "descriptions": descriptions["descriptions"]}

    return dictMuscles

dictMuscles = loadMuscle()






def plot_trials(data_dict, signals, trial_types, trial_indices=None, average=False):
    """
    Trace les courbes pour des données spécifiques, en comparant différents types de trials.
    
    Arguments :
    - data_dict : dict, dictionnaire contenant les signaux (ex: dictMuscles)
    - signals : list of str, liste des clés des signaux à tracer (ex: ["Pectoralis", "HandX", "HandY"])
    - trial_types : list of int, liste des types d'essais à comparer (ex: [1, 2])
    - trial_indices : list of int or None, index des essais spécifiques à tracer pour chaque type (si None, on calcule la moyenne)
    - average : bool, si True, calcule la moyenne des essais pour chaque type
    """
    if trial_indices is None:
        trial_indices = [None] * len(trial_types)
    elif len(trial_indices) != len(trial_types):
        raise ValueError("La liste trial_indices doit avoir la même longueur que trial_types.")
    
    plt.figure(figsize=(10, 6))
    
    for signal in signals:
        for i, trial_type in enumerate(trial_types):
            trial_index = trial_indices[i]
            extracted = data_dict["extracted"]
            if trial_type == 1:
                trial_mask = extracted[:, 2] == 1  # Essais normaux
            elif trial_type == 2:
                trial_mask = extracted[:, 2] == 2  # Essais perturbés (Fx = +13 * y)
            elif trial_type == 3:
                trial_mask = extracted[:, 2] == 3  # Essais perturbés (Fx = -13 * y)
            
            signal_data = data_dict[signal]
            filtered_trials = signal_data[trial_mask, :]
            
            if average:
                to_plot = filtered_trials.mean(axis=0)
                label = f"{signal} - Average (Type {trial_type})"
            else:
                if trial_index is None or trial_index >= filtered_trials.shape[0]:
                    raise ValueError(f"Index d'essai invalide ou non spécifié pour le type {trial_type}.")
                to_plot = filtered_trials[trial_index, :]
                label = f"{signal} - Trial {trial_index+1} (Type {trial_type})"
            
            # Définir les axes X selon le signal
            if signal in ["HandX", "HandY"]:
                if "HandX" in signals and "HandY" in signals:  # Pour des trajectoires en 2D
                    continue
                time_or_x = np.linspace(0, 1.2, 1200)
            else:
                time_or_x = np.linspace(0, 1.2, 1200)
            
            plt.plot(time_or_x, to_plot, label=label)
    
    # Gestion spéciale pour les trajectoires (HandX, HandY)
    if "HandX" in signals and "HandY" in signals:
        for i, trial_type in enumerate(trial_types):
            trial_index = trial_indices[i]
            if trial_type == 1:
                trial_mask = extracted[:, 2] == 1  # Essais normaux
            elif trial_type == 2:
                trial_mask = extracted[:, 2] == 2  # Essais perturbés (Fx = +13 * y)
            elif trial_type == 3:
                trial_mask = extracted[:, 2] == 3  # Essais perturbés (Fx = -13 * y)
            
            handx = data_dict["HandX"][trial_mask, :]
            handy = data_dict["HandY"][trial_mask, :]
            
            if average:
                plt.plot(handx.mean(axis=0), handy.mean(axis=0), label=f"Average trajectory (Type {trial_type})")
            else:
                plt.plot(handx[trial_index, :], handy[trial_index, :], label=f"Trajectory of (Type {trial_type} and trial {trial_index+ 1})")
        
        plt.xlabel("Position X (m)")
        plt.ylabel("Position Y (m)")
    else:
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude ")
    
    plt.title(f"Data visualization of the EMG activity and HandXForce of type {trial_types}")
    plt.legend()
    plt.grid()
    plt.show()

#plot_trials( dictMuscles, signals=["HandX","HandY"], trial_types=[3,3,3], trial_indices=[0,14,29])  # Moyenne des types 1 et 3
#plot_trials( dictMuscles, signals=["HandXVel","HandXForce"], trial_types=[2], average=True)  # Essai spécifique pour types 1 et 2

#plot_trials( dictMuscles, signals=["Pectoralis"], trial_types=[1,2, 3], average=True)  # Essai spécifique pour types 1 et 3
plot_trials( dictMuscles, signals=["Deltoid","HandXForce"], trial_types=[2,2], trial_indices=[0,29])  # Moyenne des types 1 et 3

#plot_trials( dictMuscles, signals=["HandXVel"], trial_types=[2,1,3], trial_indices=[10,10,10])  # Moyenne des types 1 et 3

#plot_trials( dictMuscles, signals=["HandXForce","Deltoid","Pectoralis"], trial_types=[1], average=True)  # Moyenne des types 1 et 3

#plot_trials( dictMuscles, signals=["HandYVel","Deltoid","Pectoralis"], trial_types=[3], average=True) 
#plot_trials( dictMuscles, signals=["HandYVel"], trial_types=[1,1,2,2,3,3], trial_indices=[0,10,0,10,0,10])  # Moyenne des types 1 et 3



