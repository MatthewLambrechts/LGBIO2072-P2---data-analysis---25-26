import numpy as np
import scipy.io as spio
import csv
import numpy as np
import matplotlib.pyplot as plt

"""
Loading the different files and storing them in a dictionnary. See the project 
instruction file available on moodle for a description of the data dictionaries. 

The folders dataNeuron, dataMuscle and dataAdaptation must be in the same folder as 
this script.
dataP2 : ___ loadDataP2.py
        |___ dataNeuron : *.mat
        |___ dataMuscle : *.mat
        |___ dataAdaptation : *.mat
               
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

def loadNeuron():
    
    namesSignals= [
        ('time'    ),
        ('shoang'  ),
        ('elbang'  ),
        ('handxpos'),
        ('handypos'),
        ('cells'   )]
    
    dictNeurons = {}
    for targetNum in range(1,9):
            
        target = {}
            
        for trialNum in range(1,7):
            trial = {}
            for nam in namesSignals:
                key = nam
                value = spio.loadmat('dataNeuron/target'+str(targetNum)+'trial' + str(trialNum) + 'signals'+nam+'.mat')
                trial[key]=value['a']
                
            target['trial'+str(trialNum)] = trial
    
        dictNeurons['target'+str(targetNum)] = target
        
    return dictNeurons
    
dictNeurons = loadNeuron()

def loadAdaptation():
    dictAdaptation = {}
    participants = ["ev", "fs", "ge", "jc", "jl"]
    for j in participants:
        value_x = j+'_x'
        data_x = spio.loadmat(value_x + '.mat')
        
        value_y = j+'_y'
        data_y = spio.loadmat(value_y + '.mat')
        
        dictAdaptation[value_x] = data_x[value_x]
        dictAdaptation[value_y] = data_y[value_y]
        
    sequence = spio.loadmat("sequence.mat")
    dictAdaptation["sequence"] = sequence["sequence"]
    return dictAdaptation

dictAdaptation = loadAdaptation()