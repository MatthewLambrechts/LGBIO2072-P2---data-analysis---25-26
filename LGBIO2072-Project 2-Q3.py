import numpy as np
import scipy.io as spio
import csv
import numpy as np
import matplotlib.pyplot as plt

def loadAdaptation():
    dictAdaptation = {}
    participants = ["ev", "fs", "ge", "jc", "jl"]
    for j in participants:
        value_x = j+'_x'
        data_x = spio.loadmat('loadDataP2/dataAdaptation/' + value_x + '.mat')
        
        value_y = j+'_y'
        data_y = spio.loadmat('loadDataP2/dataAdaptation/'+ value_y + '.mat')
        
        dictAdaptation[value_x] = data_x[value_x]
        dictAdaptation[value_y] = data_y[value_y]
        
    sequence = spio.loadmat("loadDataP2/dataAdaptation/sequence.mat")
    dictAdaptation["sequence"] = sequence["sequence"]
    return dictAdaptation

dictAdaptation = loadAdaptation()

print (dictAdaptation.keys())
x_pos_1 = dictAdaptation['ev_x']
y_pos_1 = dictAdaptation['ev_y']

x_pos_2 = dictAdaptation['fs_x']
y_pos_2 = dictAdaptation['fs_y']

x_pos_3 = dictAdaptation['ge_x']
y_pos_3 = dictAdaptation['ge_y']

x_pos_4 = dictAdaptation['jc_x']
y_pos_4 = dictAdaptation['jc_y']

x_pos_5 = dictAdaptation['jl_x']
y_pos_5 = dictAdaptation['jl_y']

seq = dictAdaptation['sequence']


seq_array = np.array(seq)

seq_flat = seq_array.flatten()
seq_zero_based = seq_flat - 1

reorder_indices = np.argsort(seq_zero_based)

print(f"La taille de l'array d'indices est : {reorder_indices.shape}")
print(f"Les 10 premiers indices pour réordonner les données : \n{reorder_indices[:10]}")

# 3. Appliquer l'ordre aux données (Exemple avec 'ev_x')
# Récupérez une de vos matrices (180 lignes, N colonnes de temps)
# (Ici, on doit simuler ev_x car je n'ai pas le fichier .mat)
# Simuler ev_x : 180 essais, 100 points de temps
ev_x_dummy = np.random.rand(180, 100) 
ev_x_dummy[10, :] = 9999.0 # Ligne 10 (qui correspond à l'Essai n°100)
ev_x_dummy[0, :] = 0.0001  # Ligne 0 (qui correspond à l'Essai n°1)

# Réordonner
# ev_x_reordered[0, :] sera maintenant l'Essai n°1
# ev_x_reordered[99, :] sera maintenant l'Essai n°100
ev_x_reordered = x_pos_1[reorder_indices, :]
ev_y_reordered = y_pos_1[reorder_indices, :]

# Vérification (L'essai n°1 est l'ancien indice 0)
print("\n--- Vérifications ---")
print(f"Valeur de la première ligne avant réordonnancement (ancien essai n°1) : {x_pos_1[0, 0]:.4f}")
print(f"Valeur de la ligne 10 avant réordonnancement (ancien essai n°100) : {x_pos_1[10, 0]:.4f}")

# La première ligne (indice 0) des données réordonnées doit être l'ancien indice 0
print(f"Valeur de la PREMIÈRE ligne APRÈS réordonnancement (Essai n°1) : {ev_x_reordered[0, 0]:.4f}")
# La 100ème ligne (indice 99) des données réordonnées doit être l'ancien indice 10
print(f"Valeur de la CENTIÈME ligne APRÈS réordonnancement (Essai n°100) : {ev_x_reordered[99, 0]:.4f}")

plt.plot(ev_x_reordered[0], ev_y_reordered[0])
plt.show()
plt.plot(ev_x_reordered[179], ev_y_reordered[179])
plt.show()

plt.scatter(reorder_indices, ev_x_reordered[:,-1])
plt.show()

plt.scatter(reorder_indices, ev_y_reordered[:,-1])
plt.show()

# reordered_data est le dictionnaire avec les données triées
participant_data_x = ev_x_reordered # (180, N_points)
participant_data_y = ev_y_reordered 

num_trials = participant_data_x.shape[0]
peak_x = np.zeros(num_trials)
end_x = np.zeros(num_trials)

# I. Calculer les métriques pour les 180 essais
for i in range(num_trials):
    current_x = participant_data_x[i, :]
    
    # Position Latérale Finale (Endpoint): La dernière position X enregistrée
    end_x[i] = current_x[-1]
    
    # Position Latérale Maximale (Peak): Le plus grand écart par rapport à X=0
    # Si le FF pousse en X positif, nous cherchons le max positif.
    # Si le FF pousse la main dans une seule direction (par ex. X>0), on prend max.
    peak_x[i] = np.max(current_x)
    
# II. Tracer
trials = np.arange(1, num_trials + 1)

plt.figure(figsize=(10, 6))

# Lissage des données pour la Peak Lateral Trajectory (Ligne)
window = 5
smoothed_peak_x = np.convolve(peak_x, np.ones(window)/window, mode='valid')

# 1. Tracer la Position Latérale Maximale (LIGNE)
plt.plot(trials[window-1:], smoothed_peak_x, label='Peak Lateral Position (Lissée)', color='red')

# 2. Tracer la Position Latérale Finale (POINTS)
plt.scatter(trials, end_x, label='Lateral Endpoint Position (Brute)', color='blue', s=10, alpha=0.5)

plt.axhline(0, color='grey', linestyle='--', linewidth=1)
plt.xlabel('Numéro d\'Essai')
plt.ylabel('Position Latérale (m)')
plt.title('Mesures de l\'Erreur d\'Adaptation (Participant EV)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()