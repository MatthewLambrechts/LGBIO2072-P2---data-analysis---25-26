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

def reorder_all_participants(dictAdaptation):
    seq = dictAdaptation['sequence']
    seq_flat = np.array(seq).flatten()

    seq_base = seq_flat-1

    reorder_indices = np.argsort(seq_base)

    participants = ["ev", "fs", "ge", "jc", "jl"]
    dictReordered = {}

    for p in participants :
        x = dictAdaptation[f"{p}_x"]
        y = dictAdaptation[f"{p}_y"]

        x_reord = x[reorder_indices, :]
        y_reord = y[reorder_indices, :]

        dictReordered[p] = {
            "x" : x_reord,
            "y" : y_reord
        }
    dictReordered["sequence"] = np.arange(1, len(reorder_indices)+1)

    return dictReordered

dictReorder = reorder_all_participants(dictAdaptation = dictAdaptation)

def compute_peak_lateral(X):

    peak_x = np.max(X, axis=1)
    return peak_x

def plot_peak_adaptation(X, participant_label="ev", window=8):
    peak_x = compute_peak_lateral(X)
    n = len(peak_x)
    trials = np.arange(1, n+1)

    smooth = np.convolve(peak_x, np.ones(window)/window, mode = "valid")
    t_smooth = np.arange(window//2 + 1, window//2 + 1 + len(smooth))

    plt.figure(figsize=(10,5))
    plt.scatter(trials, peak_x, s=12, alpha=0.4, label = 'lateral trajectory peaks')

    plt.plot(t_smooth, smooth, linewidth=2.5, color = 'red', label = f"Lateral peak (smoothed, window ={8})")
    plt.axhline(0, color="grey", linestyle="--", linewidth=1)
    plt.xlabel("Number of trial")
    plt.ylabel("Maximal lateral deviation (m)")
    plt.title(f"Adaptation to the force field - lateral peak {participant_label}")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.show()


def mean_traj_with_std(X_al, Y_al, start, end):
    """
    Calcule la moyenne et la std latérale dans un bloc d'essais.
    """
    Xb = X_al[start:end, :]    # (n_block_trials, n_points)
    Yb = Y_al[start:end, :]

    mean_x = Xb.mean(axis=0)
    std_x  = Xb.std(axis=0)
    mean_y = Yb.mean(axis=0)

    return mean_x, std_x, mean_y


def align_trials_Y_only(X, Y):
    """
    Aligne les trajectoires seulement sur Y, pas sur X.
    """

    Y0 = Y[:, [0]]             # (n_trials, 1)
    Y_al = Y - Y0              # alignement
    X_al = X.copy()            # X intact

    return X_al, Y_al


def plot_mean_trajs_3_blocks(X, Y, participant_label="ev"):

    X_al, Y_al = align_trials_Y_only(X, Y)

    blocks = [
        (0, 30,   "Essais 1-30"),
        (30, 150, "Essais 31-150"),
        (150, 180, "Essais 151-180"),
    ]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    plt.figure(figsize=(6,6))

    for (start, end, label), col in zip(blocks, colors):
        mean_x, std_x, mean_y = mean_traj_with_std(X_al, Y_al, start, end)

        # Mean trajectory
        plt.plot(mean_x, mean_y, color=col, label=label, linewidth=2)

        # ± std (uniquement en X, car c’est la variable intéressante)
        plt.fill_betweenx(mean_y, mean_x - std_x, mean_x + std_x,
                          color=col, alpha=0.2)

    plt.title(f"Mean trajectories aligned {participant_label}")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.show()

def get_group_peak_matrix(dict) :
    participants = ["ev", "fs", "ge", "jc", "jl"]
    peaks=[]

    for p in participants : 
        X = dict[p]["x"]
        peaks.append(compute_peak_lateral(X))
    
    return np.array(peaks) #shape = (5,180)

def plot_group_peak_adaptation(group_peaks, window=8):
    """
    Plot d'adaptation groupé : mean ± SEM sur les participants.
    """
    n_participants, n_trials = group_peaks.shape

    mean_peak = group_peaks.mean(axis=0)
    sem_peak  = group_peaks.std(axis=0) / np.sqrt(n_participants)

    # Lissage (optionnel mais joli)
    smooth_mean = np.convolve(mean_peak, np.ones(window)/window, mode="valid")
    smooth_sem  = np.convolve(sem_peak,  np.ones(window)/window, mode="valid")
    t_smooth = np.arange(window//2 + 1, window//2 + 1 + len(smooth_mean))

    plt.figure(figsize=(10, 5))

    # Bande SEM
    plt.fill_between(n_trials,
                     mean_peak - sem_peak,
                     mean_peak + sem_peak,
                     color="orange", alpha=0.3,
                     label="± SEM (5 participants)")

    # Courbe moyenne lissée
    plt.plot(n_trials, mean_peak, color="red", linewidth=2.5,
             label="Mean peak lateral")

    plt.axhline(0, color="grey", linestyle="--", linewidth=1)
    plt.xlabel("Trial number")
    plt.ylabel("Maximal lateral deviation (m)")
    plt.title("Adaptation to the force field - Group average (5 participants)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.savefig(".../plots")
    plt.show()

def compute_group_block_mean_traj(dictReorder, start, end):
    """
    Calcule la trajectoire moyenne de groupe (5 participants)
    pour un bloc d'essais [start:end].

    On fait :
    - alignement Y-only par participant
    - moyenne par bloc pour chaque participant
    - puis moyenne de ces trajectoires sur les participants
    """

    participants = ["ev", "fs", "ge", "jc", "jl"]

    all_mean_x = []
    all_mean_y = []

    for p in participants:
        X = dictReorder[p]["x"]
        Y = dictReorder[p]["y"]

        X_al, Y_al = align_trials_Y_only(X, Y)
        mean_x, std_x, mean_y = mean_traj_with_std(X_al, Y_al, start, end)

        all_mean_x.append(mean_x)
        all_mean_y.append(mean_y)

    all_mean_x = np.array(all_mean_x)  # (n_participants, n_points)
    all_mean_y = np.array(all_mean_y)

    mean_x_group = all_mean_x.mean(axis=0)
    mean_y_group = all_mean_y.mean(axis=0)
    sem_x_group  = all_mean_x.std(axis=0) / np.sqrt(all_mean_x.shape[0])

    return mean_x_group, sem_x_group, mean_y_group


def plot_group_mean_trajs_3_blocks(dictReorder):
    """
    Trace les trajectoires moyennes de GROUPE (5 participants)
    pour les 3 blocs d'essais, avec ± SEM en X.
    """

    blocks = [
        (0, 15,   "Trials 1-15"),
        (15, 50, "Trials 16-50"),
        (50, 180, "Trials 51-180"),
    ]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    plt.figure(figsize=(6, 6))

    for (start, end, label), col in zip(blocks, colors):
        mean_x_g, sem_x_g, mean_y_g = compute_group_block_mean_traj(dictReorder, start, end)

        plt.plot(mean_x_g, mean_y_g, color=col, linewidth=2, label=label)

        plt.fill_betweenx(mean_y_g,
                          mean_x_g - sem_x_g,
                          mean_x_g + sem_x_g,
                          color=col, alpha=0.25)

    plt.title("Group mean trajectories (aligned, 5 participants)")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.axis("equal")
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.show()
    
participants = ["ev", "fs", "ge", "jc", "jl"]

for p in participants : 
    X = dictReorder[p]["x"]
    Y = dictReorder[p]["y"]

    #plot_mean_trajs_3_blocks(X, Y, participant_label=p)
    plot_peak_adaptation(X, participant_label=p)

group_peaks = get_group_peak_matrix(dictReorder)
plot_group_peak_adaptation(group_peaks)

plot_group_mean_trajs_3_blocks(dictReorder)
