import matplotlib.pyplot as plt
import numpy as np

def plot_fixations(axs, scanpath, coordinate, color):
    # Sample data (time, x position, duration)
    time = scanpath['cum_fixation_duration'].to_list()
    if coordinate == "y":
        position = scanpath['fixation_position_y'].to_list()
    else:
        position = scanpath['fixation_position_x'].to_list()
    duration = scanpath['fixation_duration'].to_list()

    # Plot the horizontal lines
    list1 = []
    list2 = []
    for i in range(len(time)):
        list1.extend([position[i], position[i]])
        list2.extend([time[i], time[i] + duration[i]])
    axs.plot(list1, list2, color=color[0])

    for i in range(len(time)):
        axs.vlines(x=position[i], ymin=time[i], ymax=time[i] + duration[i], color=color[1])    

    # Add labels and title
    axs.set_ylabel('Time')
    if coordinate == "x":
        axs.set_xlabel('X Position')
    else:
        axs.set_xlabel('Y Position')
    return axs


def plot_scanpaths(axs, scanpathlist):
    color_comb = [('blue', 'red'), ('green', 'red'), ('purple', 'red') ]
    axs[0, 0] = plot_fixations(axs[0, 0], scanpathlist[0], "x", color_comb[0])
    axs[0, 1] = plot_fixations(axs[0, 1], scanpathlist[1], "x", color_comb[1])
    axs[1, 0] = plot_fixations(axs[1, 0], scanpathlist[0], "y", color_comb[0])
    axs[1, 1] = plot_fixations(axs[1, 1], scanpathlist[1], "y", color_comb[1])
    return axs



def plot_alignments_component(axs, scanpath, scanno, coordinate, color):

    # time = scanpath['cum_fixation_duration' + "_" + scanno].to_list()
    # duration = scanpath['fixation_duration'+ "_" + scanno].to_list()

    time = list(range(scanpath.shape[0]))
    duration = np.ones(shape= scanpath.shape[0]).tolist()
    if coordinate == "y":
        position = scanpath['fixation_position_y'+ "_" + scanno].to_list()
    else:
        position = scanpath['fixation_position_x'+ "_" + scanno].to_list()

    # Plot the horizontal lines
    list1 = []
    list2 = []
    for i in range(len(time)):
        list1.extend([position[i], position[i]])
        list2.extend([time[i], time[i] + duration[i]])
    axs.plot(list1, list2, color=color[0])

    for i in range(len(time)):
        axs.vlines(x=position[i], ymin=time[i], ymax=time[i] + duration[i], color=color[1])    

    # Add labels and title
    axs.set_ylabel('Steps')
    if coordinate == "x":
        axs.set_xlabel('X Position')
    else:
        axs.set_xlabel('Y Position') 

    return axs

def plot_alignments(axs, alignment):
    color_comb = [('blue', 'red'), ('green', 'red'), ('purple', 'red') ]
    axs[0, 2] = plot_alignments_component(axs[0, 2], alignment, "s", 'x', color_comb[0])
    axs[0, 3] = plot_alignments_component(axs[0, 3], alignment, "t", 'x', color_comb[1])
    axs[0, 4] = plot_alignments_component(axs[0, 4], alignment, "s", 'x', color_comb[0])
    axs[0, 4] = plot_alignments_component(axs[0, 4], alignment, "t", 'x', color_comb[1])

    axs[1, 2] = plot_alignments_component(axs[1, 2], alignment, "s", 'y', color_comb[0])
    axs[1, 3] = plot_alignments_component(axs[1, 3], alignment, "t", 'y', color_comb[1])
    axs[1, 4] = plot_alignments_component(axs[1, 4], alignment, "s", 'y', color_comb[0])
    axs[1, 4] = plot_alignments_component(axs[1, 4], alignment, "t", 'y', color_comb[1])
    return axs

