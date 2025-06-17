import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom
import matplotlib.cm as cm
from datetime import datetime
from . import manipulate_dataset as md

# Initial parameters 
default_channels = ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] # configuration for mimic-iv
datetime_format = '%Y-%m-%d-%H_%M_%S'



def quick_plot(
    signal,
    id_label,
    save_path,
    dpi=100,
    channel_names=default_channels,
    fs=500, total_length=10, mV_per_lead=1.5,
    show=False, save=False):
    '''
    Quickly plots a 12-lead ECG without a grid, using minimal formatting.

    Args:
        signal (np.ndarray): ECG signal of shape (n_samples, n_channels), eg. (5000, 12).
        id_label (str): Identifier for the ECG (used in title and filename).
        save_path (str): Directory to save the plot (must end with '/').
        dpi (int): Resolution of the saved figure.
        channel_names (list): List of ECG lead names (default: 12-lead standard).
        fs (int): Sampling frequency in Hz.
        total_length (int): Horizontal length of the figure in inches.
        mV_per_lead (float): Vertical spacing between leads.
        show (bool): Whether to display the plot.
        save (bool): Whether to save the plot to file.

    Returns:
        None
    '''
    
    # Load the signal and sampling frequency
    id_label = str(id_label)
    duration = signal.shape[0] / fs  # Total duration in seconds
    n_leads = signal.shape[1]

    mV_per_s = 2.5  # constant, for mV-sec dim ratio

    # Calculate time axis and scaling factors
    time_axis = np.linspace(0, duration, signal.shape[0])

    length_per_s = total_length / duration
    height_per_mV = length_per_s / mV_per_s
    lead_spacing = mV_per_lead * height_per_mV
    total_height = lead_spacing * n_leads


    # Set up the figure
    fig, ax = plt.subplots(figsize=(total_length, total_height), dpi=dpi)
    ax.set_xlim(0, total_length)
    ax.set_ylim(-total_height, 0)
  
    # Plot each lead
    for i, lead in enumerate(channel_names):
        lead_offset = - ((i + 0.5) * lead_spacing)  # Offset each lead
        ax.plot(time_axis, signal[:, i] * height_per_mV + lead_offset, color='black', linewidth=0.8)
        # Add lead labels
        ax.text(-0.5, lead_offset, lead, fontsize=8, verticalalignment='center', horizontalalignment='right')

    ax.set_title(f"ECG: {id_label}", fontsize=14, weight='bold')
    
    # Save the figure
    current_datetime = datetime.now().strftime(datetime_format)
    plt.tight_layout()
    if save:
        plt.savefig(f"{save_path}ECG_{id_label}_quick_{current_datetime}.png", dpi=dpi)
    if show:
        plt.show()
    plt.close()



def fine_plot(
    signal,
    id_label,
    save_path,
    dpi=100,
    channel_names=default_channels,
    fs=500,
    total_length=10, large_boxes_per_lead=5,              
    show=False, save=True):
    '''
    Plots a 12-lead ECG with standardized grid (small and large boxes), mimicking clinical ECG paper layout.

    Args:
        signal (np.ndarray): ECG signal of shape (n_samples, n_channels), eg. (5000, 12).
        id_label (str): Identifier for the ECG (used in title and filename).
        save_path (str): Directory to save the plot (must end with '/').
        dpi (int): Resolution of the saved figure.
        channel_names (list): List of ECG lead names.
        fs (int): Sampling frequency in Hz.
        total_length (int): Width of the figure in inches.
        large_boxes_per_lead (int): Vertical grid height per lead (in large boxes).
        show (bool): Whether to display the plot.
        save (bool): Whether to save the plot to file.

    Returns:
        None
    '''
    
    # Load the signal and sampling frequency
    id_label = str(id_label)
    duration = signal.shape[0] / fs  # Total duration in seconds
    n_leads = signal.shape[1]

    # Constants for grid
    small_box_ms = 40  # 40ms per small box (5 small boxes = 200ms per large box)
    small_box_mv = 0.1  # 0.1mV per small box (10 small boxes = 1mV)

    # Calculate time axis and scaling factors
    time_axis = np.linspace(0, duration, signal.shape[0])

    total_small_boxes_x = (duration * 1000) / small_box_ms
    length_per_small_box = total_length / total_small_boxes_x
    length_per_large_box = length_per_small_box * 5

    height_per_small_box = length_per_small_box
    height_per_large_box = height_per_small_box * 5
    height_per_mV = height_per_small_box / small_box_mv
    total_small_boxes_y = n_leads * large_boxes_per_lead * 5  # * 5 large boxes/lead * 5 small b/large b
    total_height = total_small_boxes_y * height_per_small_box

    lead_spacing = large_boxes_per_lead * height_per_large_box  # 5 large boxes = 2.5mV spacing per lead

    # Set up the figure
    fig, ax = plt.subplots(figsize=(total_length, total_height), dpi=dpi)
    ax.set_xlim(0, total_length)
    ax.set_ylim(-total_height, 0)

    # Draw grid lines
    for x in np.arange(0, total_length, length_per_small_box):  # Vertical small boxes
        ax.axvline(x=x, color='red', alpha=0.05, linewidth=0.5)
    for x in np.arange(0, total_length, length_per_large_box):  # Vertical large boxes
        ax.axvline(x=x, color='red', alpha=0.2, linewidth=0.8)
    for y in np.arange(-total_height, 0, height_per_small_box):  # Horizontal small boxes
        ax.axhline(y=y, color='red', alpha=0.05, linewidth=0.5)
    for y in np.arange(-total_height, 0, height_per_large_box):  # Horizontal large boxes
        ax.axhline(y=y, color='red', alpha=0.2, linewidth=0.8)
    
    # Plot each lead
    for i, lead in enumerate(channel_names):
        lead_offset = - ((i + 0.5) * lead_spacing)  # Offset each lead
        ax.plot(time_axis, signal[:, i] * height_per_mV + lead_offset, color='black', linewidth=0.8)
        # Add lead labels
        ax.text(-0.5, lead_offset, lead, fontsize=8, verticalalignment='center', horizontalalignment='right')

    # Set axis labels and ticks
    ax.set_xlabel("Time (s)")
    ax.set_xticks(np.arange(0, duration + 1, 1))
    ax.set_yticks([])
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.set_title(f"ECG: {id_label}", fontsize=14, weight='bold')
    
    # Save the figure
    current_datetime = datetime.now().strftime(datetime_format)
    plt.tight_layout()
    if save:
        plt.savefig(f"{save_path}ECG_{id_label}_fine_{current_datetime}.png", dpi=dpi)
    if show:
        plt.show()
    plt.close()



def gradcam_plot(
    signal, gradcam,
    id_label,
    save_path,
    dpi=300,
    channel_names=default_channels,
    fs=500, total_length=10, mV_per_lead=1.5, color_overlay="viridis",
    show=False, save=True):
    '''
    Plots 12-lead ECG with a Grad-CAM activation map overlay as a background heatmap.

    Args:
        signal (np.ndarray): ECG signal of shape (n_samples, n_channels), eg. (5000, 12).
        gradcam (np.ndarray): Grad-CAM activations (reduced length, 1D array).
        id_label (str): Identifier for the ECG.
        save_path (str): Directory to save the plot.
        dpi (int): Resolution of the figure.
        channel_names (list): ECG lead names.
        fs (int): Sampling frequency in Hz.
        total_length (int): Width of the figure in inches.
        mV_per_lead (float): Vertical spacing between leads.
        color_overlay (str): Matplotlib colormap name for the overlay.
        show (bool): Display the plot in notebook or script.
        save (bool): Save the figure to file.

    Returns:
        None
    '''
    
    # Ensure Grad-CAM is resized to match ECG length (5000 time steps)
    gradcam_resized = zoom(gradcam, zoom=(signal.shape[0] / gradcam.shape[0]))

    # Normalize Grad-CAM for colormap scaling (0 to 1)
    gradcam_resized = (gradcam_resized - np.min(gradcam_resized)) / (np.max(gradcam_resized) - np.min(gradcam_resized))
    
    # Set up time axis and scaling factors
    duration = signal.shape[0] / fs  # Total duration in seconds
    n_leads = signal.shape[1]
    time_axis = np.linspace(0, duration, signal.shape[0])
    
    # Set visualization parameters
    mV_per_s = 2.5  # constant, for mV-sec dim ratio
    length_per_s = total_length / duration
    height_per_mV = length_per_s / mV_per_s
    lead_spacing = mV_per_lead * height_per_mV
    total_height = lead_spacing * n_leads

    # Set up figure
    fig, ax = plt.subplots(figsize=(total_length, total_height), dpi=dpi)
    ax.set_xlim(0, total_length)
    ax.set_ylim(-total_height, 0)

    # Get colormap
    cmap = cm.get_cmap(color_overlay)

    # Create a 2D Grad-CAM heatmap (background)
    gradcam_matrix = np.tile(gradcam_resized, (n_leads, 1))  # Repeat across leads

    # Display heatmap as background
    extent = [0, total_length, -total_height, 0]
    ax.imshow(gradcam_matrix, cmap=cmap, aspect='auto', extent=extent, alpha=0.5, interpolation='bilinear')

    # Plot each lead in black
    for i, lead in enumerate(channel_names):
        lead_offset = -((i + 0.5) * lead_spacing)  # Offset each lead
        ax.plot(time_axis, signal[:, i] * height_per_mV + lead_offset, 
                color='black', linewidth=0.8)  # ECG in black

        # Add lead labels
        ax.text(-0.5, lead_offset, lead, fontsize=8, verticalalignment='center', horizontalalignment='right')

    # Add title
    ax.set_title(f"ECG: {id_label} - Grad-CAM Overlay", fontsize=14, weight='bold')
    
    # Save the figure if required
    if save and save_path:
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_path}/ECG_{id_label}_GradCAM-FULL_{current_datetime}.png", dpi=dpi)

    # Show the plot
    if show:
        plt.show()
    
    plt.close()



def gradcam_plot_single(
    signal, gradcam, 
    lead_index,
    id_label,
    save_path,  # Must end with '/'
    dpi=300,
    channel_names=default_channels,
    color_overlay="viridis", fs=500, total_length=10, mV_per_lead=1.5,
    show=False, save=True):
    '''
    Plots a single ECG lead with a Grad-CAM activation overlay.

    Args:
        signal (np.ndarray): ECG signal of shape (n_samples, n_channels), eg. (5000, 12).
        gradcam (np.ndarray): Grad-CAM activations (reduced length, 1D array).
        lead_index (int): Index of the lead to plot (0-11).
        id_label (str): Identifier for the ECG.
        save_path (str): Directory to save the plot.
        dpi (int): Plot resolution in DPI.
        channel_names (list): List of lead names (used for labeling).
        color_overlay (str): Colormap for Grad-CAM.
        fs (int): Sampling frequency.
        total_length (int): Figure width in inches.
        mV_per_lead (float): Vertical scaling for the lead.
        show (bool): Whether to display the plot.
        save (bool): Whether to save the figure.

    Returns:
        None
    '''
    
    # Validate lead_index
    if lead_index < 0 or lead_index >= signal.shape[1]:
        raise ValueError(f"Invalid lead_index {lead_index}. Must be between 0 and {signal.shape[1]-1}.")

    # Ensure Grad-CAM is resized to match ECG length (5000 time steps)
    gradcam_resized = zoom(gradcam, zoom=(signal.shape[0] / gradcam.shape[0]))

    # Normalize Grad-CAM for colormap scaling (0 to 1)
    gradcam_resized = (gradcam_resized - np.min(gradcam_resized)) / (np.max(gradcam_resized) - np.min(gradcam_resized))
    
    # Set up time axis
    duration = signal.shape[0] / fs  # Total duration in seconds
    time_axis = np.linspace(0, duration, signal.shape[0])

    # Get colormap
    cmap = cm.get_cmap(color_overlay)

    # Set up figure
    fig, ax = plt.subplots(figsize=(total_length, 3), dpi=dpi)  # Adjust height for single lead
    ax.set_xlim(0, total_length)
    ax.set_ylim(np.min(signal[:, lead_index]), np.max(signal[:, lead_index]))

    # Display Grad-CAM heatmap as background
    extent = [0, total_length, np.min(signal[:, lead_index]), np.max(signal[:, lead_index])]
    ax.imshow(gradcam_resized[np.newaxis, :], cmap=cmap, aspect="auto", extent=extent, alpha=0.5, interpolation='bilinear')

    # Plot the ECG lead signal in black
    ax.plot(time_axis, signal[:, lead_index], color='black', linewidth=1.0)

    # Add labels and title
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (mV)")
    ax.set_title(f"ECG: {id_label} - Lead {channel_names[lead_index]} - Grad-CAM Overlay")
    ax.legend()

    # Save the figure if required
    if save and save_path:
        current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"{save_path}/ECG_{id_label}_GradCAM-SINGLE_Lead_{channel_names[lead_index]}_{current_datetime}.png", dpi=dpi)

    # Show the plot
    if show:
        plt.show()
    
    plt.close()
