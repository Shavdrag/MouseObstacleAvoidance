import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
import numpy as np

from scapy.all import ARP,Ether,srp

def angle_to_grid_coords(angle, robot_position, robot_orientation, which_whisker):
    """
    Convert whisker angle and robot position to grid coordinates.

    left hind
    0.009000000000000001 -0.04

    whisker durchnummerieren
    orientierung am angfang des roboters
    + rcihtung der schnurrhaare einbeziwhen


    links -> x positiv

    momentan alle 45 grad weil axis = -1 -1
    ODER DOCH NICHT? DARAN DENKEN DASS IN ANDERE RICHTUNG SCHAUT
    """


    #[" right front 12", " right hind 12", " left hind 12"," left front 12"]
    if which_whisker > 1:
        # rechts, 0-1
        a = 1
    else:
        #links 2-3
        a = -1
    if which_whisker==0 or which_whisker==2:
        #0,2 vorne
        root_angle = 30
    else:
        root_angle = 60

    WHISKER_LENGTH = 0.05 # eig 0.05

    dx = WHISKER_LENGTH * math.sin(math.radians(root_angle * a) + angle + robot_orientation)

    dy = WHISKER_LENGTH * math.cos(math.radians(root_angle) + angle + robot_orientation)
    x, y = robot_position[0] + dx, robot_position[1] + dy

    return (x, y)

def get_whisker_contact_points(whisker_angles, robot_position, robot_orientation):
    """
    Get grid coordinates for whisker contact points.
    """
    contact_points = []
    for count, angle in enumerate(whisker_angles):

        if abs(angle) > 0.03:  # THINK ABOUT THRESHOLD
            point = angle_to_grid_coords(angle, robot_position, robot_orientation, which_whisker=count)
            contact_points.append(point)

    return contact_points


def pad_sequences(sequences, max_len, padding_value=np.nan):
    """Pad sequences to the specified max length."""
    padded_seqs = []
    for seq in sequences:
        if seq.ndim == 1:  # If sequence is effectively 1D
            padded_seq = np.pad(seq, (0, max_len - len(seq)), constant_values=padding_value)
        else:
            padded_seq = np.pad(seq, ((0, max_len - len(seq)), (0, 0)), constant_values=padding_value)
        padded_seqs.append(padded_seq)
    return np.array(padded_seqs)

def plot_results(angles, reward_l, action_l):
    fig, ax1 = plt.subplots(3)

    # Plot angles
    ax1[0].plot(angles, label='Angles')
    ax1[0].set_xlabel('Time step')
    ax1[0].set_ylabel('Angles')

    # Create second y-axis that share same x-axis
    ax1[1].plot(reward_l, label='Reward', color='r')
    ax1[1].set_ylabel('Reward')

    ax1[2].plot(action_l, label='action', color='r')
    ax1[2].set_ylabel('Action')

    plt.show()
    return None



def plot_action_hist(data):

    data_np = np.concatenate(data)

    # Now you can safely use min and max methods
    plt.hist(data_np, bins=np.arange(data_np.min(), data_np.max() + 2) - 0.5, edgecolor='black', density=True)

    # Customizations
    plt.title('Distribution of Actions')
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(data_np.min(), data_np.max() + 1))

    # Show plot
    plt.show()


def plot_angle_reward(angles, reward_l):

    fig, axs = plt.subplots(2, figsize=(10, 12))
    axs[0].plot(reward_l, label='Path')
    axs[0].legend()
    axs[0].grid(True)

    for i in range(angles.shape[1]):
        axs[1].plot(angles[:, i], label=f'Sub-Array {i + 1}')

    # Anpassen des Layouts für bessere Darstellung
    plt.tight_layout()

    plt.show()

def plot_collision_map(contact_points, all_pos, angles, WORKERS):
    contact_points = np.array(contact_points)

    if WORKERS == 1:

        fig, axs = plt.subplots(2, figsize=(10, 12))

        # Plot the path and contact points
        x = all_pos[1:-1, 0]
        y = all_pos[1:-1, 1]

        if 1:
            if contact_points.any():
                print("contact existing")
                try:
                    x_coords = contact_points[:, 0]
                    y_coords = contact_points[:, 1]
                except:
                    x_coords = contact_points[0]
                    y_coords = contact_points[1]

            axs[0].plot(x, y, label='Path')
            labels =[
                "Right Front Whisker",
                "Right Hind Whisker",
                "Left Hind Whisker",
                "Left Front Whisker"
            ]
            if contact_points.any(): axs[0].scatter(x_coords, y_coords, color='red', label='Contact Points')
            axs[0].set_title("Path and Contact Points Visualization")
            axs[0].set_xlabel("X Coordinate")
            axs[0].set_ylabel("Y Coordinate")
            axs[0].axis('square')
            axs[0].legend()
            axs[0].grid(True)

            # Plot angles
            axs[1].plot(np.squeeze(angles), label=labels)
            axs[1].set_xlabel('Time step')
            axs[1].set_ylabel('Angles')
            axs[1].legend()

            x_lim = np.max(np.abs(x))
            y_lim = np.max(np.abs(y))
            lim = max(x_lim, y_lim)

            print("x:  ",x_lim, "   y:  ",y_lim)
            axs[0].set_xlim(-lim, lim)
            axs[0].set_ylim(-lim, lim)
            axs[0].set_aspect('equal', adjustable='box')

            plt.tight_layout()
            plt.show()
        else:
            print("only one worker allowed to plot")

def find_devices_in_network(network_cidr):
    """
    Scan for devices in a network using ARP.

    :param network_cidr: The network CIDR (e.g., '192.168.1.1/24')
    :return: List of tuples (IP, MAC) of discovered devices or an empty list if no devices found.
    """
    # Create ARP request packet for the network
    arp_request = ARP(pdst=network_cidr)
    broadcast = Ether(dst="ff:ff:ff:ff:ff:ff")
    arp_request_broadcast = broadcast / arp_request

    # Send the packet and receive responses
    answered_list = srp(arp_request_broadcast, timeout=1, verbose=False)[0]

    devices = []

    # Parse the response
    for sent, received in answered_list:
        devices.append((received.psrc, received.hwsrc))

    return devices
