import matplotlib.pyplot as plt
import numpy as np


def plot_1D(optim, player, f_handle, params, rm):
    if rm != None:
        for t in rm:
            t[0].remove()

    ax = f_handle.axes[0]
    rm = []
    print(optim.getx())
    print(optim.getu())
    Hm = params["optimizer"]["Hm"]+1
    y = (-2.0*np.ones_like(optim.getx()[:Hm])).tolist() + [-2.025*np.ones_like(
        optim.getx()[Hm+1])] + (-2.05*np.ones_like(optim.getx()[Hm+1:])).tolist()
    rm.append(ax.plot(player.planned_measure_loc[0],
                      player.planned_measure_loc[1], "*", mew=2, color='tab:green', label='planned'))
    rm.append(ax.plot(player.safe_meas_loc[0][0],
                      player.safe_meas_loc[0][1], "*", mew=2, color='tab:cyan', label='safe'))
    rm.append(ax.plot(optim.getz(), -1.95, "*", mew=2, color='k', label='z'))
    rm.append(ax.plot(player.current_location[0][0],
                      player.current_location[0][1], "*", mew=5, color='tab:blue', label='start'))
    rm.append(ax.plot(player.origin[0],
                      player.origin[1] - 0.05, "*", mew=5, color='tab:blue', label='end'))
    rm.append(ax.plot(optim.getx(), y, color='tab:olive'))
    rm.append(ax.plot(optim.getx(), y,
                      "*", color='tab:brown', label='trajectory'))

    ax.set_xlim([-2.0, 1])
    ax.axis("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Agent 1D trajectory")
    ax.legend(loc='upper left', ncol=3)
    ax.grid()

    ax = f_handle.axes[1]
    rm.append(ax.plot(optim.getx()))
    ax.set_title("Position vs time")
    ax.set_xlabel("time")
    ax.set_ylabel("x")

    ax = f_handle.axes[2]
    rm.append(ax.plot(player.planned_measure_loc[0],
                      player.planned_measure_loc[1], "*", mew=5, color='tab:green', label='planned'))
    rm.append(ax.plot(player.current_location[0][0],
                      player.current_location[0][1], "*", mew=5, color='tab:blue', label='start'))
    rm.append(ax.plot(player.origin[0],
                      player.origin[1]-0.05, "*", mew=5, color='tab:blue', label='end'))
    rm.append(ax.plot(player.obj_optim.getx(), y, color='tab:olive'))
    rm.append(ax.plot(player.obj_optim.getx(), y,
                      "*", color='tab:brown', label='trajectory'))
    ax.set_xlim([-2.0, 1])
    ax.axis("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Objective trajectory")
    ax.legend(loc='upper left', ncol=3)
    ax.grid()

    ax = f_handle.axes[3]
    rm.append(ax.plot(optim.getu()))
    ax.set_title("Control vs time")
    ax.set_xlabel("time")
    ax.set_ylabel("u")
    f_handle.savefig("temp1D.png")
    print("z", optim.getz())
    return rm


def plot_2D(optim, player):

    print(optim.getx())
    print(optim.getu())
    plt.subplot(2, 2, 1)
    plt.plot(player.planned_measure_loc[0],
             player.planned_measure_loc[1], "*", color='tab:green')
    plt.plot(player.current_location[0],
             player.current_location[1], "*", color='tab:blue')
    plt.plot(optim.getz()[0],
             optim.getz()[1], "*", color='tab:orange')
    plt.plot(optim.getx()[0], optim.getx()[1])
    plt.plot(optim.getx()[0], optim.getx()[1], "*", color='k')
    plt.xlim([-2.0, 1])
    plt.axis("equal")
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(optim.getu()[0], optim.getu()[1])

    plt.subplot(2, 2, 3)
    plt.plot(optim.getu()[0])

    plt.subplot(2, 2, 4)
    plt.plot(optim.getu()[1])
    plt.savefig("temp.png")
