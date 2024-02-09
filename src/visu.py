import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import dill as pickle
from src.utils.helper import get_frame_writer

matplotlib.use('Agg')  # for debug mode to plot and save fig


class Visu():
    def __init__(self, grid_V, safe_boundary, true_constraint_function, true_objective_func, opt_goal, optimal_feasible_boundary, params, path) -> None:
        self.grid_V = grid_V
        self.params = params
        self.common_params = params["common"]
        self.agent_param = params["agent"]
        self.visu_params = params["visu"]
        self.env_params = params["env"]
        self.optim_params = params["optimizer"]
        self.Nx = self.env_params["shape"]["x"]
        self.Ny = self.env_params["shape"]["y"]
        self.step_size = self.visu_params["step_size"]
        self.num_players = self.env_params["n_players"]
        self.safe_boundary = safe_boundary
        # safety_function(test_x.numpy().reshape(-1, 1))
        self.true_constraint_function = true_constraint_function
        self.true_objective_func = true_objective_func
        self.opt_goal = opt_goal
        self.Cx_beta = self.agent_param["Cx_beta"]
        self.Fx_beta = self.agent_param["Fx_beta"]
        self.mean_shift_val = self.agent_param["mean_shift_val"]
        self.agent_current_loc = {}
        self.agent_current_goal = {}  # Dict of all the agent current goal
        self.discs_nodes = {}
        self.optimal_feasible_boundary = optimal_feasible_boundary
        self.prev_w = torch.zeros([101])
        self.x = self.grid_V.transpose(0, 1).reshape(-1, self.Nx, self.Ny)[0]
        self.y = self.grid_V.transpose(0, 1).reshape(-1, self.Nx, self.Ny)[1]
        self.tr_constraint = self.true_constraint_function.reshape(
            self.Nx, self.Ny)
        self.tr_density = self.true_objective_func.reshape(self.Nx, self.Ny)
        self.x_dim = params["optimizer"]["x_dim"]
        if self.visu_params["show"]:
            self.initialize_plot_handles(path)
            if self.Ny != 1:
                self.plot_contour_env()
        self.plot_unsafe_debug = False
        self.temp_objects = {}
        self.temp_objects["Cx"] = None
        self.temp_objects["Fx"] = None
        self.temp_objects["Dyn"] = None
        self.traj = None
        self.state_traj = []
        self.input_traj = []
        self.meas_traj = []
        self.player_train_pts = []
        self.player_model= []
        self.opti_path_list = []
        self.utility_minimizer_list = []
        self.num_safe_nodes_list = []
        self.save_path = path
        self.iteration_time = []
        # X = self.grid_V
        # lb = self.Cx_model(X.float()).mean - self.Cx_beta*2*torch.sqrt(self.Cx_model(X.float()).variance)
        # Lc = np.diff(lb.transpose()).max()/(0.025)

    def initialize_plot_handles(self, path):
        fig_gp, ax = plt.subplots(figsize=(16/2.4, 14/2.4))
        fig_gp.tight_layout(pad=0)
        # ax.grid(which='both', axis='both')
        # ax.minorticks_on()
        # ax2.grid(which='both', axis='both')
        # ax2.minorticks_on()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_aspect('equal', 'box')
        ax.set_xlim(self.env_params["start"],self.env_params["start"] + self.visu_params["step_size"]*self.env_params["shape"]["x"])
        ax.set_ylim(self.env_params["start"],self.env_params["start"] + self.visu_params["step_size"]*self.env_params["shape"]["y"])
        fig_dyn, ax2 = plt.subplots() # plt.subplots(2,2)

        # ax2.set_aspect('equal', 'box')
        self.f_handle = {}
        self.f_handle["gp"] = fig_gp
        self.f_handle["dyn"] = fig_dyn
        self.plot_contour_env("dyn")


        # Move it to visu
        self.writer_gp = get_frame_writer()
        self.writer_dyn = get_frame_writer()
        self.writer_dyn.setup(fig_dyn, path +
                              "/video_dyn.mp4", dpi=200)
        self.writer_gp.setup(fig_gp, path +
                             "/video_gp.mp4", dpi=200)

    def fmt(self, x):
        s = f"{x:.1f}"
        if s.endswith("0"):
            s = f"{x:.0f}"
        return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"

    def plot_contour_env(self, f="gp", rm=None):
        ax = self.f_handle[f].axes[0]
        # CS = ax.contour(self.x.numpy(), self.y.numpy(),
        #                 self.tr_constraint.numpy(), np.array([self.common_params["constraint"], self.common_params["constraint"] + self.common_params["epsilon"]]))
        CS = ax.contour(self.x.numpy(), self.y.numpy(), 
                       self.tr_constraint.numpy(), np.array([self.common_params["constraint"] - 0.01*k for k in range(200,2,-1)]), colors=["black"], linewidths=14,linestyles='solid')
        Cs_q_eps=ax.contour(self.x.numpy(), self.y.numpy(),  
                       self.tr_constraint.numpy(), np.array([self.common_params["constraint"] + 0.2]), colors=["black"], linewidths=2,linestyles='--')
        ax.clabel(Cs_q_eps, Cs_q_eps.levels, fmt=r'$q(x)=\epsilon$',inline=True, fontsize=12)
        # rm.append([CS])
        CS2 = ax.contourf(self.x.numpy(), self.y.numpy(),
                          self.tr_density.numpy(),np.array([0.02*k for k in range(150)]), alpha=0.4, antialiased=True)
        for c in CS2.collections:
            # c.set_edgecolor('face')
            c.set_linewidth(0.00)
        # rm.append([CS2])
        # ax.clabel(CS, CS.levels, inline=True,
        #           fmt=self.fmt, fontsize=10)

        CS2.cmap.set_over('red')
        CS2.cmap.set_under('blue')
        # self.f_handle[f].colorbar(CS2)
        # rm.append([self.f_handle.colorbar(CS2)])
        return ax, rm

    def plot_optimal_point(self):
        ax = self.plot_contour_env()
        for key in self.optimal_feasible_boundary:
            loc = self.grid_V[self.optimal_feasible_boundary[key]]
            ax.plot(loc[:, 0], loc[:, 1], ".",
                    color="mediumpurple", mew=2)

            # ax.plot(loc[0], loc[1], color="mediumpurple")
        for agent_idx, init_pt in enumerate(self.safe_boundary):
            ax.text(init_pt[0], init_pt[1], str(
                agent_idx), color="cyan", fontsize=12)
            ax.plot(init_pt[0], init_pt[1], "*", color="cyan", mew=2)

        # for loc in self.opt_goal:
        # for i in range(self.opt_goal["Fx_X"].shape[0]):
        #     loc = self.opt_goal["Fx_X"][i]
        #     ax.text(loc[0], loc[1], str(
        #         i), color="tab:green", fontsize=12)
        ax.plot(self.opt_goal["Fx_X"][:, 0],
                self.opt_goal["Fx_X"][:, 1], "*", color="tab:green",  mew=3)

        # ax.clabel(CS2, CS2.levels, inline=True, fmt=fmt, fontsize=10)
        ax.axis("equal")
        # plt.show()
        # plt.savefig("check.png")

    def CxVisuUpdate(self, Cx_model, current_loc,  Cx_data, agent_key):
        self.Cx_model = Cx_model
        self.Cx_data = Cx_data
        self.Cx_agent_key = agent_key
        if not (self.Cx_agent_key in self.agent_current_loc):
            self.agent_current_loc[self.Cx_agent_key] = []
        self.agent_current_loc[self.Cx_agent_key].append(current_loc)

    def FxUpdate(self, Fx_model, current_goal, acq_density, Fx_data, agent_key):
        self.Fx_model = Fx_model
        self.Fx_agent_key = agent_key
        self.Fx_data = Fx_data
        if not (self.Fx_agent_key in self.agent_current_goal):
            self.agent_current_goal[self.Fx_agent_key] = []
        self.agent_current_goal[self.Fx_agent_key].append(current_goal)
        self.acq_density = acq_density
        if not (self.Fx_agent_key in self.discs_nodes):
            self.discs_nodes[self.Fx_agent_key] = []

    def plot_Fx(self, f_handle):
        ax = f_handle.axes[0]
        rm = []
        for agent_key in self.agent_current_goal:
            data = self.agent_current_goal[agent_key][-1]
            # print("Visu check", data[0])
            # rm.append([ax.text(data["Fx_X"][0], data["Fx_X"][1], str(
            #     agent_key), color="gold", fontsize=12)])
            # Plot the currently pursuing goal along with goal of 3 agent with text
        rm.append(ax.plot(self.agent_current_goal[self.Fx_agent_key][-1]["Fx_X"][0], self.agent_current_goal[self.Fx_agent_key][-1]["Fx_X"][1],
                          "*", color="gold", mew=1.5))

        # for agent_key in self.discs_nodes:
        #     single_disc_connections = self.discs_nodes[agent_key]
        #     for edges in single_disc_connections:
        #         loc = self.grid_V[edges].reshape(-1, 2)
        #         rm.append(ax.plot(loc[:, 0], loc[:, 1],
        #                   color="tab:brown"))
                # for loc in self.opt_goal:
        # ax.savefig("check2.png")
        # ax.close()
        ax.axis("equal")
        if self.use_goose:
            ax.set_title("Iteration " + str(self.n_iter) + " Goose Iter " + str(self.goose_step) +
                         " Agent " + str(self.Fx_agent_key))
        else:
            ax.set_title("Iteration " + str(self.n_iter) +
                         " Agent " + str(self.Fx_agent_key))
        return rm

    def plot1Dobj_GP(self, f_handle):
        ax = f_handle.axes[0]
        x = self.grid_V
        # observed_pred = self.Cx_model.likelihood(self.Cx_model(test_x))
        # posterior is only avaialble with botorch and not in gpytorch
        observed_posterior = self.Fx_model.posterior(x)
        # ax.plot(observed_pred.mean.detach().numpy())
        lower, upper = observed_posterior.mvn.confidence_region()
        lower = lower + self.mean_shift_val
        upper = upper + self.mean_shift_val
        temp = lower*(1+self.Fx_beta)/2 + upper*(1-self.Fx_beta)/2
        upper = upper*(1+self.Fx_beta)/2 + lower*(1-self.Fx_beta)/2
        lower = temp
        rm = []
        rm.append([ax.fill_between(x[:, 0].numpy(), lower.detach().numpy(),
                                   upper.detach().numpy(), alpha=0.5, color="tab:purple")])
        rm.append(ax.plot(x[:, 0].numpy(),
                          observed_posterior.mean.detach().numpy() + self.mean_shift_val, color="tab:purple", label="Fx-mean"))
        ax.plot(x[:, 0].numpy(), self.true_objective_func, color="tab:orange")
        # rm.append(ax.plot(x[:, 0].reshape(-1, 1)[self.S_opti[self.Fx_agent_key].StateInSet], self.acq_density.detach().reshape(-1, 1)[self.S_opti[self.Fx_agent_key].StateInSet],
        #                    color="tab:pink"))

        ax.plot(self.Fx_data["Fx_X"][:, 0].numpy(),
                self.Fx_data["Fx_Y"].numpy(), "*", color="red", mew=2)
        # ax.plot(self.opt_goal["Fx_X"][:, 0].numpy(),
        #         self.opt_goal["Fx_Y"].numpy(), "*", color="tab:green",  mew=3)

        y_loc = -0.5
        fact = 0.1
        # for agent_key in self.discs_nodes:
        #     single_disc_nodes = self.discs_nodes[agent_key]
        #     left = self.grid_V[np.min(single_disc_nodes)][0]
        #     right = self.grid_V[np.max(single_disc_nodes)][0]
        #     rm.append(ax.plot([left, right], [y_loc-fact*agent_key, y_loc-fact*agent_key],
        #               color="tab:brown", linewidth=6.0))

        for agent_key in self.agent_current_goal:
            data = self.agent_current_goal[agent_key][-1]  # last data point
            # print("Visu check", data[0])
            # rm.append([ax.text(data["Fx_X"][0], data["Fx_Y"].view(-1), str(
            #     agent_key), color="gold", fontsize=12)])
        # Plot the currently pursuing goal along with goal of 3 agent with text
        rm.append(ax.plot(self.agent_current_goal[self.Fx_agent_key][-1]["Fx_X"][0], self.agent_current_goal[self.Fx_agent_key][-1]["Fx_Y"].view(-1),
                          "*", color="gold", mew=1.5))
        ax.legend(loc='upper left')
        # rm.append(ax.plot(x[:, 0].numpy(),
        #           self.M_dist[0]/1e6*2 + 4, color="brown"))
        # rm.append(ax.plot(x[:, 0].numpy(),
        #           self.M_dist[1]/1e6*2 + 1, color="brown"))
        ax.set_title("Iteration " + str(self.n_iter) + ", Fx NLP")
        return rm

    def UpdateIter(self, iter, goose_step):
        self.n_iter = iter + 1
        self.goose_step = goose_step

    def plot_safe_GP(self, f_handle):
        rm = []
        # _, _, rm = self.plot_contour_env(f_handle, rm)
        ax = f_handle.axes[0]
        ax = self.f_handle["gp"].axes[0]
        self.Cx_model.eval()
        X = self.grid_V
        lb = self.Cx_model(X.float()).mean - self.Cx_beta*2*torch.sqrt(self.Cx_model(X.float()).variance).detach()
        self.lb_grid_numpy = lb.reshape(self.Nx, self.Ny).detach().numpy()
        # lb = self.tr_constraint.numpy()
        if self.params["algo"]["type"] == "ret_expander" or self.params["algo"]["type"] == "MPC_expander":
            # # pick a neighbouring tile that it can guarentee to be safe
            # self.safe = (self.lb_grid_numpy>=0)*self.lb_grid_numpy
            # self.tiles = np.floor((((self.lb_grid_numpy>=0)*self.lb_grid_numpy/self.common_params["Lc"]))/self.visu_params["step_size"])
            # self.safe_loc_x = (self.tiles>=0)*self.x.numpy()
            # self.safe_loc_y = (self.tiles>=0)*self.y.numpy()
            # loc_i, loc_j = np.where(self.tiles>=1)
            # self.lb_grid_numpy_Lc = lb.reshape(self.Nx, self.Ny).detach().numpy().copy()
            # for i,j in zip(loc_i, loc_j):
            #     size = int(self.tiles[i,j])
            #     self.lb_grid_numpy_Lc[i-size:i+size,j-size:j+size] = np.ones_like(self.lb_grid_numpy_Lc[i-size:i+size,j-size:j+size])
            dist_matrix = torch.cdist(self.grid_V,self.grid_V,p=2)
            V_lower_Cx_mat = torch.vstack([lb]*lb.shape[0])
            self.lb_grid_numpy_Lc = torch.max(V_lower_Cx_mat - self.common_params["Lc"]*dist_matrix,1)[0].detach().numpy().reshape(self.Nx, self.Ny)
            CS2 = ax.contour(self.x.numpy(), self.y.numpy(),
                            self.lb_grid_numpy_Lc, np.array([self.common_params["constraint"], self.common_params["constraint"] + 0.1]), alpha=0.5)
            # rm.append([CS2.collections[0]])
            # rm.append([CS2.collections[1]])
        CS = ax.contour(self.x.numpy(), self.y.numpy(),
                        self.lb_grid_numpy, np.array([self.common_params["constraint"], self.common_params["constraint"] + 0.1]))
        rm.append([CS.collections[0]])
        rm.append([CS.collections[1]])
        
        if self.params["visu"]["show_opti_set"]:
            ub = self.Cx_model(X.float()).mean + self.Cx_beta*2*torch.sqrt(self.Cx_model(X.float()).variance)
            self.ub_grid_numpy = ub.reshape(self.Nx, self.Ny).detach().numpy()
            # ub = self.tr_constraint.numpy()
            # if np.min(self.ub_grid_numpy)<0.3:
            #     stop =1
            CS = ax.contour(self.x.numpy(), self.y.numpy(),
                            self.ub_grid_numpy, np.array([self.common_params["constraint"]+self.common_params["epsilon"], self.common_params["constraint"] + self.common_params["epsilon"] + 0.2]), colors=['tab:orange','tab:green'])
            for it in CS.collections:
                rm.append([it])
        # rm.append([CS.collections[1]])
        # for key in self.optimal_feasible_boundary:
        #     loc = self.grid_V[self.optimal_feasible_boundary[key]]
        #     ax.plot(loc[:, 0], loc[:, 1], ".", color="mediumpurple", mew=2)

        # for key in self.opti_boundary:
        #     loc = self.grid_V[self.opti_boundary[key]]
        #     rm.append(ax.plot(loc[:, 0], loc[:, 1], ".", color="gold", mew=2))

        # for key in self.pessi_boundary:
        #     loc = self.grid_V[self.pessi_boundary[key]]
        #     rm.append(ax.plot(loc[:, 0], loc[:, 1], ".", color="black", mew=2))

        # for init_pt in self.safe_boundary:
        #     ax.plot(init_pt[0], init_pt[1], "*", color="cyan", mew=2)

        # for loc in self.opt_goal:
        # rm.append(ax.plot(self.opt_goal["Fx_X"][:, 0],
        #                   self.opt_goal["Fx_X"][:, 1], "*", color="tab:green",  mew=3))
        if self.params["visu"]["show_path"]:
            if self.traj is not None:
                ax.plot(self.traj[:self.optim_params["Hm"]+1, 0], self.traj[:self.optim_params["Hm"]+1, 1], color="tab:blue")
                rm.append(ax.plot(self.traj[self.optim_params["Hm"]:, 0], self.traj[self.optim_params["Hm"]:, 1], color="tab:red"))
        if self.visu_params["show_current_loc"]:
            for key in self.agent_current_loc:
                data = self.agent_current_loc[key][-1].reshape(-1)
                rm.append(
                    [ax.text(data[0] - 2*self.step_size, data[1]-2*self.step_size, str(key), color="tab:brown", fontsize=12, weight='bold')])
                # rm.append(ax.plot(data[0], data[1],
                #                   ".", color="tab:brown",  mew=100))
                rm.append(ax.plot(data[0], data[1],
                                ".", color="tab:brown",  mew=10))
        if self.plot_unsafe_debug:
            all_unsafe_loc = self.grid_V[self.all_unsafe_nodes]
            ax.plot(all_unsafe_loc[:, 0],
                    all_unsafe_loc[:, 1], 'x', color="red", mew=1)
            unreachable_nodes = self.grid_V[self.unreachable_nodes]
            ax.plot(unreachable_nodes[:, 0],
                    unreachable_nodes[:, 1], 'x', color="black", mew=1)

            for edge in list(self.unsafe_edges_set):
                st = self.grid_V[edge[0]]
                ed = self.grid_V[edge[1]]
                ax.plot([st[0], ed[0]], [st[1], ed[1]])
        # rm.append(ax.plot(self.Cx_data["Cx_X"][-self.num_players:, 0], self.Cx_data["Cx_X"]
        #                   [-self.num_players:, 1], ".", color="tab:brown",  mew=100))
        # rm.append(ax.plot(self.Cx_data["Cx_X"][-self.num_players:, 0], self.Cx_data["Cx_X"]
        #                   [-self.num_players:, 1], ".", color="tab:brown",  mew=10))
        ax.plot(self.env_params["start_loc"][0],self.env_params["start_loc"][1], "*", color="tab:red",  ms=15)
        ax.plot(self.Cx_data["Cx_X"][:, 0], self.Cx_data["Cx_X"]
                [:, 1], ".", color="red",  mew=1, ms=6, alpha=0.6)
        if self.params["algo"]["objective"] == "GO":
            rm.append(ax.plot(self.utility_minimizer[0],self.utility_minimizer[1], "*", ms=15,color="tab:green"))
        if self.visu_params["opti_path"]:
            rm.append(ax.plot(self.opti_path[:,0],self.opti_path[:,1], color='k', alpha=0.2))

        # ax.clabel(CS2, CS2.levels, inline=True, fmt=fmt, fontsize=10)
        # ax.axis("equal", 'box')
        ax.set_title("Iteration " + str(self.n_iter))
        ax.set_aspect('equal', 'box')
        ax.set_xlim(self.env_params["start"]+0.2,self.env_params["start"] + self.visu_params["step_size"]*(self.env_params["shape"]["x"]-1))
        ax.set_ylim(self.env_params["start"],self.env_params["start"] + self.visu_params["step_size"]*(self.env_params["shape"]["y"]-1))
        
        # plt.show()
        # plt.savefig("check1.png")
        return rm

    def plot1Dsafe_GP(self, f_handle):
        ax = f_handle.axes[0]
        x = self.grid_V
        # observed_pred = self.Cx_model.likelihood(self.Cx_model(test_x))
        # posterior is only avaialble with botorch and not in gpytorch
        observed_posterior = self.Cx_model.posterior(x)
        # plt.plot(observed_pred.mean.detach().numpy())
        lower, upper = observed_posterior.mvn.confidence_region()
        temp1 = lower*(1+self.Cx_beta)/2 + upper*(1-self.Cx_beta)/2
        upper = upper*(1+self.Cx_beta)/2 + lower*(1-self.Cx_beta)/2
        lower = temp1
        w = upper-lower
        # print(self.prev_w - w)
        self.prev_w = upper - lower
        ax.plot(x[:, 0].numpy(), self.true_constraint_function.numpy(),
                color="tab:orange")
        ax.axhline(
            y=self.common_params["constraint"], color='k', linestyle='--')
        rm = []
        rm.append([ax.fill_between(x[:, 0].numpy(), lower.detach().numpy(),
                                   upper.detach().numpy(), alpha=0.5, color="tab:blue")])
        rm.append(ax.plot(x[:, 0].numpy(), upper.detach().numpy() -
                  self.common_params["epsilon"], "--", color="tab:blue"))
        rm.append(ax.plot(x[:, 0].numpy(),
                          observed_posterior.mean.detach().numpy(), color="tab:blue", label="Cx-mean"))

        # for lines in self.lines:
        #     rm.append(ax.plot(lines["opti"]["left"]["X"], lines["opti"]
        #                        ["left"]["Y"], color="tab:olive"))
        #     rm.append(ax.plot(lines["opti"]["right"]["X"], lines["opti"]
        #                        ["right"]["Y"], color="tab:olive"))
        #     rm.append(ax.plot(lines["pessi"]["left"]["X"], lines["pessi"]
        #                        ["left"]["Y"], color="tab:pink"))
        #     rm.append(ax.plot(lines["pessi"]["right"]["X"], lines["pessi"]
        #                        ["right"]["Y"], color="tab:pink"))
        # n_agents = len(self.lines)
        if self.Cx_data["info_pt_z"] is not None:
            rm.append(ax.plot(self.Cx_data["info_pt_z"], self.common_params["constraint"],
                              "*", color="black", mew=2))
        rm.append(ax.plot(self.Cx_data["plan_meas"][0], self.common_params["constraint"],
                          "*", color="tab:purple", mew=2))
        rm.append(ax.plot(self.Cx_data["safe_meas"][0][0], self.common_params["constraint"],
                          "*", color="tab:blue", mew=2))
        ax.plot(self.Cx_data["Cx_X"][:-1][:, 0].numpy(), self.Cx_data["Cx_Y"]
                [:-1].numpy(), "*", color="red", mew=2)
        ax.plot(self.Cx_data["Cx_X"][-1:][:, 0].numpy(), self.Cx_data["Cx_Y"]
                [-1:].numpy(), "*", color="yellow", mew=2)

        # for data in [torch.Tensor([0.7200, -2.0]), torch.Tensor([0.6700, -2.0])]:
        #     #    data = self.agent_current_loc[key][-1]
        #     y_mean = self.Cx_model.posterior(
        #         data.reshape(-1, 2)).mvn.mean.detach().numpy()
        #     y_ub = self.Cx_model.posterior(
        #         data.reshape(-1, 2)).mvn.variance.detach().numpy()
        #     y_ub = y_mean + 3*np.sqrt(y_ub)
        #     dy = (y_ub - self.common_params["constraint"])
        #     dx = dy/self.agent_param["Lc"]
        #     lx = [data[0].item(), data[0].item() - dx[0]]
        #     ly = [y_ub[0], self.common_params["constraint"]]
        #     plt.plot(lx, ly, label='lc', color='tab:olive')

        for key in self.agent_current_loc:
            data = self.agent_current_loc[key][-1]
            y_mean = self.Cx_model.posterior(
                data.reshape(-1, 2)).mvn.mean.detach().numpy()
            rm.append(
                [ax.text(data[0][0] - 2*self.step_size, y_mean, str(key), color="tab:brown", fontsize=12, weight='bold')])

            # if not (self.Cx_agent_key in self.agent_current_loc):
            #     self.agent_current_loc[self.Cx_agent_key] = []
            # self.agent_current_loc[self.Cx_agent_key].append(self.Cx_data["loc"])
            # for agent_key in self.agent_current_loc:
            #     data = self.agent_current_loc[agent_key][-1]
            #     y_mean = self.Cx_model.posterior(
            #         data.reshape(-1, 2)).mvn.mean.detach().numpy()
            #     # print("Reached loc", data, "for agent ", agent_key)
            #     rm.append([ax.text(data[0], y_mean, str(
            #         agent_key), color="red", fontsize=12)])
        # ax.plot(self.Cx_data["Cx_X"], self.Cx_data["Cx_Y"], "*", color="red", mew=2)
        # plt_bound.lower.Xleft.set_data([self.S_pessi.Xleft.detach().numpy(), Safe.Xleft.detach().numpy()], [
        #     self.constraint, Safe_bound.lower.Xleft.detach().numpy()])
        # plt_bound.lower.Xright.set_data([self.S_pessi.Xright.detach().numpy(), Safe.Xright.detach().numpy()], [
        #     self.constraint, Safe_bound.lower.Xright.detach().numpy()])
        # plt_bound.upper.Xleft.set_data([self.S_opti.Xleft.detach().numpy(), Safe.Xleft.detach().numpy()], [
        #     self.constraint, Safe_bound.upper.Xleft.detach().numpy()])
        # plt_bound.upper.Xright.set_data([self.S_opti.Xright.detach().numpy(), Safe.Xright.detach().numpy()], [
        #     self.constraint, Safe_bound.upper.Xright.detach().numpy()])

        ax.legend(loc='upper left')
        # Draw pessimistic, optimistic and reachable sets
        if self.use_goose:
            k = -0.5
            w = 0.06
            for init_loc in self.safe_boundary:
                rm.append(ax.plot([init_loc.numpy()[0]-w, init_loc.numpy()[0]+w], [
                    k-0.35, k-0.35], color="cyan", linewidth=6.0))
            for key in self.pessi_boundary:
                rm.append(ax.plot([self.grid_V[self.pessi_boundary[key][0]][0]-w, self.grid_V[self.pessi_boundary[key][-1]][0]+w], [
                    k-0.5, k-0.5], color="green", linewidth=6.0))
            for key in self.opti_boundary:
                rm.append(ax.plot([self.grid_V[self.opti_boundary[key][0]][0]-w, self.grid_V[self.opti_boundary[key][-1]][0]+w], [
                    k-0.65-key*0.1, k-0.65-key*0.1], color="gold", linewidth=6.0))
            # for key in self.optimal_feasible_boundary:
            #     rm.append(ax.plot([self.grid_V[self.optimal_feasible_boundary[key][0]-1][0], self.grid_V[self.optimal_feasible_boundary[key][-1]+1][0]], [
            #         k-0.80, k-0.80], color="mediumpurple", linewidth=6.0))

            ax.axhline(y=self.constraint, linestyle='--', color="k")

        # ax.set_title("Iteration " + str(self.n_iter) + " Goose Iter " + str(self.goose_step) +
        #              " Agent " + str(self.Cx_agent_key))
        ax.set_title("Iteration " + str(self.n_iter) + ", SE NLP")

        return rm

    def plot_SE_traj(self, optim, player, f_handle, rm):
        if rm != None:
            for t in rm:
                t[0].remove()

        ax = f_handle.axes[0]
        rm = []
        print(optim.getx())
        print(optim.getu())
        Hm = self.optim_params["Hm"]+1
        y = (-2.0*np.ones_like(optim.getx()[:Hm])).tolist() + [-2.025*np.ones_like(
            optim.getx()[Hm+1])] + (-2.05*np.ones_like(optim.getx()[Hm+1:])).tolist()
        rm.append(ax.plot(player.planned_measure_loc[0],
                          player.planned_measure_loc[1], "*", mew=2, color='tab:green', label='planned'))
        rm.append(ax.plot(player.safe_meas_loc[0][0],
                          player.safe_meas_loc[0][1], "*", mew=5, color='tab:cyan', label='safe'))
        rm.append(ax.plot(optim.getz(), -1.95, "*",
                  mew=2, color='k', label='z'))
        data = player.origin
        if len(self.agent_current_loc[0]) > 1:
            # -2 because -1, is the safe location already.
            data = self.agent_current_loc[0][-2][0]
        rm.append(ax.plot(data[0], data[1], "*", mew=5,
                  color='tab:blue', label='start'))
        rm.append(ax.plot(player.origin[0],
                          player.origin[1] - 0.05, "*", mew=5, color='tab:blue', label='end'))
        rm.append(ax.plot(optim.getx(), y, color='tab:olive'))
        rm.append(ax.plot(optim.getx(), y,
                          "*", color='tab:brown', label='trajectory'))

        ax.set_xbound(lower=-2.0, upper=1)  # ax.set_xlim([-2.0, 1])
        ax.axis("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Iteration " + str(self.n_iter) + ", SE NLP")
        ax.legend(loc='upper left', ncol=3)
        ax.grid()

        ax = f_handle.axes[1]
        rm.append(ax.plot(optim.getx()))
        ax.set_title("Position vs time")
        ax.set_xlabel("time")
        ax.set_ylabel("x")

        ax = f_handle.axes[3]
        rm.append(ax.plot(optim.getu()))
        ax.set_title("Control vs time")
        ax.set_xlabel("time")
        ax.set_ylabel("u")

        print("z", optim.getz())
        return rm

    def plot_Fx_traj(self, player):
        rm = []
        # print(player.obj_optim.getx())
        # print(player.obj_optim.getu())
        Hm = int(self.optim_params["H"]/2)
        y = (-2.0*np.ones_like(player.get_x[:Hm])).tolist() + [-2.025*np.ones_like(
            player.get_x[Hm+1])] + (-2.05*np.ones_like(player.get_x[Hm+1:])).tolist()

        ax = self.f_handle['dyn'].axes[2]
        rm.append(ax.plot(player.planned_measure_loc[0],
                          player.planned_measure_loc[1], "*", mew=5, color='tab:green', label='planned'))
        rm.append(ax.plot(player.current_location[0][0],
                          player.current_location[0][1], "*", mew=5, color='tab:blue', label='start'))
        rm.append(ax.plot(player.origin[0],
                          player.origin[1]-0.05, "*", mew=5, color='tab:blue', label='end'))
        rm.append(ax.plot(player.get_x, y, color='tab:olive'))
        rm.append(ax.plot(player.get_x, y,
                          "*", color='tab:brown', label='trajectory'))
        ax.set_xbound(lower=-2.0, upper=1)  # ax.set_xlim([-2.0, 1])
        ax.axis("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title("Iteration " + str(self.n_iter) + ", Fx NLP")
        ax.legend(loc='upper left', ncol=3)
        ax.grid()
        return rm

    def UpdateDynVisu(self, agent_key, players):
        self.remove_temp_objects('Dyn')
        self.temp_objects['Dyn'] = self.plot_Fx_traj(players[agent_key])

    def UpdateObjectiveVisu(self, agent_key, players, env,  acq_density):
        current_goal = {}
        # xn_planned_dict["measure"][agent_key]
        # it's the next goal, obs point
        current_goal["Fx_X"] = players[agent_key].get_next_to_go_loc()
        current_goal["Fx_Y"] = env.get_density_observation(torch.from_numpy(current_goal["Fx_X"]).float())[
            0].detach()
        self.FxUpdate(players[agent_key].Fx_model, current_goal,
                      acq_density, players[agent_key].get_Fx_data(), agent_key)
        self.remove_temp_objects('Fx')

        if env.Ny == 1:
            self.temp_objects['Fx'] = self.plot1Dobj_GP(self.f_handle['gp'])
        else:
            self.temp_objects['Fx'] = self.plot_Fx(self.f_handle['gp'])

    def remove_temp_objects(self, gp_string):
        if self.temp_objects[gp_string] != None:
            for t in self.temp_objects[gp_string]:
                t[0].remove()

    def UpdateSafeVisu(self, agent_key, players, env):
        self.CxVisuUpdate(players[agent_key].Cx_model, players[agent_key].current_location,
                          players[agent_key].get_Cx_data(), agent_key)  # All playes have communivated hence we can call this function with any agent key

        self.remove_temp_objects('Cx')
        if env.Ny == 1:
            self.temp_objects['Cx'] = self.plot1Dsafe_GP(self.f_handle['gp'])
        else:
            self.temp_objects['Cx'] = self.plot_safe_GP(self.f_handle['gp'])
        # plt.savefig("curr-iter-plot.png")
    
    def time_record(self, time):
        self.iteration_time.append(time)
        print("Time taken", time)

    def record(self, X, U, xi_star, agent_key, players):
        self.traj = X
        self.state_traj.append(X)
        self.input_traj.append(U)
        self.meas_traj.append(xi_star)
        self.opti_path_list.append(self.opti_path)
        self.num_safe_nodes_list.append(self.num_safe_nodes)
        self.utility_minimizer_list.append(self.utility_minimizer)
        self.player_train_pts.append(players[agent_key].get_Cx_data())
        self.player_model.append(players[agent_key].Cx_model)
    
    def save_data(self):
        data_dict = {}
        data_dict["state_traj"] = self.state_traj
        data_dict["input_traj"] = self.input_traj
        data_dict["meas_traj"] = self.meas_traj
        data_dict["player_train_pts"] = self.player_train_pts
        data_dict["player_model"] = self.player_model
        data_dict["opti_path_list"] = self.opti_path_list
        data_dict["utility_minimizer_list"] = self.utility_minimizer_list
        data_dict["num_safe_nodes_list"] = self.num_safe_nodes_list
        data_dict["iteration_time"] = self.iteration_time
        a_file = open(self.save_path + "/data.pkl", "wb")
        pickle.dump(data_dict, a_file)
        a_file.close()

    def extract_data(self):
        a_file = open(self.save_path + "/data.pkl", "rb")
        data_dict = pickle.load(a_file)
        a_file.close()
        self.state_traj = data_dict["state_traj"]
        self.input_traj = data_dict["input_traj"]
        self.meas_traj = data_dict["meas_traj"]
        self.player_train_pts = data_dict["player_train_pts"]
        self.player_model = data_dict["player_model"]
        self.opti_path_list = data_dict["opti_path_list"]
        self.utility_minimizer_list = data_dict["utility_minimizer_list"]
        self.num_safe_nodes_list = data_dict["num_safe_nodes_list"]
        self.iteration_time = data_dict["iteration_time"]