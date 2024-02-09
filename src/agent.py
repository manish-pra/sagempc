# from utils.datatypes import SafeSet
from copy import copy
from dataclasses import dataclass

import gpytorch
import networkx as nx
import numpy as np
import torch
from botorch.models import SingleTaskGP
from gpytorch.kernels import (LinearKernel, MaternKernel,
                              PiecewisePolynomialKernel, PolynomialKernel,
                              RBFKernel, ScaleKernel)
from src.central_graph import (CentralGraph, diag_grid_world_graph,
                               expansion_operator, grid_world_graph)
from src.solver import GoalOPT


class Agent(object):
    def __init__(self, my_key, X_train, Cx_Y_train, Fx_Y_train, params, grid_V) -> None:
        self.my_key = my_key
        self.max_density_sigma = 10
        self.params = params
        self.env_dim = params["common"]["dim"]
        self.Fx_X_train = X_train.reshape(-1, self.env_dim)
        self.Cx_X_train = X_train.reshape(-1, self.env_dim)
        self.Fx_Y_train = Fx_Y_train.reshape(-1, 1)
        self.Cx_Y_train = Cx_Y_train.reshape(-1, 1)
        self.mean_shift_val = params["agent"]["mean_shift_val"]
        self.converged = False
        self.opti = grid_V
        self.grid_V = grid_V
        self.grid_V_prev = grid_V
        self.origin = X_train
        self.x_dim = params["optimizer"]["x_dim"]
        self.Cx_beta = params["agent"]["Cx_beta"]
        self.Fx_beta = params["agent"]["Fx_beta"]
        self.Fx_lengthscale = params["agent"]["Fx_lengthscale"]
        self.Fx_noise = params["agent"]["Fx_noise"]
        self.Cx_lengthscale = params["agent"]["Cx_lengthscale"]
        self.Cx_noise = params["agent"]["Cx_noise"]
        self.constraint = params["common"]["constraint"]
        self.epsilon = params["common"]["epsilon"]
        self.Nx = params["env"]["shape"]["x"]
        self.Ny = params["env"]["shape"]["y"]
        self.counter=11
        self.goal_in_pessi = False
        self.param = params
        if params["env"]["cov_module"] == 'Sq_exp':
            self.Cx_covar_module = ScaleKernel(
                base_kernel=RBFKernel(),)  # ard_num_dims=self.env_dim
            self.Fx_covar_module = ScaleKernel(
                base_kernel=RBFKernel(),)  # ard_num_dims=self.env_dim
        elif params["env"]["cov_module"] == 'Matern':
            self.Cx_covar_module = ScaleKernel(
                base_kernel=MaternKernel(nu=2.5),)  # ard_num_dims=self.env_dim
            self.Fx_covar_module = ScaleKernel(
                base_kernel=MaternKernel(nu=2.5),)  # ard_num_dims=self.env_dim
        else:
            self.Cx_covar_module = ScaleKernel(
                base_kernel=PiecewisePolynomialKernel())  # ard_num_dims=self.env_dim
            self.Fx_covar_module = ScaleKernel(
                base_kernel=PiecewisePolynomialKernel())  # ard_num_dims=self.env_dim

        self.base_graph = diag_grid_world_graph((self.Nx, self.Ny))
        self.diag_graph = diag_grid_world_graph((self.Nx, self.Ny))
        self.optimistic_graph = diag_grid_world_graph((self.Nx, self.Ny))
        self.pessimistic_graph = nx.empty_graph(n=0, create_using=nx.DiGraph())
        self.centralized_safe_graph = diag_grid_world_graph((self.Nx, self.Ny))

        self.Fx_model = self.__update_Fx()
        self.Cx_model = self.__update_Cx()
        self.planned_disk_center = self.Fx_X_train
        self.all_safe_nodes = self.base_graph.nodes
        self.all_unsafe_nodes = []
        self.max_constraint_sigma_goal = None
        self.set_greedy_lcb_pessi_goal = None
        self.planned_disk_center_at_last_meas = X_train.reshape(
            -1, self.env_dim)

        # self.env_start = params["env"]["shape"]["lx"] + params["env"]["start"]
        # x_bound = torch.Tensor(
        #     [params["env"]["start"], params["env"]["start"]+params["env"]["shape"]["lx"]])
        # y_bound = torch.Tensor(
        #     [params["env"]["start"], params["env"]["start"]+params["env"]["shape"]["ly"]])
        # if params["env"]["shape"]["ly"] == 0:
        #     self.dim = 1
        #     a = params["env"]["start"]
        #     y_bound = torch.Tensor([a, a])
        # self.bounds = torch.stack([x_bound, y_bound]).transpose(0, 1)
        a = 1
        self.infeasible = False
        self.info_pt_z = None
        self.safe_meas_loc = self.origin.reshape(-1, 2)
        self.planned_measure_loc =  self.origin
        self.get_utility_minimizer = np.array(params["env"]["goal_loc"])

    def get_gp_sensitivities(self, x_hat, bound, gp):
        self.st_bound = bound
        self.st_gp = gp
        # TODO: change this, it is really bad way of writing/hack
        if x_hat.shape[1] == 1:
            x_hat = np.hstack([x_hat, -2*np.ones_like(x_hat)])
        with gpytorch.settings.fast_pred_var():
            dlb_dx = torch.autograd.functional.jacobian(
                self.funct_sum, torch.from_numpy(x_hat))
            lb = self.funct(torch.from_numpy(x_hat))
        return lb.detach().numpy(), dlb_dx.detach().numpy()[:, :self.params["optimizer"]["x_dim"]]

    def funct_sum(self, X):
        return self.funct(X).sum()

    def funct(self, X):
        if self.st_bound == "LB" and self.st_gp == "Cx":
            self.Cx_model.eval()
            return self.Cx_model(X.float()).mean - self.Cx_beta*2*torch.sqrt(self.Cx_model(X.float()).variance)
        if self.st_bound == "UB" and self.st_gp == "Cx":
            self.Cx_model.eval()
            return self.Cx_model(X.float()).mean + self.Cx_beta*2*torch.sqrt(self.Cx_model(X.float()).variance)
        if self.st_bound == "UB" and self.st_gp == "Fx":
            self.Fx_model.eval()
            return self.Fx_model(X.float()).mean + self.Fx_beta*2*torch.sqrt(self.Fx_model(X.float()).variance)
        if self.st_bound == "LB" and self.st_gp == "Fx":
            self.Fx_model.eval()
            return self.Fx_model(X.float()).mean - self.Fx_beta*2*torch.sqrt(self.Fx_model(X.float()).variance)

    def get_lb_at_curr_loc(self):
        self.st_bound = "LB"
        self.st_gp = "Cx"
        return self.funct(torch.Tensor(self.current_location).reshape(-1,2)).detach().numpy()

    def get_width_at_curr_loc(self):
        self.st_bound = "LB"
        self.st_gp = "Cx"
        lb = self.funct(torch.Tensor(self.current_location).reshape(-1,2)).detach().numpy()
        self.st_bound = "UB"
        ub = self.funct(torch.Tensor(self.current_location).reshape(-1,2)).detach().numpy()
        return ub - lb

    def model_dynamics(self, u):
        x = x+u
        return x

    def policy(self, state):
        K = 1
        u = K*state
        return u

    def opti_UCB(self):
        self.Cx_model.eval()
        X = self.grid_V
        Cx_ucb = self.Cx_model(X.float()).mean + self.Cx_beta*2*torch.sqrt(self.Cx_model(X.float()).variance)
        opti_safe = self.grid_V[Cx_ucb>0]
        self.Fx_model.eval()
        Fx_ucb = self.Fx_model(opti_safe.float()).mean + self.Fx_beta*2*torch.sqrt(self.Fx_model(opti_safe.float()).variance)
        return opti_safe[Fx_ucb.argmax().item()].numpy()
    
    def uncertainity_sampling(self, const_set="pessi"):
        """_summary_: The function will return the point with the highest uncertainty in the pessimistic set

        Returns:
            _type_: a numpy array of shape (2, ) representing the point with highest uncertainty in the pessimistic set
        """
        # self.counter+=1
        # if self.counter>10:
        # self.counter=0
        self.Cx_model.eval()
        V_lower_Cx, V_upper_Cx = self.get_Cx_bounds(self.grid_V)
        self.num_safe_nodes = len(V_lower_Cx[V_lower_Cx>0])
        # X = self.grid_V
        # if set == "pessi":
        #     Cx_cb = self.Cx_model(X.float()).mean - self.Cx_beta*2*torch.sqrt(self.Cx_model(X.float()).variance)
        # else:
        #     Cx_cb = self.Cx_model(X.float()).mean + self.Cx_beta*2*torch.sqrt(self.Cx_model(X.float()).variance)
        intersect_pessi_opti = V_upper_Cx - self.params["common"]["epsilon"]
        init_node = self.get_idx_from_grid(self.origin)
        curr_node = self.get_idx_from_grid(torch.from_numpy(self.current_location))
        self.update_optimistic_graph(intersect_pessi_opti, init_node, self.params["common"]["constraint"],  curr_node, Lc=0)
        Cx_width = V_upper_Cx - V_lower_Cx
        if self.params["algo"]["type"]=="ret_expander" or self.params["algo"]["type"]=="MPC_expander":
            V_lower_Cx_old = V_lower_Cx.clone()
            V_lower_Cx = self.get_Lc_lb(V_lower_Cx)
            sampling_set = np.arange(self.grid_V.shape[0])[torch.logical_and(V_lower_Cx>0,V_lower_Cx_old<=0)]
        else:
            sampling_set = np.arange(self.grid_V.shape[0])[V_lower_Cx>=0]
        if sampling_set.size==0:
            return 0, self.current_location
        global_idx = sampling_set[Cx_width[sampling_set].argmax().item()]
        uncertainity_val = Cx_width[global_idx]
        self.update_pessimistic_graph(V_lower_Cx, init_node, self.params["common"]["constraint"], Lc=0)     
        rem_nodes = list(set(self.optimistic_graph.nodes) - set(self.pessimistic_graph.nodes))
        print(len(rem_nodes))
        # if self.params["algo"]["type"]=="ret_expander" or self.params["algo"]["type"]=="MPC_expander":
        #     pass
        # # Find the uncertainity sampling location directly doing argmax in a set
        # Cx_cb = V_lower_Cx.clone()
        # sampling_set = self.grid_V[Cx_cb>0]
        # Cx_width = V_upper_Cx - V_lower_Cx
        # idx = Cx_width[Cx_cb>0].argmax().item()
        # self.prev_uncertainity_sampling = Cx_width[Cx_cb>0][idx].item(), sampling_set[idx].numpy()
        if self.params["algo"]["init"]=="discrete":
            curr_global_idx = self.get_nearest_pessi_idx(torch.from_numpy(self.current_location)).item()
            self.solver_init_path = self.grid_V[self.get_pessimistic_path(curr_global_idx, global_idx)]
        if len(rem_nodes)==0:
            return 0, self.grid_V[global_idx].numpy()
        else:
            return uncertainity_val.item(), self.grid_V[global_idx].numpy()

    def get_Lc_lb(self, V_lower_Cx):
        dist_matrix = torch.cdist(self.grid_V,self.grid_V,p=2)
        V_lower_Cx_mat = torch.vstack([V_lower_Cx]*V_lower_Cx.shape[0])
        V_lower_Cx_Lc = torch.max(V_lower_Cx_mat - self.params["common"]["Lc"]*dist_matrix,1)[0]
        return V_lower_Cx_Lc
        # V_mod = V_lower_Cx.clone()
        # for idx in range(len(self.grid_V)):
        #     dist_norm = torch.norm(self.grid_V-self.grid_V[idx],1,dim=1)
        #     V_mod[idx] = (V_lower_Cx - self.params["common"]["Lc"]*dist_norm).max()
        # return V_mod

    def update_current_location(self, loc):
        self.current_location = loc
    
    def update_current_state(self, state):
        self.current_state = state
        self.update_current_location(state[:self.x_dim])

    def get_recommendation_pt(self):
        if not self.params["agent"]["Two_stage"]:
            return self.planned_disk_center
        else:
            return self.planned_disk_center
            # PtsToexp = list(set(self.optimistic_graph.nodes) -
            #                 set(self.pessimistic_graph.nodes))
            # if len(PtsToexp) == 0:
            #     return self.planned_disk_center
            # else:
            #     self.set_goal_max_constraint_sigma_under_disc(PtsToexp)
            # return self.max_constraint_sigma_goal

    def update_disc_boundary(self, loc):
        # disc_nodes = self.get_expected_disc(idxfromloc(self.grid_V, loc))
        G = self.base_graph.subgraph(self.full_disc_nodes).copy()
        disc_bound_nodes = [x for x in G.nodes() if (
            G.out_degree(x) <= 3)]
        G1 = self.diag_graph.subgraph(
            disc_bound_nodes).copy()
        self.disc_boundary = list(nx.simple_cycles(G1))
        if len(G1.nodes) == 1:
            self.disc_boundary = list(G1.nodes())

    def communicate_constraint(self, X_set, Cx_set):
        for newX, newY in zip(X_set, Cx_set):
            self.__update_Cx_set(newX, newY)

    def communicate_density(self, X_set, Fx_set):
        for newX, newY in zip(X_set, Fx_set):
            self.__update_Fx_set(newX, newY)

    def update_Cx_gp(self, newX, newY):
        self.__update_Cx_set(newX, newY)
        self.__update_Cx()
        return self.Cx_model

    def update_Cx_gp_with_current_data(self):
        self.__update_Cx()
        return self.Cx_model

    def update_Fx_gp(self, newX, newY):
        self.__update_Fx_set(newX, newY)
        self.__update_Fx()
        return self.Fx_model

    def update_Fx_gp_with_current_data(self):
        self.__update_Fx()
        return self.Fx_model

    def __update_Cx_set(self, newX, newY):
        newX = newX.reshape(-1, self.env_dim)
        newY = newY.reshape(-1, 1)
        self.Cx_X_train = torch.cat(
            [self.Cx_X_train, newX]).reshape(-1, self.env_dim)
        self.Cx_Y_train = torch.cat([self.Cx_Y_train, newY]).reshape(-1, 1)

    def __update_Cx(self):
        self.Cx_model = SingleTaskGP(self.Cx_X_train, self.Cx_Y_train)
        # 1.2482120543718338
        self.Cx_model.covar_module.base_kernel.lengthscale = self.Cx_lengthscale
        self.Cx_model.likelihood.noise = self.Cx_noise
        # mll = ExactMarginalLogLikelihood(
        #     self.Cx_model.likelihood, self.Cx_model)
        # fit_gpytorch_model(mll)
        return self.Cx_model

    def __update_Fx_set(self, newX, newY):
        newX = newX.reshape(-1, self.env_dim)
        newY = newY.reshape(-1, 1)
        self.Fx_X_train = torch.cat(
            [self.Fx_X_train, newX]).reshape(-1, self.env_dim)
        self.Fx_Y_train = torch.cat([self.Fx_Y_train, newY]).reshape(-1, 1)

    def __update_Fx(self):
        Fx_Y_train = self.__mean_corrected(self.Fx_Y_train)
        self.Fx_model = SingleTaskGP(self.Fx_X_train, Fx_Y_train)
        self.Fx_model.covar_module.base_kernel.lengthscale = self.Fx_lengthscale
        self.Fx_model.likelihood.noise = self.Fx_noise
        # mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # fit_gpytorch_model(mll)
        return self.Fx_model

    def __predict_Fx(self, newX):
        newX = newX.reshape(-1, self.env_dim)
        newY = self.Fx_model.posterior(newX).mean
        Fx_Y_train = self.__mean_corrected(self.Fx_Y_train)
        Fx_X_train = torch.cat([self.Fx_X_train, newX]
                               ).reshape(-1, self.env_dim)
        Fx_Y_train = torch.cat([Fx_Y_train, newY]).reshape(-1, 1)
        Fx_model = SingleTaskGP(Fx_X_train, Fx_Y_train)
        Fx_model.covar_module.base_kernel.lengthscale = self.Fx_lengthscale
        Fx_model.likelihood.noise = self.Fx_noise
        return Fx_model

    def __mean_corrected(self, variable):
        return variable - self.mean_shift_val

    def get_Fx_bounds(self, V):
        lower_Fx, upper_Fx = self.Fx_model.posterior(V).mvn.confidence_region()
        lower_Fx, upper_Fx = scale_with_beta(lower_Fx, upper_Fx, self.Fx_beta)
        self.lower_Fx = torch.max(self.lower_Fx, lower_Fx)
        self.upper_Fx = torch.min(self.upper_Fx, upper_Fx)
        return self.lower_Fx, self.upper_Fx

    def save_posterior_normalization_const(self, ):
        lower_Fx, upper_Fx = self.Fx_model.posterior(
            self.grid_V).mvn.confidence_region()
        lower_Fx, upper_Fx = scale_with_beta(lower_Fx, upper_Fx, self.Fx_beta)
        # acq_density = self.Fx_beta*upper_Fx  # acq without mean shift
        diff = upper_Fx - lower_Fx
        self.posterior_normalization_const = diff.max().detach()

    def UpdateConvergence(self, converged):
        self.converged = converged

    def update_union_graph(self, union_graph):
        self.union_graph = union_graph

    def update_optimistic_graph(self, upper_bound, init_node, thresh,curr_node, Lc):
        self.optimistic_graph = expansion_operator(
            self.optimistic_graph, upper_bound, init_node, thresh,curr_node, Lc)
        print("Nodes in optimistic graph:", len(self.optimistic_graph.nodes))
        # Lc*step_size is imp since this is the best we can do to create expander set. If this is not satisfied we may not be able to expand
        return True
    
    def dist(self, a, b):
        return 1
        # return 1/np.abs(a-b)
        # (x1, y1) = a
        # (x2, y2) = b
        # return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
    def get_optimistic_path(self, source, target):
        return nx.algorithms.shortest_paths.astar_path(self.optimistic_graph, source, target, heuristic=self.dist)
        # return nx.algorithms.shortest_paths.dijkstra_path(self.optimistic_graph, source, target)
        # return nx.algorithms.shortest_paths.bidirectional_dijkstra(self.optimistic_graph, source, target)[1]
        # return nx.algorithms.shortest_paths.bellman_ford_path(self.optimistic_graph, source, target)

    def get_pessimistic_path(self, source, target):
        return nx.algorithms.shortest_paths.astar_path(self.pessimistic_graph, source, target, heuristic=self.dist)

    def update_pessimistic_graph(self, lower_bound, init_node, thresh, Lc):
        total_safe_nodes = torch.arange(0, lower_bound.shape[0])[
            lower_bound > thresh]
        total_safe_nodes = torch.unique(
            torch.cat([total_safe_nodes, init_node.reshape(-1)]))
        total_safe_graph = self.base_graph.subgraph(total_safe_nodes.numpy())
        edges = nx.algorithms.traversal.breadth_first_search.bfs_edges(
            total_safe_graph, init_node.item())  # to remove non connected areas
        connected_nodes = [init_node.item()] + [v for u, v in edges]
        self.pessimistic_graph = update_graph(
            self.pessimistic_graph, self.base_graph, nodes_to_add=connected_nodes)
        print("Nodes in pesimistic graph:", len(self.pessimistic_graph.nodes))
        return True

    def get_nearest_pessi_idx(self, loc):
        list_pessi_nodes = list(self.pessimistic_graph.nodes)
        dist = self.grid_V[list_pessi_nodes] - loc
        ret = list_pessi_nodes[torch.norm(dist, 2, dim=1).argmin()]
        return torch.IntTensor([ret])
    
    def get_nearest_opti_idx(self, loc):
        list_pessi_nodes = list(self.optimistic_graph.nodes)
        dist = self.grid_V[list_pessi_nodes] - loc
        ret = list_pessi_nodes[torch.norm(dist, 2, dim=1).argmin()]
        return torch.IntTensor([ret])

    def update_centralized_unit(self, all_safe_nodes, all_unsafe_nodes, centralized_safe_graph, unsafe_edges_set, unreachable_nodes):
        self.all_safe_nodes = all_safe_nodes
        self.all_unsafe_nodes = all_unsafe_nodes
        self.centralized_safe_graph = centralized_safe_graph
        self.unsafe_edges_set = unsafe_edges_set
        self.unreachable_nodes = unreachable_nodes

    def set_others_meas_loc(self, list_meas_loc):
        self.list_others_meas_loc = list_meas_loc.copy()

    def set_maximizer_goal(self, xi_star):
        self.planned_measure_loc = xi_star

    def get_next_to_go_loc(self):
        return self.planned_measure_loc

    def update_next_to_go_loc(self, loc):
        self.planned_measure_loc = loc

    def update_oracle_XU(self, X, U):
        self.get_x = X
        self.get_u = U

    def extract_optim_info(self, optim):
        self.safe_meas_loc = torch.Tensor(
            [optim.getx()[self.params["optimizer"]["Hm"]+1], -2.0]).reshape(-1, 2)
        reached = torch.norm(self.safe_meas_loc -
                             self.planned_measure_loc) < 0.05
        min_infeasible = np.min(optim.opti.stats()['iterations']['inf_pr'])
        self.infeasible = min_infeasible > 0.004
        print("feasibility check", self.infeasible, min_infeasible)
        self.info_pt_z = optim.getz()
        self.optim_getx = optim.getx()
        self.optim_getu = optim.getu()
        return reached

    def get_maxCI_point(self, V):
        # 2.1) Get the density function \mu to optimize
        lower_Fx, upper_Fx = self.Fx_model.posterior(V).mvn.confidence_region()
        # acq_density = self.Fx_beta*upper_Fx  # acq without mean shift
        lower_Fx, upper_Fx = scale_with_beta(lower_Fx, upper_Fx, self.Fx_beta)
        diff = upper_Fx - lower_Fx
        xn_star = torch.Tensor(
            [V[diff.argmax()], V[diff.argmax()]]).reshape(-1)
        acq_density = diff
        return xn_star, acq_density, self.V

    def get_uncertain_points(self, V, model_Fx):
        # 2.1) Get the density function \mu to optimize
        lower_Fx, upper_Fx = model_Fx.posterior(V).mvn.confidence_region()
        # acq_density = self.Fx_beta*upper_Fx  # acq without mean shift
        lower_Fx, upper_Fx = scale_with_beta(lower_Fx, upper_Fx, self.Fx_beta)
        diff = upper_Fx - lower_Fx
        x1_star = V[diff.argmax()]
        return x1_star

    def get_2maxCI_points(self, V, n_soln):
        model_Fx = self.Fx_model
        xn_star = torch.empty(0)
        for _ in range(n_soln):
            x1_star = self.get_uncertain_points(V, model_Fx)
            xn_star = torch.cat([xn_star, x1_star.reshape(-1)])
            model_Fx = self.__predict_Fx(xn_star)

        lower_Fx, upper_Fx = self.Fx_model.posterior(V).mvn.confidence_region()
        # acq_density = self.Fx_beta*upper_Fx  # acq without mean shift
        lower_Fx, upper_Fx = scale_with_beta(lower_Fx, upper_Fx, self.Fx_beta)
        acq_density = (upper_Fx + lower_Fx)/2 + self.mean_shift_val
        return xn_star, acq_density.detach(), self.V


    def get_lcb_density(self):
        # 2.1) Get the density function \mu to optimize
        lower_Fx, upper_Fx = self.Fx_model.posterior(
            self.grid_V).mvn.confidence_region()
        # acq_density = self.Fx_beta*upper_Fx  # acq without mean shift
        lower_Fx, upper_Fx = scale_with_beta(lower_Fx, upper_Fx, self.Fx_beta)
        acq_density = lower_Fx + self.mean_shift_val
        return acq_density

    # def get_maximizer_point(self, n_soln):
        # self.UCB = UpperConfidenceBound(self.Fx_model, beta=2*self.Fx_beta)
        # candidate, acq_value = optimize_acqf(
        #     self.UCB, bounds=self.bounds, q=1, num_restarts=5, raw_samples=20,
        # )
        # print(candidate, acq_value)
    #     return candidate.detach(), acq_value.detach()

    def get_maximizer_point(self, n_soln):
        # self.obj_optim.setstartparam(self.origin[0])
        self.obj_optim.setstartparam(self.current_location[0][0])
        self.obj_optim.setendparam(self.origin[0])
        self.obj_optim.solve()
        self.obj_optim.print()
        print(self.obj_optim.opti.value(self.obj_optim.x))
        candidate = self.obj_optim.get_candidate()
        acq_value = self.obj_optim.UB_obj_eval.call([candidate])[0].toarray()
        candidate = torch.Tensor([candidate, -2.0]).reshape(-1, 2)
        return candidate, torch.from_numpy(acq_value)

    # def get_F_of_x_for_fix_pts(self, X_fix):
    def get_Cx_bounds(self, grid_V):
        V_lower_Cx, V_upper_Cx = self.Cx_model.posterior(
            grid_V).mvn.confidence_region()
        V_lower_Cx = V_lower_Cx.detach()
        V_upper_Cx = V_upper_Cx.detach()
        V_lower_Cx, V_upper_Cx = scale_with_beta(
            V_lower_Cx, V_upper_Cx, self.Cx_beta)
        # front_shift_idx = int((grid_V[0] - self.V_prev[0])/0.12 + 0.01)
        # rear_shift_idx = int((self.V_prev[-1]-grid_V[-1])/0.12 + 0.01)
        # n = self.V_lower_Cx.shape[0]
        # temp_lower_Cx = self.V_lower_Cx[front_shift_idx:
        #                                 n-1*rear_shift_idx]
        # temp_upper_Cx = self.V_upper_Cx[front_shift_idx:
        #                                 n-1*rear_shift_idx]
        # delta_w = (temp_upper_Cx - temp_lower_Cx) - (V_upper_Cx-V_lower_Cx)
        # # print(self.Cx_X_train.shape, self.Cx_X_train)
        # # print("W",  delta_w)
        # # self.V_lower_Cx = torch.max(
        # #     temp_lower_Cx, V_lower_Cx)  # element wise max
        # # self.V_upper_Cx = torch.min(
        # #     temp_upper_Cx, V_upper_Cx)  # element wise min
        # self.V_lower_Cx = V_lower_Cx
        # self.V_upper_Cx = V_upper_Cx
        # self.V_prev = grid_V
        return V_lower_Cx, V_upper_Cx

    def update_graph(self, Safe):
        V_lower_Cx, V_upper_Cx = self.get_Cx_bounds(self.grid_V)

        # Order matters here
        self.update_pessimistic_graph(
            V_lower_Cx, Safe, self.constraint, self.Lc)

        self.update_optimistic_graph(
            V_upper_Cx-self.epsilon, Safe, self.constraint, self.Lc)

        return True

    def get_idx(self, positions):
        idx = []
        for position in positions:
            idx.append(torch.abs(torch.Tensor(
                self.V) - position).argmin().item())
        return idx
    
    def get_idx_from_grid(self, position):
        idx = torch.sum(torch.abs(position - self.grid_V),1).argmin()
        return idx

    def get_Cx_data(self):
        data = {}
        data["Cx_X"] = self.Cx_X_train.detach()
        data["Cx_Y"] = self.Cx_Y_train.detach()
        data["loc"] = self.current_location
        data["info_pt_z"] = self.info_pt_z
        data["safe_meas"] = self.safe_meas_loc
        data["plan_meas"] = self.planned_measure_loc
        return data

    def get_Fx_data(self):
        data = {}
        data["Fx_X"] = self.Fx_X_train.detach()
        data["Fx_Y"] = self.Fx_Y_train.detach()
        data["loc"] = self.current_location
        return data


def scale_with_beta(lower, upper, beta):
    temp = lower*(1+beta)/2 + upper*(1-beta)/2
    upper = upper*(1+beta)/2 + lower*(1-beta)/2
    lower = temp
    return lower, upper


def update_graph(G, base_G, nodes_to_remove=None, nodes_to_add=None):
    """
    Updates nodes of a given graph using connectivity structure of base graph.

    Parameters
    ----------
    G: nx.Graph
        Graph to update
    base_G: nx.Graph
        Base graph that gives connectivity structure
    nodes_to_remove: ndarray
        array of nodes to remove from G
    nodes_to_add: ndarray
        array of nodes to add to G

    Returns
    -------
    G: nx.Graph
        Updated graph
    """
    if nodes_to_add is not None and len(nodes_to_add) > 0:
        nodes = np.unique(
            np.hstack((np.asarray(list(G.nodes)), np.asarray(nodes_to_add))))
        nodes = nodes.astype(np.int64)
        G = base_G.subgraph(nodes).copy()

    if nodes_to_remove is not None and nodes_to_remove.size > 0:
        for n in nodes_to_remove:
            G.remove_node(n)
            G.remove_edges_from(base_G.edges(n))

    return G


if __name__ == "__main__":
    # Initialization:
    S0 = [70, 71, 72]
    X_train = torch.Tensor([i for i in S0]).reshape(-1, 1)
    Fx_Y_train = torch.Tensor([i for i in S0]).reshape(-1, 1)
    Cx_Y_train = torch.Tensor([i for i in S0]).reshape(-1, 1)
    p1 = Agent(X_train, Cx_Y_train, Fx_Y_train, beta=3,
               mean_shift_val=2, constraint=0.5, eps=1e-2, explore_exploit_strategy=1, init_safe=S0, V=S0, Lc=0.5)

    print(p1.update_Cx_gp(X_train, Cx_Y_train))
    print(p1.update_Fx_gp(X_train, Fx_Y_train))
