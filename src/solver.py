import timeit

import casadi as ca
import numpy as np
import torch
import matplotlib.pyplot as plt
from acados_template import AcadosOcpSolver, AcadosSimSolver

from src.utils.ocp import export_oracle_ocp, export_sempc_ocp, export_sim

# The class below is an optimizer class,
# it takes in GP function, x_g and rest are parameters
class SEMPC_solver(object):
    def __init__(self, params) -> None:
        ocp = export_sempc_ocp(params)
        self.name_prefix = params["algo"]["type"] + '_env_' + str(params["env"]["name"]) + '_i_' + str(params["env"]["i"]) + '_'
        self.ocp_solver = AcadosOcpSolver(
            ocp, json_file= self.name_prefix + 'acados_ocp_sempc.json')
        self.ocp_solver.store_iterate(self.name_prefix + 'ocp_initialization.json')

        # sim = export_sim(params, 'sim_sempc')
        # self.sim_solver = AcadosSimSolver(
        #     sim, json_file='acados_sim_sempc.json')
        self.H = params["optimizer"]["H"]
        self.Hm = params["optimizer"]["Hm"]
        self.max_sqp_iter = params["optimizer"]["SEMPC"]["max_sqp_iter"]
        self.tol_nlp = params["optimizer"]["SEMPC"]["tol_nlp"]
        self.nx = ocp.model.x.size()[0]
        self.nu = ocp.model.u.size()[0]
        self.eps = params["common"]["epsilon"]
        self.n_order = params["optimizer"]["order"]
        self.x_dim = params["optimizer"]["x_dim"]
        self.params = params
        if params["agent"]["dynamics"]=="robot":
            self.state_dim = self.n_order*self.x_dim + 1
        else:
            self.state_dim = self.n_order*self.x_dim

    def initilization(self, sqp_iter, x_h, u_h):
        for stage in range(self.H):
            # current stage values
            x_h[stage, :] = self.ocp_solver.get(stage, "x") 
            u_h[stage, :] = self.ocp_solver.get(stage, "u")
        x_h[self.H, :] = self.ocp_solver.get(self.H, "x")
        if sqp_iter == 0:
            x_h_old = x_h.copy()
            u_h_old = u_h.copy()
            if self.params["algo"]["type"] == "ret_expander" or self.params["algo"]["type"] == "MPC_expander":
                u_h_old[:, -self.x_dim:] = x_h_old[:-1, :self.x_dim].copy()
            # initialize the first SQP iteration.
            for stage in range(self.H):
                if stage < (self.H - self.Hm):
                    # current stage values
                    x_init = x_h_old[stage + self.Hm, :].copy() 
                    u_init = u_h_old[stage + self.Hm, :].copy()
                    x_init[-1] = (x_h_old[stage + self.Hm, -1] - x_h_old[self.Hm, -1]).copy() 
                    self.ocp_solver.set(stage, "x", x_init)
                    self.ocp_solver.set(stage, "u", u_init)
                    x_h[stage, :] = x_init.copy()
                    u_h[stage, :] = u_init.copy()
                    half_time = x_init[-1].copy() 
                else:
                    dt = (1.0-half_time)/self.Hm
                    x_init = x_h_old[self.H, :].copy() # reached the final state
                    x_init[-1] = half_time + dt*(stage-self.Hm)
                    z_init = x_init[0:self.x_dim]
                    if self.params["algo"]["type"] == "ret_expander" or self.params["algo"]["type"] == "MPC_expander":
                        u_init = np.concatenate([np.array([0.0,0.0, dt]), z_init])
                    else:
                        u_init = np.array([0.0,0.0, dt])
                    self.ocp_solver.set(stage, "x", x_init)
                    self.ocp_solver.set(stage, "u", u_init)
                    x_h[stage, :] = x_init.copy()
                    u_h[stage, :] = u_init.copy()
            self.ocp_solver.set(self.H, "x", x_init)
            x_init[-1] = half_time + dt*(self.H-self.Hm)
            x_h[self.H, :] = x_init.copy() 
            # x0 = np.zeros(self.state_dim)
            # x0[:self.x_dim] = np.ones(self.x_dim)*0.72
            # x0=np.concatenate([x0, np.array([0.0])])
            # x_init=x0.copy()
            # # x_init = self.ocp_solver.get(0, "x")
            # u_init = self.ocp_solver.get(0, "u")
            # Ts = 1/200
            # # MPC controller
            # x_init = np.array([0.72,0.72,0.0,0.0, 0.0])
            # u_init = np.array([-0.2,-0.2, Ts])


            #     x_h[stage, :] = x_init
            #     u_h[stage, :] = u_init
            # x_h[self.H, :] = x_init
            # self.ocp_solver.set(self.H, "x", x_init)    
        return x_h, u_h    

    def path_init(self,path):
        split_path = np.zeros((self.H+1,self.x_dim))
        interp_h = np.arange(self.Hm)
        path_step = np.linspace(0,self.Hm,path.shape[0])
        x_pos = np.interp(interp_h, path_step, path.numpy()[:,0])
        y_pos = np.interp(interp_h, path_step, path.numpy()[:,1])
        split_path[:self.Hm,0], split_path[:self.Hm,1] = x_pos, y_pos
        split_path[self.Hm:,:] = np.ones_like(split_path[self.Hm:,:])*path[-1].numpy()
        # split the path into horizons
        for stage in range(self.H+1):
            x_init = self.ocp_solver.get(stage, "x")
            x_init[:self.x_dim] = split_path[stage]
            self.ocp_solver.set(stage, "x", x_init)

    def solve(self, player):
        x_h = np.zeros((self.H+1, self.state_dim+1))
        z_h = np.zeros((self.H+1, self.x_dim))
        if self.params["algo"]["type"] == "ret_expander" or self.params["algo"]["type"] == "MPC_expander":
            u_h = np.zeros((self.H, self.x_dim+1+self.x_dim)) # u_dim
        else:
            u_h = np.zeros((self.H, self.x_dim+1)) # u_dim
        w = 1e-3*np.ones(self.H+1)
        we = 1e-8*np.ones(self.H+1)
        we[int(self.H-1)] = 10000
        # w[:int(self.Hm)] = 1e-1*np.ones(self.Hm)
        w[int(self.Hm)] = self.params["optimizer"]["w"]
        cw = 1e+3*np.ones(self.H+1)
        if not player.goal_in_pessi:
            cw[int(self.Hm)] = 1
        xg = np.ones((self.H+1, self.x_dim))*player.get_next_to_go_loc()
        x_origin = player.origin[:self.x_dim].numpy()
        x_terminal = np.zeros(self.state_dim)
        x_terminal[:self.x_dim] = np.ones(self.x_dim)*x_origin
        for sqp_iter in range(self.max_sqp_iter):
            self.ocp_solver.options_set('rti_phase', 1)
            if self.params["algo"]["type"] == "ret" or self.params["algo"]["type"] == "ret_expander":
                if player.goal_in_pessi:
                    x_h, u_h = self.initilization(sqp_iter, x_h, u_h)
                else:
                    for stage in range(self.H):
                        # current stage values
                        x_h[stage, :] = self.ocp_solver.get(stage, "x") 
                        u_h[stage, :] = self.ocp_solver.get(stage, "u")
                    x_h[self.H, :] = self.ocp_solver.get(self.H, "x")
            else:
            #    pass
               x_h, u_h = self.initilization(sqp_iter, x_h, u_h)
               if self.params["algo"]["init"]=="discrete":
                   self.path_init(player.solver_init_path)

            gp_val, gp_grad = player.get_gp_sensitivities(
                x_h[:, :self.x_dim], "LB", "Cx")  # pessimitic safe location
            UB_cx_val, UB_cx_grad = player.get_gp_sensitivities(
                x_h[:, :self.x_dim], "UB", "Cx")  # optimistic safe location
            if self.params["algo"]["type"] == "ret_expander" or self.params["algo"]["type"] == "MPC_expander":
                LB_cz_val, LB_cz_grad = player.get_gp_sensitivities(u_h[:,-self.x_dim:], "LB", "Cx") 
                for stage in range(self.H):
                    self.ocp_solver.set(stage, "p", np.hstack(
                        (gp_val[stage], gp_grad[stage], x_h[stage, :self.state_dim], 
                        xg[stage], w[stage], x_terminal, UB_cx_val[stage], UB_cx_grad[stage], cw[stage],u_h[stage,-self.x_dim:], LB_cz_val[stage], LB_cz_grad[stage])))
                stage = self.H # stage already is at self.H
                self.ocp_solver.set(stage, "p", np.hstack(
                        (gp_val[stage], gp_grad[stage], x_h[stage, :self.state_dim], 
                        xg[stage], w[stage], x_terminal, UB_cx_val[stage], UB_cx_grad[stage], cw[stage],u_h[stage-1,-self.x_dim:], LB_cz_val[stage-1], LB_cz_grad[stage-1]))) # last 3 "stage-1" are dummy values
            elif self.params["algo"]["type"] == "MPC_Xn":
                for stage in range(self.H+1):
                    self.ocp_solver.set(stage, "p", np.hstack(
                        (gp_val[stage], gp_grad[stage], x_h[stage, :self.state_dim],
                        xg[stage], w[stage], x_terminal, UB_cx_val[stage], UB_cx_grad[stage], cw[stage], we[stage])))                
            else:
                for stage in range(self.H+1):
                    self.ocp_solver.set(stage, "p", np.hstack(
                        (gp_val[stage], gp_grad[stage], x_h[stage, :self.state_dim],
                        xg[stage], w[stage], x_terminal, UB_cx_val[stage], UB_cx_grad[stage], cw[stage])))
            status = self.ocp_solver.solve()

            self.ocp_solver.options_set('rti_phase', 2)
            t_0 = timeit.default_timer()
            status = self.ocp_solver.solve()
            t_1 = timeit.default_timer()
            # self.ocp_solver.print_statistics()
            print("cost", self.ocp_solver.get_cost())
            residuals = self.ocp_solver.get_residuals()
            
            X, U, Sl = self.get_solution()
            # print(X)
            # for stage in range(self.H):
            #     print(stage, " constraint ", self.constraint(LB_cz_val[stage], LB_cz_grad[stage], U[stage,3:5], X[stage,0:4], u_h[stage,-self.x_dim:], x_h[stage, :self.state_dim], self.params["common"]["Lc"]))
            if sqp_iter==(self.max_sqp_iter-1):
                if self.params["visu"]["show"]:
                    plt.figure(2)
                    if self.params["algo"]["type"] == "ret_expander" or self.params["algo"]["type"] == "MPC_expander":
                        plt.plot(X[:,0],X[:,1], color="tab:green") # state
                        plt.plot(U[:,3],U[:,4], color="tab:blue") # l(x)
                    else:
                        plt.plot(X[:,0],X[:,1], color="tab:green")
                    plt.xlim(self.params["env"]["start"],self.params["env"]["start"] + self.params["visu"]["step_size"]*self.params["env"]["shape"]["x"])
                    plt.ylim(self.params["env"]["start"],self.params["env"]["start"] + self.params["visu"]["step_size"]*self.params["env"]["shape"]["y"])
                    # plt.axes().set_aspect('equal')
                    plt.savefig("temp.png")
            # print("statistics", self.ocp_solver.get_stats("statistics"))
            if max(residuals) < self.tol_nlp:
                print("Residual less than tol", max(
                    residuals), " ", self.tol_nlp)
                break
            if self.ocp_solver.status != 0:
                print("acados returned status {} in closed loop solve".format(
                    self.ocp_solver.status))
                self.ocp_solver.reset()
                self.ocp_solver.load_iterate(self.name_prefix + 'ocp_initialization.json')


    def constraint(self, lb_cz_lin, lb_cz_grad, model_z, model_x, z_lin, x_lin, Lc):
        x_dim = self.x_dim
        tol = 1e-5
        ret = lb_cz_lin + lb_cz_grad.T @ (model_z-z_lin) - (Lc/(ca.norm_2(x_lin[:x_dim] - z_lin)+tol))*((x_lin[:x_dim] - z_lin).T@(model_x-x_lin)[:x_dim]) - (Lc/(ca.norm_2(x_lin[:x_dim] - z_lin)+tol))*((z_lin-x_lin[:x_dim]).T@(model_z-z_lin)) - Lc*ca.norm_2(x_lin[:x_dim] - z_lin)
        # ret = lb_cz_lin + lb_cz_grad.T @ (model_z-z_lin) - 2*Lc*(x_lin[:x_dim] - z_lin).T@(model_x-x_lin)[:x_dim] - 2*Lc*(z_lin-x_lin[:x_dim]).T@(model_z-z_lin) - Lc*(x_lin[:x_dim] - z_lin).T@(x_lin[:x_dim] - z_lin)
        return ret, lb_cz_lin + lb_cz_grad.T @ (model_z-z_lin)
    
    def model_ss(self, model_x):
        val = (model_x - model.f_expl_expr[:-1])

    def get_solution(self):
        X = np.zeros((self.H+1, self.nx))
        U = np.zeros((self.H, self.nu))
        Sl = np.zeros((self.H+1))

        # get data
        for i in range(self.H):
            X[i, :] = self.ocp_solver.get(i, "x")
            U[i, :] = self.ocp_solver.get(i, "u")
            # Sl[i] = self.ocp_solver.get(i, "sl")

        X[self.H, :] = self.ocp_solver.get(self.H, "x")
        return X, U, Sl

    def get_solver_status():
        return None


class Oracle_solver(object):
    def __init__(self, params) -> None:
        ocp = export_oracle_ocp(params)
        self.ocp_solver = AcadosOcpSolver(
            ocp, json_file= params["algo"]["type"] +  'acados_ocp_oracle.json')

        # sim = export_sim(params, 'sim_oracle')
        # self.sim_solver = AcadosSimSolver(
        #     sim, json_file='acados_sim_oracle.json')
        self.H = params["optimizer"]["H"]
        self.Hm = params["optimizer"]["Hm"]
        self.max_sqp_iter = params["optimizer"]["oracle"]["max_sqp_iter"]
        self.tol_nlp = params["optimizer"]["oracle"]["tol_nlp"]
        self.nx = ocp.model.x.size()[0]
        self.nu = ocp.model.u.size()[0]
        self.eps = params["common"]["epsilon"]
        self.n_order = params["optimizer"]["order"]
        self.x_dim = params["optimizer"]["x_dim"]
        # should every player have its own solver?
        pass

    def solve(self, player):
        x_h = np.zeros((self.H+1, self.state_dim+1))
        u_h = np.zeros((self.H, self.x_dim+1))
        w = 1e-3*np.ones(self.H+1)
        w[int(self.H/2)] = 10
        xg = np.ones((self.H+1, self.x_dim))*player.get_utility_minimizer
        x_origin = player.origin[:self.x_dim].numpy()
        x_terminal = np.zeros(self.state_dim)
        x_terminal[:self.x_dim] = np.ones(self.x_dim)*x_origin
        # xg = player.opti_UCB()*np.ones((self.H+1, self.x_dim))
        for sqp_iter in range(self.max_sqp_iter):
            self.ocp_solver.options_set('rti_phase', 1)
            for stage in range(self.H):
                # current stage values
                x_h[stage, :] = self.ocp_solver.get(stage, "x") 
                u_h[stage, :] = self.ocp_solver.get(stage, "u")
                # if sqp_iter<1:
                #     x_h[stage,0:2] += np.random.randint(-100,100, size=(2))/100000
            x_h[self.H, :] = self.ocp_solver.get(self.H, "x")
            # if sqp_iter == 0:
            #     x_h_old = x_h.copy()
            #     u_h_old = u_h.copy()
            #     # initialize the first SQP iteration.
            #     for stage in range(self.H):
            #         if stage < self.Hm:
            #             # current stage values
            #             x_init = x_h_old[stage + self.Hm, :].copy() 
            #             u_init = u_h_old[stage + self.Hm, :].copy()
            #             x_init[-1] = (x_h_old[stage + self.Hm, -1] - x_h_old[self.Hm, -1]).copy() 
            #             self.ocp_solver.set(stage, "x", x_init)
            #             self.ocp_solver.set(stage, "u", u_init)
            #             x_h[stage, :] = x_init.copy()
            #             u_h[stage, :] = u_init.copy()
            #             half_time = x_init[-1].copy() 
            #         else:
            #             dt = (1-half_time)/self.Hm
            #             x_init = x_h_old[self.H, :].copy() # reached the final state
            #             x_init[-1] = half_time + dt*(stage-self.Hm)
            #             u_init = np.array([0.0,0.0, dt])
            #             self.ocp_solver.set(stage, "x", x_init)
            #             self.ocp_solver.set(stage, "u", u_init)
            #             x_h[stage, :] = x_init.copy()
            #             u_h[stage, :] = u_init.copy()
            #     self.ocp_solver.set(self.H, "x", x_init)
            #     x_init[-1] = half_time + dt*(self.H-self.Hm)
            #     x_h[self.H, :] = x_init.copy() 
            # # print("x_h", x_h)
            # # eps_lin = self.ocp_solver.get(stage, "eps")
            UB_cx_val, UB_cx_grad = player.get_gp_sensitivities(
                x_h[:, :self.x_dim], "UB", "Cx")  # optimistic safe location
            UB_cx_val -= self.eps
            # UB_fx_val, UB_fx_grad = player.get_gp_sensitivities(
            #     x_h[:, :self.x_dim], "UB", "Fx")
            # LB_cx_val, LB_cx_grad = player.get_gp_sensitivities(
            #     x_h[:, 0], "LB", "Cx")
            # UB_fx_grad = UB_cx_grad - LB_cx_grad

            for stage in range(self.H+1):
                self.ocp_solver.set(stage, "p",
                                    np.hstack((UB_cx_val[stage],
                                               UB_cx_grad[stage],
                                               x_h[stage, :self.state_dim], x_terminal, xg[stage], w[stage])))
            # self.ocp_solver.set(int(self.H/2), "x",
            #                     player.planned_measure_loc[0].reshape(1, 1).numpy())
            status = self.ocp_solver.solve()
            #
            self.ocp_solver.options_set('rti_phase', 2)
            t_0 = timeit.default_timer()
            status = self.ocp_solver.solve()
            t_1 = timeit.default_timer()
            # self.ocp_solver.print_statistics()
            print("cost", self.ocp_solver.get_cost())
            residuals = self.ocp_solver.get_residuals()
            if max(residuals) < self.tol_nlp:
                print("Residual less than tol", max(
                    residuals), " ", self.tol_nlp)
                break

    def get_solution(self):
        X = np.zeros((self.H+1, self.nx))
        U = np.zeros((self.H, self.nu))

        # get data
        for i in range(self.H):
            X[i, :] = self.ocp_solver.get(i, "x")
            U[i, :] = self.ocp_solver.get(i, "u")

        X[self.H, :] = self.ocp_solver.get(self.H, "x")
        return X, U

    def get_solver_status():
        return None


class GoalOPT(object):
    def __init__(self, optim_param, common_param, agent_param, LB_const_eval, UB_const_eval, UB_obj_eval) -> None:
        self.H = optim_param["H"]
        self.optim_param = optim_param
        self.Hm = optim_param["Hm"]
        self.u_min = optim_param["u_min"]
        self.u_max = optim_param["u_max"]
        self.x_min = optim_param["x_min"]
        self.x_max = optim_param["x_max"]
        self.Lc = agent_param["Lc"]
        self.constraint = common_param["constraint"]
        self.epsQ = common_param["epsilon"]
        self.UB_const_eval = UB_const_eval
        self.LB_const_eval = LB_const_eval
        self.UB_obj_eval = UB_obj_eval
        self.formulation1D()
        pass

    def getx(self):
        return self.opti.value(self.x)

    def getu(self):
        return self.opti.value(self.u)

    def setstartparam(self, loc):
        self.opti.set_value(self.p_start, loc.tolist())
        self.opti.set_initial(self.x[0], loc.tolist())

    def setendparam(self, loc):
        self.opti.set_value(self.p_end, loc.tolist())
        self.opti.set_initial(self.x[1:], loc.tolist())
        # init_rand = loc + (torch.rand(self.H) - 0.5)
        # self.opti.set_initial(self.x[1:], init_rand.reshape(-1, 1).numpy())

    def get_candidate(self):
        return self.opti.value(self.x)[self.Hm+1]

    def formulation1D(self):
        self.opti = ca.casadi.Opti()
        self.x = self.opti.variable(1, self.H+1)
        self.u = self.opti.variable(1, self.H)
        self.p_start = self.opti.parameter(1)
        self.p_end = self.opti.parameter(1)

        self.opti.subject_to(self.x[1:] == self.x[:-1] + self.u[:])
        self.opti.subject_to(self.u[0:] <= self.u_max)
        self.opti.subject_to(self.u[0:] >= self.u_min)
        self.opti.subject_to(self.x[1:] <= self.x_max)
        self.opti.subject_to(self.x[1:] >= self.x_min)
        # self.opti.subject_to(self.UB_const_eval.call(
        #     [self.x[k+1]])[0] - self.epsQ >= self.constraint)
        self.opti.subject_to(ca.fmax(self.LB_const_eval.call(
            [self.x[1:]])[0], self.UB_const_eval.call(
            [self.x[1:]])[0] - self.epsQ) >= self.constraint)

        # for k in range(0, self.H):
        #     self.opti.set_initial(self.x[k+1], self.p_start)
        #     self.opti.subject_to(self.x[k+1] == self.x[k] + self.u[k])
        #     self.opti.subject_to(self.u[k] <= self.u_max)
        #     self.opti.subject_to(self.u[k] >= self.u_min)
        #     self.opti.subject_to(self.x[k+1] <= self.x_max)
        #     self.opti.subject_to(self.x[k+1] >= self.x_min)
        #     # self.opti.subject_to(self.UB_const_eval.call(
        #     #     [self.x[k+1]])[0] - self.epsQ >= self.constraint)
        #     self.opti.subject_to(ca.fmax(self.LB_const_eval.call(
        #         [self.x[k+1]])[0], self.UB_const_eval.call(
        #         [self.x[k+1]])[0] - self.epsQ) >= self.constraint)

        self.opti.subject_to(self.x[0] == self.p_start)
        self.opti.subject_to(self.x[self.H] == self.p_end)
        # % And choose a concrete value for p, latter can be passed from another location
        self.opti.minimize(-10*ca.sumsqr(self.UB_obj_eval.call(
            [self.x[self.Hm+1]])[0]) + 0.001*ca.sumsqr(self.u))

    def solve(self):  # parameterize and then solve
        # p_opts = {"expand": True}
        # self.opti.print_options()
        s_opts = {"ipopt.max_iter":  10, "ipopt.tol": 1e-8, "ipopt.print_timing_statistics": 'yes',
                  "ipopt.derivative_test": "none", "ipopt.linear_solver": self.optim_param["linear_solver"]}
        self.opti.solver("ipopt", s_opts)
        # self.opti.solver("sqpmethod")
        try:
            sol = self.opti.solve()
        except:
            print("did not converge, let see .debug value")
            self.opti = self.opti.debug
        # print("solver status", self.opti.stats()['success'])
        # print("solver status", self.opti.stats())

    def print(self):
        print(self.opti)


class SEMPC(object):
    def __init__(self, optim_param, common_param, agent_param, LB_eval, UB_eval) -> None:
        self.optim_param = optim_param
        self.H = optim_param["H"]
        self.Hm = optim_param["Hm"]
        self.u_min = optim_param["u_min"]
        self.u_max = optim_param["u_max"]
        self.x_min = optim_param["x_min"]
        self.x_max = optim_param["x_max"]
        self.Lc = agent_param["Lc"]
        self.constraint = common_param["constraint"]
        self.epsQ = common_param["epsilon"]
        self.LB_eval = LB_eval
        self.UB_eval = UB_eval
        self.GPformulation1D()
        return None

    def update_const_func(self, LB, UB):
        self.LB = LB
        self.UB = UB

    def formulation(self, x_goal):
        self.opti = ca.casadi.Opti()
        self.x = self.opti.variable(2, self.H+1)
        self.u = self.opti.variable(2, self.H)
        self.z = self.opti.variable(2, 1)

        p = self.opti.parameter(2, 1)

        for k in range(0, self.H):
            self.opti.subject_to(self.x[:, k+1] == self.x[:, k] + self.u[:, k])
            # self.opti.subject_to(lb(self.x[k+1]) >= 2.25)
            self.opti.subject_to(self.u[:, k] <= self.u_max)
            self.opti.subject_to(self.u[:, k] >= self.u_min)
            self.opti.subject_to(self.x[:, k+1] <= self.x_max)
            self.opti.subject_to(self.x[:, k+1] >= self.x_min)
            # self.opti.subject_to(self.x[1, k+1] == -2)
        self.opti.subject_to(self.x[:, self.Hm] == self.z)
        self.opti.subject_to(self.x[:, 0] == self.p_start)
        self.opti.subject_to(self.x[:, self.H] == self.p_end)
        # % And choose a concrete value for p, latter can be passed from another location
        # self.opti.set_value(p, [0, -2])
        self.opti.minimize(
            ca.sumsqr(self.z-x_goal.numpy())*3 + 0.001*ca.sumsqr(self.u))

    def setparam(self, loc):
        self.p = loc.tolist()

    def setstartparam(self, loc):
        self.opti.set_value(self.p_start, loc.tolist())

    def setendparam(self, loc):
        self.opti.set_value(self.p_end, loc.tolist())

    def setgoalparam(self, loc):
        self.opti.set_value(self.x_goal_p, loc.tolist())
        self.opti.set_initial(self.z, [loc.numpy()])

    def getx(self):
        return self.opti.value(self.x)

    def getu(self):
        return self.opti.value(self.u)

    def getmin_constraint(self):
        return self.opti.value(self.u)

    def getz(self):
        return self.opti.value(self.z)

    def setwarmstartparam(self, x, u):
        self.warm_x = x
        self.warm_u = u
        self.opti.set_initial(self.x, self.warm_x)
        self.opti.set_initial(self.u, self.warm_u)

    def GPformulation1D(self):
        self.opti = ca.casadi.Opti()
        self.x = self.opti.variable(1, self.H+1)
        self.u = self.opti.variable(1, self.H)
        self.z = self.opti.variable(1, 1)
        self.x_goal_p = self.opti.parameter(1)
        self.p_start = self.opti.parameter(1)
        self.p_end = self.opti.parameter(1)

        self.opti.subject_to(self.x[1:] == self.x[:-1] + self.u[:])
        self.opti.subject_to(self.u[0:] <= self.u_max)
        self.opti.subject_to(self.u[0:] >= self.u_min)
        self.opti.subject_to(self.x[1:] <= self.x_max)
        self.opti.subject_to(self.x[1:] >= self.x_min)
        self.opti.subject_to(self.LB_eval.call(
            [self.x[1:]])[0] >= self.constraint)

        # for k in range(0, self.H):
        #     # self.opti.set_initial(self.x[k+1], self.p_start)
        #     self.opti.subject_to(self.x[k+1] == self.x[k] + self.u[k])
        #     self.opti.subject_to(self.u[k] <= self.u_max)
        #     self.opti.subject_to(self.u[k] >= self.u_min)
        #     self.opti.subject_to(self.x[k+1] <= self.x_max)
        #     self.opti.subject_to(self.x[k+1] >= self.x_min)
        #     self.opti.subject_to(self.LB_eval.call(
        #         [self.x[k+1]])[0] >= self.constraint)
        #     # self.opti.subject_to(self.UB_eval.call(
        #     #     [self.x[k+1]])[0] - self.epsQ >= self.constraint)

        # Informative points
        self.opti.subject_to(self.UB_eval.call(
            [self.x[self.Hm+1]])[0] - self.LB_eval.call([self.x[self.Hm+1]])[0] >= self.epsQ)
        # Expander condition
        self.opti.subject_to(self.UB_eval.call(
            [self.x[self.Hm+1]])[0] - ca.mtimes(ca.DM([self.Lc]), ca.norm_2(self.x[self.Hm+1]-self.z)) >= self.constraint)

        self.opti.subject_to(self.UB_eval.call([self.z])[
                             0] - self.epsQ >= self.constraint)
        self.opti.subject_to(self.z <= self.x_max)
        self.opti.subject_to(self.z >= self.x_min)
        self.opti.subject_to(self.LB_eval.call([self.z])[0] <= self.constraint)
        self.opti.subject_to(self.x[0] == self.p_start)
        self.opti.subject_to(self.x[self.H] == self.p_end)
        # % And choose a concrete value for p, latter can be passed from another location
        # self.opti.set_value(p, self.p)
        self.opti.minimize(
            10*ca.sumsqr(self.z-self.x_goal_p) + 0.0001*ca.sumsqr(self.u))

    def GPformulation(self, x_goal):
        self.opti = ca.casadi.Opti()
        self.x = self.opti.variable(2, self.H+1)
        self.u = self.opti.variable(2, self.H)
        self.z = self.opti.variable(2, 1)

        p = self.opti.parameter(2, 1)
        self.opti.set_initial(self.z, [0.3, -2])
        self.opti.set_initial(self.x[:, 0], self.p)
        for k in range(0, self.H):
            self.opti.set_initial(self.x[:, k+1], self.p)
            self.opti.subject_to(self.x[:, k+1] == self.x[:, k] + self.u[:, k])
            # self.opti.subject_to(lb(self.x[k+1]) >= 2.25)
            self.opti.subject_to(self.u[:, k] <= self.u_max)
            self.opti.subject_to(self.u[:, k] >= self.u_min)
            self.opti.subject_to(self.x[:, k+1] <= self.x_max)
            self.opti.subject_to(self.x[:, k+1] >= self.x_min)
            # self.opti.subject_to(self.LB_eval.call(
            #     [self.x[:, k+1]])[0] >= self.constraint)
            self.opti.subject_to(self.x[1, k+1] == -2.0)
        # self.opti.subject_to(self.UB_eval.call(
        #     [self.x[:, self.Hm+1]])[0] - self.LB_eval.call([self.x[:, self.Hm+1]])[0] >= 0.01)
        # self.opti.subject_to(self.UB_eval.call(
        #     [self.x[0, self.Hm+1]])[0] - self.Lc*ca.sqrt(ca.sumsqr(self.x[0, self.Hm+1]-self.z[0, :])) >= self.constraint)

        self.opti.subject_to(self.UB_eval.call([self.z[0, :]])[0] >= 0.01)
        # self.opti.subject_to(self.LB_eval.call([self.z])[0] <= self.constraint)
        self.opti.subject_to(self.x[:, 0] == p)
        self.opti.subject_to(self.x[:, self.H] == p)
        # % And choose a concrete value for p, latter can be passed from another location
        self.opti.set_value(p, self.p)
        self.opti.minimize(
            ca.sumsqr(self.z-x_goal.numpy()) + 0.0001*ca.sumsqr(self.u))

    def solve(self):  # parameterize and then solve
        # https://or.stackexchange.com/questions/2669/ipopt-with-hsl-vs-mumps
        # for feasibility: https://coin-or.github.io/Ipopt/OPTIONS.html
        # p_opts = {"expand": True}
        # self.opti.print_options()
        s_opts = {"ipopt.max_iter":  15, "ipopt.tol": 1e-8,
                  "ipopt.derivative_test": "none", "ipopt.linear_solver": self.optim_param["linear_solver"],
                  "ipopt.inf_pr_output": "original"}
        self.opti.solver("ipopt", s_opts)
        # self.opti.solver("sqpmethod")
        try:
            sol = self.opti.solve()
        except:
            print("did not converge, let see .debug value")
            self.opti = self.opti.debug
        a = 1

    def print(self):
        print(self.opti)


class PytorchGPEvaluator(ca.Callback):
    def __init__(self,  t_out, func, get_sparsity_out, opts={}):
        """
          t_in: list of inputs (pytorch tensors)
          t_out: list of outputs (pytorch tensors)
        """
        ca.casadi.Callback.__init__(self)
        self.t_out = t_out  # it is a function
        self.func = func
        self.get_sparsity_out = get_sparsity_out
        self.construct("PytorchGPEvaluator", opts)
        self.refs = []

    def update_func(self, function):
        self.func = function
        self.t_out = function  # it is a function

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(1, 2)

    def eval(self, x):
        print('in eval', x)
        print('I am returning', [self.t_out(torch.from_numpy(
            x[0].toarray()).reshape(-1, 2)).detach().numpy()])
        # return [self.t_out(x).detach().numpy()]
        return [self.t_out(torch.from_numpy(x[0].toarray()).reshape(-1, 2)).detach().numpy()]

    def gradjac(self, x):
        jacob = torch.autograd.functional.jacobian(
            self.func, x)
        return jacob.reshape(1, 2).detach()

    def gradhes(self, x):
        jacob = torch.autograd.functional.hessian(
            self.func, x)
        return jacob.reshape(-1, 2).detach()

    def jacob_sparsity_out(self, i):

        return ca.Sparsity.dense(1, 2)

    def hessian_sparsity_out(self, i):

        return ca.Sparsity.dense(2, 2)

    def has_jacobian(self): return True

    def get_jacobian(self, name, inames, onames, opts):
        callback = PytorchGPEvaluator(
            self.gradjac, self.func, self.jacob_sparsity_out)
        callback.jacob_sparsity_out = self.hessian_sparsity_out
        callback.gradjac = self.gradhes

        # Make sure you keep a reference to it
        self.refs.append(callback)

        # Package it in the nominal_in+nominal_out form that Cas.casadiexpects
        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        print(name, nominal_in+nominal_out,
              callback.call(nominal_in), inames, onames, opts)
        return ca.Function(name, nominal_in+nominal_out, callback.call(nominal_in), inames, onames)


class PytorchGPEvaluator1D(ca.Callback):
    def __init__(self,  t_out, func, get_sparsity_out, opts={}):
        """
          t_in: list of inputs (pytorch tensors)
          t_out: list of outputs (pytorch tensors)
        """
        ca.casadi.Callback.__init__(self)
        self.t_out = t_out  # it is a function
        self.func = func
        self.get_sparsity_out = get_sparsity_out
        self.construct("PytorchGPEvaluator1D", opts)
        self.refs = []

    def update_func(self, function):
        self.func = function
        self.t_out = function  # it is a function

    def get_n_in(self):
        return 1

    def get_n_out(self):
        return 1

    def get_sparsity_in(self, i):
        return ca.Sparsity.dense(1, 1)

    def eval(self, x):
        y = torch.Tensor(x[0].toarray().tolist()[0] + [-2.0])
        # print("eval", y, [self.t_out(y.reshape(-1, 2)).detach().numpy()])
        return [self.t_out(y.reshape(-1, 2)).detach().numpy()]

    # def eval(self, x):
    #     y = torch.Tensor(x[0].toarray().tolist()[0])
    #     return [self.t_out(y.reshape(-1, 1)).detach().numpy()]

    def gradjac(self, x):
        jacob = torch.autograd.functional.jacobian(
            self.func, x)[0][0]
        return jacob.reshape(-1).detach()

    def gradhes(self, x):
        jacob = torch.autograd.functional.hessian(
            self.func, x)[0][0][0][0]
        return jacob.reshape(-1).detach()

    def jacob_sparsity_out(self, i):

        return ca.Sparsity.dense(1, 1)

    def hessian_sparsity_out(self, i):

        return ca.Sparsity.dense(1, 1)

    def has_jacobian(self): return True

    def get_jacobian(self, name, inames, onames, opts):
        # In the callback of jacobian we change to hessian functions
        callback = PytorchGPEvaluator1D(
            self.gradjac, self.func, self.jacob_sparsity_out)
        callback.jacob_sparsity_out = self.hessian_sparsity_out
        callback.gradjac = self.gradhes

        # Make sure you keep a reference to it
        self.refs.append(callback)

        # Package it in the nominal_in+nominal_out form that Cas.casadiexpects
        nominal_in = self.mx_in()
        nominal_out = self.mx_out()
        return ca.Function(name, nominal_in+nominal_out, callback.call(nominal_in), inames, onames)
