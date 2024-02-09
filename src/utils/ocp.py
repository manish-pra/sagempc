import casadi as ca
import numpy as np
from acados_template import AcadosOcp
from acados_template.acados_sim import AcadosSim
from src.utils.model import (export_integrator_model,
                             export_n_integrator_model,
                             export_pendulum_ode_model_with_discrete_rk4,
                             export_pendulum_ode_model_with_discrete_rk4_Lc,
                             export_NH_integrator_ode_model_with_discrete_rk4,
                             export_robot_model_with_discrete_rk4,
                             export_bicycle_model_with_discrete_rk4,
                             export_unicycle_model_with_discrete_rk4,
                             export_unicycle_model_with_discrete_rk4_LC)


def oracle_const_expr(model, x_dim, n_order, params, model_x):
    ub_cx_lin = ca.SX.sym('ub_cx_lin')
    ub_cx_grad = ca.SX.sym('ub_cx_grad', x_dim)
    x_lin = ca.SX.sym('x_lin', n_order*x_dim)
    ub_fx_lin = ca.SX.sym('ub_fx_lin')
    ub_fx_grad = ca.SX.sym('ub_fx_grad', x_dim)
    x_terminal = ca.SX.sym('x_terminal', n_order*x_dim)
    xg = ca.SX.sym('xg', x_dim)
    w = ca.SX.sym('w', 1, 1)
    # eps_lin = ca.SX.sym('eps_lin')
    # eps = ca.SX.sym('eps')
    # p_lin = ca.vertcat(ub_cx_lin, ub_cx_grad, x_lin, ub_fx_lin, ub_fx_grad, eps_lin)
    p_lin = ca.vertcat(ub_cx_lin, ub_cx_grad, x_lin, x_terminal, xg, w)
    # eps = params["common"]["epsilon"]
    q_th = params["common"]["constraint"]
    model.con_h_expr = ca.vertcat(
        ub_cx_lin + ub_cx_grad.T @ (model_x-x_lin)[:x_dim] - q_th)
    model.con_h_expr_e = ca.vertcat(
        ub_cx_lin + ub_cx_grad.T @ (model_x-x_lin)[:x_dim] - q_th)
    #     ub_cx_lin + ub_cx_grad.T @ (model_x-x_lin)[:x_dim] - q_th)
    # model.con_h_expr = ca.vertcat(
    #     h_lin + h_grad @ (model_x-x_lin), ub_fx_lin - eps_lin + ub_fx_grad @ (model_x-x_lin) - (eps - eps_lin))
    # model.con_h_expr_e = ca.vertcat(
    #     h_lin + h_grad @ (model_x-x_lin), ub_fx_lin - eps_lin + ub_fx_grad @ (model_x-x_lin) - (eps - eps_lin))
    model.p = p_lin
    return model, xg, w


def oracle_cost_expr(ocp, model_x, model_u, x_dim, w, xg):
    # cost
    q = 1e-3*np.diag(np.ones(x_dim))
    qx = np.diag(np.ones(x_dim))
    # cost
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost =  w * \
        (model_x[:x_dim] - xg).T @ qx @ (model_x[:x_dim] - xg) + model_u.T @ (q) @ model_u 
    ocp.model.cost_expr_ext_cost_e =  w * (model_x[:x_dim] - xg).T @ qx @ (model_x[:x_dim] - xg)

    # q = -1e-5*np.diag(np.ones(x_dim))
    # xg_discrete = np.array([-1.8, 0.5])
    # ocp.cost.cost_type = 'NONLINEAR_LS'
    # ocp.cost.cost_type_e = 'NONLINEAR_LS'
    # ocp.cost.W_e = np.diag(1*np.ones(x_dim))
    # ocp.cost.W = np.diag(
    #     np.hstack([1*np.ones(x_dim), 1e-5*np.ones(x_dim)]))
    # ocp.model.cost_y_expr = ca.vertcat(w*(model_x[:x_dim]-xg_discrete), model_u)
    # ocp.model.cost_y_expr_e = w*(model_x[:x_dim]-xg_discrete)
    # ocp.cost.yref = np.zeros(2*x_dim)
    # ocp.cost.yref_e = np.zeros(1*x_dim)

    # TO DO for later
    # ocp.cost.cost_type = 'EXTERNAL'
    # ocp.cost.cost_type_e = 'EXTERNAL'
    # # ocp.model.cost_expr_ext_cost = -ub_fx_grad.T @ (
    # #     model_x-x_lin)[:x_dim] + model_u.T @ (-q) @ model_u + (model_x[0]).T @ 0.001 @ (model_x[0])
    # # ocp.model.cost_expr_ext_cost_e = -ub_fx_grad.T @ (
    # #     model_x-x_lin)[:x_dim]
    # # cg = 2*np.ones(x_dim)
    # xg_discrete = np.array([0.5, 0.5])*0.0
    # ocp.model.cost_expr_ext_cost = model_u.T @ (-q) @ model_u + (
    #     model_x[:x_dim]-xg_discrete).T @ 100 @ (model_x[:x_dim]-xg_discrete) #-1e-3*model.u[-1]**2
    # ocp.model.cost_expr_ext_cost_e = (
    #     model_x[:x_dim]-xg_discrete).T @ 100 @ (model_x[:x_dim]-xg_discrete)

    # ocp.model.cost_expr_ext_cost = (
    #     model_x[0]).T @ 0.001 @ (model_x[0]) + model_u.T @ (-q) @ model_u
    # ocp.model.cost_expr_ext_cost_e = (
    #     model_x[0]).T @ 0.001 @ (model_x[0])

    # ocp.model.cost_expr_ext_cost = -eps**2 + model_u.T @ (-q) @ model_u
    # ocp.model.cost_expr_ext_cost_e = -eps**2
    return ocp


def oracle_const_val(ocp, params, x_dim, n_order):
    # ocp.constraints.set('eps', np.array([0]))
    # constraints
    ocp.constraints.lbu = params["optimizer"]["u_min"]*np.ones(x_dim)
    ocp.constraints.ubu = params["optimizer"]["u_max"]*np.ones(x_dim)
    ocp.constraints.idxbu = np.zeros((x_dim,))

    x0 = np.zeros(n_order*x_dim+1) 
    x0[:x_dim] = np.array(params["env"]["start_loc"])
    ocp.constraints.x0 = x0.copy() # this is always required to set the dimension of x_0, rest you can set the constraint with lbx and ubx

    lbx = params["optimizer"]["u_min"]*np.ones(n_order*x_dim)
    lbx[:x_dim] = params["optimizer"]["x_min"]*np.ones(x_dim)
    ocp.constraints.lbx_e = lbx.copy()

    ubx = params["optimizer"]["u_max"]*np.ones(n_order*x_dim)
    ubx[:x_dim] = params["optimizer"]["x_max"]*np.ones(x_dim)
    ocp.constraints.ubx_e = ubx.copy()
    ocp.constraints.idxbx_e = np.arange(ocp.model.x.shape[0]-1)

    ocp.constraints.lbx = lbx.copy()
    ocp.constraints.ubx = ubx.copy()
    ocp.constraints.idxbx = np.arange(ocp.model.x.shape[0]-1)

    ocp.constraints.lh = np.array([0])
    ocp.constraints.uh = np.array([10.0])
    ocp.constraints.lh_e = np.array([0])
    ocp.constraints.uh_e = np.array([10.0])

    ocp.parameter_values = np.zeros((ocp.model.p.shape[0],))
    return ocp


def concat_const_val(ocp, params):
    x_dim = params["optimizer"]["x_dim"]
    if params["algo"]["type"] == "ret_expander" or params["algo"]["type"] == "MPC_expander":
        lbx = np.array(params["optimizer"]["x_min"])[:x_dim]
        ubx = np.array(params["optimizer"]["x_max"])[:x_dim]
        ocp.constraints.lbu = np.concatenate(
            [ocp.constraints.lbu, np.array([params["optimizer"]["dt"]]), lbx])
        ocp.constraints.ubu = np.concatenate(
            [ocp.constraints.ubu, np.array([1.0]), ubx])
        ocp.constraints.idxbu = np.arange(ocp.constraints.idxbu.shape[0] + 1 + x_dim)
    else:
        ocp.constraints.lbu = np.concatenate(
            [ocp.constraints.lbu, np.array([params["optimizer"]["dt"]])])
        ocp.constraints.ubu = np.concatenate(
            [ocp.constraints.ubu, np.array([1.0])])
        ocp.constraints.idxbu = np.concatenate([ocp.constraints.idxbu, np.array([ocp.model.u.shape[0]-1])])

    # ocp.constraints.x0 = np.concatenate(
    #     [ocp.constraints.x0, np.array([0.0])])

    ocp.constraints.lbx_e = np.concatenate(
        [ocp.constraints.lbx_e, np.array([1.0])])
    ocp.constraints.ubx_e = np.concatenate(
        [ocp.constraints.ubx_e, np.array([1.0])])
    ocp.constraints.idxbx_e = np.concatenate([ocp.constraints.idxbx_e, np.array([ocp.model.x.shape[0]-1])])

    ocp.constraints.lbx = np.concatenate([ocp.constraints.lbx, np.array([0])])
    ocp.constraints.ubx = np.concatenate([ocp.constraints.ubx, np.array([2])])
    ocp.constraints.idxbx = np.concatenate([ocp.constraints.idxbx, np.array([ocp.model.x.shape[0]-1])])
    return ocp


def oracle_set_options(ocp, params):
    # discretization
    ocp.dims.N = params["optimizer"]["H"]
    ocp.solver_options.tf = params["optimizer"]["Tf"]
    ocp.solver_options.qp_solver_warm_start = 1
    # set options
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'  # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = 'DISCRETE'  # IRK
    # ocp.solver_options.print_level = 1
    ocp.solver_options.levenberg_marquardt = 1.0e-1
    ocp.solver_options.nlp_solver_ext_qp_res = 1
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # SQP_RTI, SQP
    # ocp.solver_options.qp_solver_warm_start = 1
    return ocp


def export_oracle_ocp(params):
    ocp = AcadosOcp()
    n_order = params["optimizer"]["order"]
    x_dim = params["optimizer"]["x_dim"]
    # model = export_integrator_model('oracle')
    # model = export_n_integrator_model('oracle', n_order, x_dim)
    if params["algo"]["type"] == "ret_expander" or params["algo"]["type"] == "MPC_expander":
        model = export_pendulum_ode_model_with_discrete_rk4_Lc('oracle', n_order, x_dim)
    else:
        model = export_pendulum_ode_model_with_discrete_rk4('oracle', n_order, x_dim)
    model_u = model.u[:x_dim]
    model_x = model.x[:-1]

    model, xg, w = oracle_const_expr(model, x_dim, n_order, params, model_x) 

    ocp.model = model

    ocp = oracle_cost_expr(ocp, model_x, model_u, x_dim, w, xg)

    ocp = oracle_const_val(ocp, params, x_dim, n_order)

    ocp = concat_const_val(ocp,params)

    ocp = oracle_set_options(ocp, params)

    return ocp

def sempc_const_expr(model, x_dim, n_order, params, model_x, model_z):
    lb_cx_lin = ca.SX.sym('lb_cx_lin')
    lb_cx_grad = ca.SX.sym('lb_cx_grad', x_dim, 1)
    ub_cx_lin = ca.SX.sym('lb_cx_lin')
    ub_cx_grad = ca.SX.sym('lb_cx_grad', x_dim, 1)
    if params["agent"]["dynamics"] == "robot": # this is misleading, it is actually the number of states
        x_lin = ca.SX.sym('x_lin', n_order*x_dim+1)
        x_terminal = ca.SX.sym('x_terminal', n_order*x_dim+1)
    else:
        x_lin = ca.SX.sym('x_lin', n_order*x_dim)
        x_terminal = ca.SX.sym('x_terminal', n_order*x_dim)
    xg = ca.SX.sym('xg', x_dim)
    w = ca.SX.sym('w', 1, 1)
    we = ca.SX.sym('we', 1, 1)
    cw = ca.SX.sym('cw', 1, 1)
    
    q_th = params["common"]["constraint"]
    var = (ub_cx_lin + ub_cx_grad.T @ (model_x-x_lin)
            [:x_dim] - (lb_cx_lin + lb_cx_grad.T @ (model_x-x_lin)[:x_dim]))
    if params["algo"]["type"] == "ret_expander" or params["algo"]["type"] == "MPC_expander":
        lb_cz_lin = ca.SX.sym('lb_cz_lin')
        lb_cz_grad = ca.SX.sym('lb_cz_grad', x_dim, 1)
        z_lin = ca.SX.sym('z_lin', x_dim)
        p_lin = ca.vertcat(lb_cx_lin, lb_cx_grad, x_lin, xg, w,
                       x_terminal, ub_cx_lin, ub_cx_grad, cw, z_lin, lb_cz_lin, lb_cz_grad)
        Lc = params["common"]["Lc"]

        # expanders
        # l_n(z) \leq 0 -lb_cz_lin - lb_cz_grad.T @ (model_z-z_lin), # lets remove this constraint for easiness
        # u_n(x) - Lc (x-z)^2 \geq 0 
        # ub_cx_lin + ub_cx_grad.T @ (model_x-x_lin) - 2*Lc*(x_lin[:x_dim] - z_lin).T@(model_x-x_lin)[:x_dim] - 2*Lc*(z_lin-x_lin[:x_dim]).T@(model_z-z_lin) - Lc*(x_lin[:x_dim] - z_lin)**2
        
        # pessimistic set and uncertainity
        # l_n(z) - Lc sqrt((x1-z1)^2 + (x1-z1)^2) \leq 0, w(x) \geq \epsilon
        tol = 1.0e-3
        model.con_h_expr = ca.vertcat(lb_cz_lin + lb_cz_grad.T @ (model_z-z_lin) - (Lc/(ca.norm_2(x_lin[:x_dim] - z_lin)+tol))*((x_lin[:x_dim] - z_lin).T@(model_x-x_lin)[:x_dim]) 
                                      - (Lc/(ca.norm_2(x_lin[:x_dim] - z_lin)+tol))*((z_lin-x_lin[:x_dim]).T@(model_z-z_lin)) 
                                      - Lc*ca.norm_2(x_lin[:x_dim] - z_lin) - q_th,  cw*var, w*(lb_cx_lin + lb_cx_grad.T @ (model_x-x_lin)[:x_dim]))
        # model.con_h_expr = ca.vertcat(lb_cz_lin + lb_cz_grad.T @ (model_z-z_lin) - Lc*(ca.sign(x_lin[:x_dim]-z_lin).T@(model_x-x_lin)[:x_dim])
        #                         - Lc*(ca.sign(z_lin-x_lin[:x_dim]).T@(model_z-z_lin))
        #                         - Lc*ca.norm_1(x_lin[:x_dim] - z_lin) - q_th, cw*var, w*(lb_cx_lin + lb_cx_grad.T @ (model_x-x_lin)[:x_dim]))
        # model.con_h_expr = ca.vertcat(lb_cz_lin + lb_cz_grad.T @ (model_z-z_lin) - Lc*(ca.sign(x_lin[:x_dim]-z_lin + tol).T@(model_x-x_lin)[:x_dim])
        #                 - Lc*(ca.sign(z_lin-x_lin[:x_dim] + tol).T@(model_z-z_lin)) - Lc*ca.norm_1(x_lin[:x_dim] - z_lin + tol) - q_th,  cw*var)
        # model.con_h_expr = ca.vertcat(lb_cx_lin +
        #                                 lb_cx_grad.T @ (model_x-x_lin)[:x_dim] - q_th,cw*var,1)
        # model.con_h_expr = ca.vertcat(lb_cz_lin + lb_cz_grad.T @ (model_z-z_lin) - Lc*(ca.sign(x_lin[:x_dim]-z_lin).T@(model_x-x_lin)[:x_dim])
        #                 - Lc*(ca.sign(z_lin-x_lin[:x_dim]).T@(model_z-z_lin)) - Lc*ca.norm_1(x_lin[:x_dim] - z_lin) - q_th,
        #                   lb_cz_lin + lb_cz_grad.T @ (model_z-z_lin),
        #                   ub_cx_lin + ub_cx_grad.T @ (model_x-x_lin)[:x_dim] - Lc*ca.norm_1(x_lin[:x_dim] - z_lin) - Lc*(ca.sign(x_lin[:x_dim]-z_lin).T@(model_x-x_lin)[:x_dim])
        #                 - Lc*(ca.sign(z_lin-x_lin[:x_dim]).T@(model_z-z_lin)), cw*var)
        # Since the variable z is actually a u, we cannot have a terminal constraint on u for H+1
        model.con_h_expr_e = ca.vertcat(lb_cx_lin +
                                        lb_cx_grad.T @ (model_x-x_lin)[:x_dim] - q_th)
    elif params["algo"]["type"] == "MPC_Xn":
        p_lin = ca.vertcat(lb_cx_lin, lb_cx_grad, x_lin, xg, w,
                    x_terminal, ub_cx_lin, ub_cx_grad, cw,we)
        model.con_h_expr = ca.vertcat(lb_cx_lin +
                                    lb_cx_grad.T @ (model_x-x_lin)[:x_dim] - q_th, cw*var, we*(model_x - model.f_expl_expr[:-1]))
        model.con_h_expr_e = ca.vertcat(lb_cx_lin +
                                        lb_cx_grad.T @ (model_x-x_lin)[:x_dim] - q_th)
    else:
        p_lin = ca.vertcat(lb_cx_lin, lb_cx_grad, x_lin, xg, w,
                    x_terminal, ub_cx_lin, ub_cx_grad, cw)
        model.con_h_expr = ca.vertcat(lb_cx_lin +
                                    lb_cx_grad.T @ (model_x-x_lin)[:x_dim] - q_th, cw*var)
        model.con_h_expr_e = ca.vertcat(lb_cx_lin +
                                        lb_cx_grad.T @ (model_x-x_lin)[:x_dim] - q_th)
        # model.con_h_expr_e = ca.vertcat(model_x - model.disc_dyn_expr[:-1])
        # model.con_h_expr_e = ca.vertcat(model_x - model.f_expl_expr[:-1])
    # model.con_h_expr = ca.vertcat(lb_cx_lin +
    #                               lb_cx_grad.T @ (model_x-x_lin)[:x_dim] - q_th)
    # model.con_h_expr_e = ca.vertcat(lb_cx_lin +
    #                                 lb_cx_grad.T @ (model_x-x_lin)[:x_dim] - q_th)
    model.p = p_lin
    return model, w, xg, var


def sempc_cost_expr(ocp, model_x, model_u, x_dim, w, xg, var, params):
    q = 1e-3*np.diag(np.ones(x_dim))
    qx = np.diag(np.ones(x_dim))
    # cost
    ocp.cost.cost_type = 'EXTERNAL'
    ocp.cost.cost_type_e = 'EXTERNAL'
    ocp.model.cost_expr_ext_cost =  w * \
        (model_x[:x_dim] - xg).T @ qx @ (model_x[:x_dim] - xg) + model_u.T @ (q) @ model_u + ocp.model.x[-1]*w/1000
    ocp.model.cost_expr_ext_cost_e =  w * (model_x[:x_dim] - xg).T @ qx @ (model_x[:x_dim] - xg)

    if params["algo"]["type"] == "ret_expander" or params["algo"]["type"] == "MPC_expander":
        ocp.constraints.idxsh = np.array([1,2])
        ocp.cost.zl = 1e2 * np.array([1,1])
        ocp.cost.zu = 1e1 * np.array([1,0.1])
        ocp.cost.Zl = 1e1 * np.array([[1,0],[0,1]])
        ocp.cost.Zu = 1e1 * np.array([[1,0],[0,1]])
    else:
        ocp.constraints.idxsh = np.array([1])
        ocp.cost.zl = 1e2 * np.array([1])
        ocp.cost.zu = 1e1 * np.array([1])
        ocp.cost.Zl = 1e1 * np.array([1])
        ocp.cost.Zu = 1e1 * np.array([1])

    # ocp.cost.cost_type = 'NONLINEAR_LS'
    # ocp.cost.cost_type_e = 'NONLINEAR_LS'
    # ocp.cost.W_e = np.diag(1*np.ones(x_dim))
    # ocp.cost.W = np.diag(
    #     np.hstack([1*np.ide(x_dim), 1e-3*np.ones(x_dim), 1e-4]))
    # ocp.model.cost_y_expr = ca.vertcat(
    #     w*(model_x[:x_dim] - xg), model_u, w*var)
    # ocp.model.cost_y_expr_e = w*(model_x[:x_dim] - xg)
    # yref = np.zeros(2*x_dim+1)
    # ocp.cost.yref = yref
    # ocp.cost.yref_e = np.zeros(1*x_dim)
    return ocp

def sempc_const_val(ocp, params, x_dim, n_order):
    # constraints
    eps = params["common"]["epsilon"] #- 0.05
    if params["agent"]["dynamics"] == "bicycle":
        ocp.constraints.lbu = np.array(params["optimizer"]["u_min"])
        ocp.constraints.ubu = np.array(params["optimizer"]["u_max"])
        ocp.constraints.idxbu = np.arange(x_dim)

        lbx = np.array(params["optimizer"]["x_min"])
        ubx = np.array(params["optimizer"]["x_max"])
        # lbx = np.concatenate([lbx, np.array([-1.0e2, -1.0e2])])
        # ubx = np.concatenate([ubx, np.array([1.0e2, 1.0e2])])
    else:
        ocp.constraints.lbu = np.array(params["optimizer"]["u_min"])
        ocp.constraints.ubu = np.array(params["optimizer"]["u_max"])
        ocp.constraints.idxbu = np.arange(x_dim)

        lbx = np.array(params["optimizer"]["x_min"])
        ubx = np.array(params["optimizer"]["x_max"])

        # lbx = params["optimizer"]["u_min"][0]*np.ones(n_order*x_dim)
        # lbx[:x_dim] = params["optimizer"]["x_min"]*np.ones(x_dim)

        # ubx = params["optimizer"]["u_max"][0]*np.ones(n_order*x_dim)
        # ubx[:x_dim] = params["optimizer"]["x_max"]*np.ones(x_dim)
    
    x0 = np.zeros(ocp.model.x.shape[0])
    x0[:x_dim] = np.array(params["env"]["start_loc"]) #np.ones(x_dim)*0.72
    ocp.constraints.x0 = x0.copy()


    ocp.constraints.lbx_e = lbx.copy()
    ocp.constraints.ubx_e = ubx.copy()
    ocp.constraints.idxbx_e = np.arange(lbx.shape[0])
    
    ocp.constraints.lbx = lbx.copy()
    ocp.constraints.ubx = ubx.copy()
    ocp.constraints.idxbx = np.arange(lbx.shape[0])
    if params["algo"]["type"] == "MPC_Xn":
        wee = 1.0e-5
        ocp.constraints.lh = np.array([0,  eps, -1*wee, -1*wee, -1*wee, -1*wee])
        ocp.constraints.uh = np.array([10.0, 1e8, wee, wee, wee, wee])
    elif params["algo"]["type"] == "ret_expander" or params["algo"]["type"] == "MPC_expander":
        ocp.constraints.lh = np.array([0, eps, -1e8])
        ocp.constraints.uh = np.array([10.0, 1e8, 0.2])
    else:
        ocp.constraints.lh = np.array([0, eps])
        ocp.constraints.uh = np.array([10.0, 1e8])
    ocp.constraints.lh_e = np.array([0.0])
    ocp.constraints.uh_e = np.array([10.0])

    # ocp.constraints.lh = np.array([0, eps])
    # ocp.constraints.uh = np.array([10.0, 1.0e9])
    # lh_e = np.zeros(n_order*x_dim+1)
    # lh_e[0] = 0
    # ocp.constraints.lh_e = lh_e
    # uh_e = np.zeros(n_order*x_dim+1)
    # uh_e[0] = 10
    # ocp.constraints.uh_e = uh_e

    ocp.parameter_values = np.zeros((ocp.model.p.shape[0],))
    return ocp

def sempc_set_options(ocp, params):
    # discretization
    ocp.dims.N = params["optimizer"]["H"]
    ocp.solver_options.tf = params["optimizer"]["Tf"]


    ocp.solver_options.qp_solver_warm_start = 1
    # set options
    ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'  # FULL_CONDENSING_QPOASES
    # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
    # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.levenberg_marquardt = 1.0e-1
    ocp.solver_options.integrator_type = 'DISCRETE' #'IRK'  # IRK
    # ocp.solver_options.print_level = 1
    ocp.solver_options.nlp_solver_ext_qp_res = 1
    ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # SQP_RTI, SQP
    # ocp.solver_options.tol = 1e-6
    # ocp.solver_options.regularize_method = 'CONVEXIFY'
    # ocp.solver_options.globalization = 'FIXED_STEP'
    # ocp.solver_options.alpha_min = 1e-2
    # ocp.solver_options.__initialize_t_slacks = 0
    # ocp.solver_options.regularize_method = 'CONVEXIFY'
    # ocp.solver_options.levenberg_marquardt = 1e-1
    # ocp.solver_options.print_level = 2
    # ocp.solver_options.qp_solver_iter_max = 400
    # ocp.solver_options.regularize_method = 'MIRROR'
    # ocp.solver_options.exact_hess_constr = 0
    # ocp.solver_options.line_search_use_sufficient_descent = line_search_use_sufficient_descent
    # ocp.solver_options.globalization_use_SOC = globalization_use_SOC
    # ocp.solver_options.eps_sufficient_descent = 5e-1
        # params = {'globalization': ['MERIT_BACKTRACKING', 'FIXED_STEP'],
        #       'line_search_use_sufficient_descent': [0, 1],
        #       'globalization_use_SOC': [0, 1]}
    return ocp

def export_sempc_ocp(params):
    ocp = AcadosOcp()
    name_prefix = params["algo"]["type"] + '_env_' + str(params["env"]["name"]) + '_i_' + str(params["env"]["i"]) + '_'
    n_order = params["optimizer"]["order"]
    x_dim = params["optimizer"]["x_dim"]
    # model = export_integrator_model('sempc')
    # model = export_n_integrator_model('sempc', n_order, x_dim)
    if params["algo"]["type"] == "ret_expander" or params["algo"]["type"] == "MPC_expander":
        if params["agent"]["dynamics"] == "unicycle":
            model = export_unicycle_model_with_discrete_rk4_LC(name_prefix + 'sempc')
        else:
            model = export_pendulum_ode_model_with_discrete_rk4_Lc(name_prefix + 'sempc', n_order, x_dim)
    else:
        if params["agent"]["dynamics"] == "NH_integrator":
            model = export_NH_integrator_ode_model_with_discrete_rk4(name_prefix + 'sempc')
        elif params["agent"]["dynamics"] == "unicycle":
            model = export_unicycle_model_with_discrete_rk4(name_prefix + 'sempc')
        elif params["agent"]["dynamics"] == "robot":
            model = export_robot_model_with_discrete_rk4(name_prefix + 'sempc')
        elif params["agent"]["dynamics"] == "bicycle":
            model = export_bicycle_model_with_discrete_rk4(name_prefix + 'sempc')
        else:
            model = export_pendulum_ode_model_with_discrete_rk4(name_prefix + 'sempc', n_order, x_dim)
    model_u = model.u[:x_dim]
    model_x = model.x[:-1]
    model_z = model.u[-x_dim:]

    model, w, xg, var = sempc_const_expr(model, x_dim, n_order, params, model_x, model_z)

    ocp.model = model

    ocp = sempc_cost_expr(ocp, model_x, model_u, x_dim, w, xg, var, params)

    ocp = sempc_const_val(ocp, params, x_dim, n_order)

    ocp = concat_const_val(ocp,params)

    ocp = sempc_set_options(ocp, params)
    
    return ocp


def export_sim(params, name):
    n_order = params["optimizer"]["order"]
    x_dim = params["optimizer"]["x_dim"]
    sim = AcadosSim()
    # model = export_integrator_model(name)
    # model = export_n_integrator_model(name, n_order, x_dim)
    model = export_pendulum_ode_model_with_discrete_rk4(
        'oracle_sim', n_order, x_dim)
    h_lin = ca.SX.sym('h_lin')
    h_grad = ca.SX.sym('h_grad')
    x_lin = ca.SX.sym('x_lin')
    p_lin = ca.vertcat(h_lin, h_grad, x_lin)
    model.con_h_expr = h_lin + h_grad @ (model.x-x_lin)
    model.con_h_expr_e = h_lin + h_grad @ (model.x-x_lin)
    model.p = p_lin
    sim.model = model
    # solver options
    # sim.solver_options.integrator_type = 'IRK'
    sim.solver_options.integrator_type = 'ERK'
    sim.parameter_values = np.zeros(model.p.shape)
    # sim.solver_options.sim_method_num_stages = 2
    # sim.solver_options.sim_method_num_steps = 1  # 1
    # sim.solver_options.nlp_solver_tol_eq = 1e-9

    # set prediction horizon
    sim.solver_options.Tsim = params["optimizer"]["Tf"] / \
        params["optimizer"]["H"]
    return sim


# def export_ocp(params):
#     ocp = AcadosOcp()
#     model = export_integrator_model()
#     h_lin = ca.SX.sym('h_lin')
#     h_grad = ca.SX.sym('h_grad')
#     x_lin = ca.SX.sym('x_lin')
#     p_lin = ca.vertcat(h_lin, h_grad, x_lin)
#     model.con_h_expr = h_lin + h_grad @ (model.x-x_lin)
#     model.con_h_expr_e = h_lin + h_grad @ (model.x-x_lin)
#     model.p = p_lin
#     ocp.model = model

#     # discretization
#     ocp.dims.N = params["optimizer"]["H"]
#     ocp.solver_options.tf = params["optimizer"]["Tf"]
#     q = np.array([-1.0])
#     # cost
#     ocp.cost.cost_type = 'EXTERNAL'
#     ocp.cost.cost_type_e = 'EXTERNAL'
#     ocp.model.cost_expr_ext_cost = model.x.T @ q @ model.x + \
#         model.u.T @ (-q) @ model.u
#     ocp.model.cost_expr_ext_cost_e = model.x.T @ q @ model.x

#     # constraints
#     ocp.constraints.lbu = np.array([-1.0])
#     ocp.constraints.ubu = np.array([1.0])
#     ocp.constraints.idxbu = np.array([0])
#     ocp.constraints.x0 = np.array([0.40])
#     ocp.constraints.lbx_e = np.array([0.40])
#     ocp.constraints.ubx_e = np.array([0.40])
#     ocp.constraints.idxbx_e = np.array([0])
#     # ocp.model.con_h_expr = np.array([1.05]) - (model.x-np.array([4.0]))**2
#     # ocp.model.con_h_expr_e = np.array([1.05]) - (model.x-np.array([4.0]))**2
#     ocp.constraints.lh = np.array([-1])
#     ocp.constraints.uh = np.array([10.0])
#     ocp.constraints.lh_e = np.array([-1])
#     ocp.constraints.uh_e = np.array([10.0])
#     ocp.parameter_values = np.zeros((ocp.model.p.shape[0],))
#     # set options
#     ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'  # FULL_CONDENSING_QPOASES
#     # PARTIAL_CONDENSING_HPIPM, FULL_CONDENSING_QPOASES, FULL_CONDENSING_HPIPM,
#     # PARTIAL_CONDENSING_QPDUNES, PARTIAL_CONDENSING_OSQP, FULL_CONDENSING_DAQP
#     ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'  # 'GAUSS_NEWTON', 'EXACT'
#     ocp.solver_options.integrator_type = 'IRK'  # IRK
#     # ocp.solver_options.print_level = 1
#     ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # SQP_RTI, SQP
#     return ocp
