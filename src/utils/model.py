import casadi as ca
import numpy as np
from acados_template import AcadosModel


def export_integrator_model(name):
    model = AcadosModel()
    x = ca.SX.sym('x')
    x_dot = u = ca.SX.sym('x_dot')
    u = ca.SX.sym('u')

    model.f_expl_expr = u  # xdot=u
    model.f_impl_expr = x_dot - u  # xdot=u
    model.xdot = x_dot
    model.x = x
    model.u = u
    model.name = name
    return model


def export_n_integrator_model(name, n_order=4, x_dim=2):
    # x^n = A x + Bu
    model = AcadosModel()
    x = ca.SX.sym('x', n_order*x_dim)
    x_dot = ca.SX.sym('x_dot', n_order*x_dim)
    u = ca.SX.sym('u', x_dim)

    # for i in range(n_order):

    A = np.diag(np.ones((n_order-1)*x_dim), x_dim)
    B = np.zeros((n_order*x_dim, x_dim))
    np.fill_diagonal(np.fliplr(np.flipud(B)), 1)

    f_expl = A@x + B@u
    f_impl = x_dot - f_expl

    model.f_expl_expr = f_expl  # xdot=u
    model.f_impl_expr = f_impl  # xdot=u
    model.xdot = x_dot
    model.x = x
    model.u = u
    model.name = name
    return model

def export_N_model():
    
    return model

def export_pendulum_ode_model_with_discrete_rk4(name, n_order=4, x_dim=2):
    model = export_n_integrator_model(name, n_order, x_dim)
    dT = ca.SX.sym('dt', 1)
    T = ca.SX.sym('T', 1)
    x = model.x
    u = model.u
    model.x = ca.vertcat(x, T)
    model.u = ca.vertcat(u, dT)
    x = model.x
    u = model.u
    xdot = ca.vertcat(model.xdot, 1)
    f_expl = ca.vertcat(model.f_expl_expr, 1)
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl

    ode = ca.Function('ode', [x, u], [model.f_expl_expr])
    # set up RK4
    k1 = ode(x,       u)
    k2 = ode(x+dT/2*k1, u)
    k3 = ode(x+dT/2*k2, u)
    k4 = ode(x+dT*k3,  u)
    xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.xdot = xdot
    model.disc_dyn_expr = xf
    # print("built RK4 for pendulum model with dT = ", dT)
    # print(xf)
    return model


def export_pendulum_ode_model_with_discrete_rk4_Lc(name, n_order=4, x_dim=2):
    model = export_n_integrator_model(name, n_order, x_dim)
    dT = ca.SX.sym('dt', 1)
    z = ca.SX.sym('z', x_dim)
    T = ca.SX.sym('T', 1)
    x = model.x
    u = model.u
    model.x = ca.vertcat(x, T)
    model.u = ca.vertcat(u, dT, z)
    x = model.x
    u = model.u
    xdot = ca.vertcat(model.xdot, 1)
    f_expl = ca.vertcat(model.f_expl_expr, 1)
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl

    ode = ca.Function('ode', [x, u], [model.f_expl_expr])
    # set up RK4
    k1 = ode(x,       u)
    k2 = ode(x+dT/2*k1, u)
    k3 = ode(x+dT/2*k2, u)
    k4 = ode(x+dT*k3,  u)
    xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.xdot = xdot
    model.disc_dyn_expr = xf
    # print("built RK4 for pendulum model with dT = ", dT)
    # print(xf)
    return model

def export_NH_integrator_ode_model_with_discrete_rk4(name, n_order=4, x_dim=2):
    model = export_NH_integrator_model(name)
    dT = ca.SX.sym('dt', 1)
    T = ca.SX.sym('T', 1)
    x = model.x
    u = model.u
    model.x = ca.vertcat(x, T)
    model.u = ca.vertcat(u, dT)
    x = model.x
    u = model.u
    xdot = ca.vertcat(model.xdot, 1)
    f_expl = ca.vertcat(model.f_expl_expr, 1)
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl

    ode = ca.Function('ode', [x, u], [model.f_expl_expr])
    # set up RK4
    k1 = ode(x,       u)
    k2 = ode(x+dT/2*k1, u)
    k3 = ode(x+dT/2*k2, u)
    k4 = ode(x+dT*k3,  u)
    xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.xdot = xdot
    model.disc_dyn_expr = xf
    # print("built RK4 for pendulum model with dT = ", dT)
    # print(xf)
    return model


def export_NH_integrator_model(name):
    # x^n = A x + Bu
    model = AcadosModel()
    x = ca.SX.sym('x', 3)
    x_dot = ca.SX.sym('x_dot', 3)
    u = ca.SX.sym('u', 2)

    # x_dot = V*cos(theta), V*sin(theta), omega
    f_expl = ca.vertcat(u[0]*ca.cos(x[2]), u[0]*ca.sin(x[2]), u[1])
    f_impl = x_dot - f_expl

    model.f_expl_expr = f_expl  # xdot=u
    model.f_impl_expr = f_impl  # xdot=u
    model.xdot = x_dot
    model.x = x
    model.u = u
    model.name = name
    return model

def export_robot_model_with_discrete_rk4(name):
    model = export_robot_model(name)
    dT = ca.SX.sym('dt', 1)
    T = ca.SX.sym('T', 1)
    x = model.x
    u = model.u
    model.x = ca.vertcat(x, T)
    model.u = ca.vertcat(u, dT)
    x = model.x
    u = model.u
    xdot = ca.vertcat(model.xdot, 1)
    f_expl = ca.vertcat(model.f_expl_expr, 1)
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl

    ode = ca.Function('ode', [x, u], [model.f_expl_expr])
    # set up RK4
    k1 = ode(x,       u)
    k2 = ode(x+dT/2*k1, u)
    k3 = ode(x+dT/2*k2, u)
    k4 = ode(x+dT*k3,  u)
    xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.xdot = xdot
    model.disc_dyn_expr = xf
    # print("built RK4 for pendulum model with dT = ", dT)
    # print(xf)
    return model

def export_unicycle_model_with_discrete_rk4(name):
    model = export_unicycle_model(name)
    dT = ca.SX.sym('dt', 1)
    T = ca.SX.sym('T', 1)
    x = model.x
    u = model.u
    model.x = ca.vertcat(x, T)
    model.u = ca.vertcat(u, dT)
    x = model.x
    u = model.u
    xdot = ca.vertcat(model.xdot, 1)
    f_expl = ca.vertcat(model.f_expl_expr, 1)
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl

    ode = ca.Function('ode', [x, u], [model.f_expl_expr])
    # set up RK4
    k1 = ode(x,       u)
    k2 = ode(x+dT/2*k1, u)
    k3 = ode(x+dT/2*k2, u)
    k4 = ode(x+dT*k3,  u)
    xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.xdot = xdot
    model.disc_dyn_expr = xf
    # print("built RK4 for pendulum model with dT = ", dT)
    # print(xf)
    return model

def export_robot_model(name) -> AcadosModel:
    # model_name = "unicycle_ode"

    # set up states & controls
    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    v = ca.SX.sym("x_d")
    theta = ca.SX.sym("theta")
    theta_d = ca.SX.sym("theta_d")

    x = ca.vertcat(x, y, theta, v, theta_d)

    F = ca.SX.sym("F")
    T = ca.SX.sym("T")
    u = ca.vertcat(F, T)

    # xdot
    x_dot = ca.SX.sym("x_dot")
    y_dot = ca.SX.sym("y_dot")
    v_dot = ca.SX.sym("v_dot")
    theta_dot = ca.SX.sym("theta_dot")
    theta_ddot = ca.SX.sym("theta_ddot")

    xdot = ca.vertcat(x_dot, y_dot, theta_dot, v_dot, theta_ddot)

    # algebraic variables
    # z = None

    # parameters
    p = []

    # dynamics
    f_expl = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), theta_d, F, T)

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    model.name = name

    return model

def export_unicycle_model(name) -> AcadosModel:
    # model_name = "unicycle_ode"

    # set up states & controls
    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    v = ca.SX.sym("x_d")
    theta = ca.SX.sym("theta")
    theta_d = ca.SX.sym("theta_d")

    x = ca.vertcat(x, y, theta,v)

    F = ca.SX.sym("F")
    T = ca.SX.sym("T")
    u = ca.vertcat(F, T)

    # xdot
    x_dot = ca.SX.sym("x_dot")
    y_dot = ca.SX.sym("y_dot")
    v_dot = ca.SX.sym("v_dot")
    theta_dot = ca.SX.sym("theta_dot")
    theta_ddot = ca.SX.sym("theta_ddot")

    xdot = ca.vertcat(x_dot, y_dot, theta_dot, v_dot)

    # algebraic variables
    # z = None

    # parameters
    p = []

    # dynamics
    f_expl = ca.vertcat(v * ca.cos(theta), v * ca.sin(theta), T, F)

    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    model.name = name

    return model


def export_bicycle_model_with_discrete_rk4(name):
    model = export_bicycle_model(name)
    dT = ca.SX.sym('dt', 1)
    T = ca.SX.sym('T', 1)
    x = model.x
    u = model.u
    model.x = ca.vertcat(x, T)
    model.u = ca.vertcat(u, dT)
    x = model.x
    u = model.u
    xdot = ca.vertcat(model.xdot, 1)
    f_expl = ca.vertcat(model.f_expl_expr, 1)
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl

    ode = ca.Function('ode', [x, u], [model.f_expl_expr])
    # set up RK4
    k1 = ode(x,       u)
    k2 = ode(x+dT/2*k1, u)
    k3 = ode(x+dT/2*k2, u)
    k4 = ode(x+dT*k3,  u)
    xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.xdot = xdot
    model.disc_dyn_expr = xf
    # print("built RK4 for pendulum model with dT = ", dT)
    # print(xf)
    return model

def export_bicycle_model(name):
    model = AcadosModel()

    # set up states & controls
    x = ca.SX.sym("x")
    y = ca.SX.sym("y")
    phi = ca.SX.sym("phi")
    v = ca.SX.sym("v")

    x = ca.vertcat(x, y, phi, v)

    a = ca.SX.sym("a")
    delta = ca.SX.sym("delta")
    u = ca.vertcat(a, delta)

    # xdot
    x_dot = ca.SX.sym("x_dot")
    y_dot = ca.SX.sym("y_dot")
    phi_dot = ca.SX.sym("phi_dot")
    v_dot = ca.SX.sym("v_dot")

    xdot = ca.vertcat(x_dot, y_dot, phi_dot, v_dot)
    factor = 0.01
    lf = 1.105*factor
    lr = 1.738*factor
    # # dynamics
    b = ca.atan(lr * ca.tan(delta) / (lf + lr))  
    f_expl = ca.vertcat(v * ca.cos(phi + b), v * ca.sin(phi + b), v * ca.sin(b) / lr, a)
    # f_expl = ca.vertcat(v * ca.cos(phi), v * ca.sin(phi), v * ca.tan(delta) / (lr+lf), a)
    f_impl = xdot - f_expl

    model.f_expl_expr = f_expl  # xdot=u
    model.f_impl_expr = f_impl  # xdot=u
    model.xdot = xdot
    model.x = x
    model.u = u
    model.name = name
    return model


##############################Lipchitz constant#############################################
def export_unicycle_model_with_discrete_rk4_LC(name):
    model = export_unicycle_model(name)
    dT = ca.SX.sym('dt', 1)
    z = ca.SX.sym('z', 2) # z = [x,y]
    T = ca.SX.sym('T', 1)
    x = model.x
    u = model.u
    model.x = ca.vertcat(x, T)
    model.u = ca.vertcat(u, dT, z)
    x = model.x
    u = model.u
    xdot = ca.vertcat(model.xdot, 1)
    f_expl = ca.vertcat(model.f_expl_expr, 1)
    model.f_expl_expr = f_expl
    model.f_impl_expr = xdot - f_expl

    ode = ca.Function('ode', [x, u], [model.f_expl_expr])
    # set up RK4
    k1 = ode(x,       u)
    k2 = ode(x+dT/2*k1, u)
    k3 = ode(x+dT/2*k2, u)
    k4 = ode(x+dT*k3,  u)
    xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.xdot = xdot
    model.disc_dyn_expr = xf
    # print("built RK4 for pendulum model with dT = ", dT)
    # print(xf)
    return model