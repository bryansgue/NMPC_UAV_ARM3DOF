from casadi import Function
from casadi import MX
from casadi import reshape
from casadi import vertcat
from casadi import horzcat
from casadi import cos
from casadi import sin
from casadi import atan2
from casadi import solve
from casadi import inv
from casadi import mtimes
from casadi import norm_2
from casadi import if_else
from casadi import jacobian
from acados_template import AcadosModel
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import matplotlib.pyplot as plt
import time
import rospy
from std_msgs.msg import Float64MultiArray



# Global variables Odometry Drone
x_real = 0.0
y_real = 0.0
z_real = 5
vx_real = 0.0
vy_real = 0.0
vz_real = 0.0
qw_real = 1.0
qx_real = 0
qy_real = 0.0
qz_real = 0.0
wx_real = 0.0
wy_real = 0.0
wz_real = 0.0


def f_system_simple_model():
    # Name of the system
    model_name = 'Drone_ode'
    # Dynamic Values of the system

    
    # set up states & controls
    # Position
    nx = MX.sym('nx') 
    ny = MX.sym('ny')
    nz = MX.sym('nz')
    psi = MX.sym('psi')
    px = MX.sym('px')
    py = MX.sym('py')
    pz = MX.sym('pz')
    ul = MX.sym('ul')
    um = MX.sym('um')
    un = MX.sym('un')
    uw = MX.sym('w')
    q1 = MX.sym('q1')
    q2 = MX.sym('q2')
    q3 = MX.sym('q3')
    q1_p = MX.sym('q1_p')
    q2_p = MX.sym('q2_p')
    q3_p = MX.sym('q3_p')


    # General vector of the states
    x = vertcat(nx, ny, nz, psi, px, py, pz, q1, q2, q3,  ul, um, un, uw, q1_p, q2_p, q3_p)

    # Action variables
    ul_ref = MX.sym('ul_ref')
    um_ref = MX.sym('um_ref')
    un_ref = MX.sym('un_ref')
    w_ref = MX.sym('w_ref')
    q1_p_ref = MX.sym('q1_p_ref')
    q2_p_ref = MX.sym('q2_p_ref')
    q3_p_ref = MX.sym('q3_p_ref')

    # General Vector Action variables
    u = vertcat(ul_ref,um_ref,un_ref,w_ref, q1_p_ref, q2_p_ref, q3_p_ref)

    # Variables to explicit function
    nx_p = MX.sym('nx_p')
    ny_p = MX.sym('ny_p')
    nz_p = MX.sym('nz_p')
    psi_p = MX.sym('psi_p')
    ul_p = MX.sym('ul_p')
    um_p = MX.sym('um_p')
    un_p = MX.sym('un_p')
    w_p = MX.sym('w_p')
    px_p = MX.sym('px_p')
    py_p = MX.sym('py_p')
    pz_p = MX.sym('pz_p')
    q1_pp = MX.sym('q1_pp')
    q2_pp = MX.sym('q2_pp')
    q3_pp = MX.sym('q3_pp')

    # general vector X dot for implicit function
    xdot = vertcat(nx_p, ny_p, nz_p, psi_p, px_p, py_p, pz_p, q1_p, q2_p, q3_p,  ul_p, um_p, un_p, w_p, q1_pp, q2_pp, q3_pp)

    # Ref system as a external value
    nx_d = MX.sym('nx_d')
    ny_d = MX.sym('ny_d')
    nz_d = MX.sym('nz_d')
    psi_d = MX.sym('psi_d')
    ul_d= MX.sym('ul_d')
    um_d= MX.sym('um_d')
    un_d = MX.sym('un_d')
    w_d = MX.sym('w_d')
    q1_p_d = MX.sym('q1_p_d')
    q2_p_d = MX.sym('q2_p_d')
    q3_p_d = MX.sym('q3_p_d')

    px_d = MX.sym('px_d')
    py_d = MX.sym('py_d')
    pz_d = MX.sym('pz_d')
    
    q1_d = MX.sym('q1_d')
    q2_d = MX.sym('q2_d')
    q3_d = MX.sym('q3_d')

    ul_ref_d= MX.sym('ul_ref_d')
    um_ref_d= MX.sym('um_ref_d')
    un_ref_d = MX.sym('un_ref_d')
    w_ref_d = MX.sym('w_ref_d')
    q1_p_ref_d = MX.sym('q1_p_ref_d')
    q2_p_ref_d = MX.sym('q2_p_ref_d')
    q3_p_ref_d = MX.sym('q3_p_ref_d')

    
    p = vertcat(nx_d, ny_d, nz_d, psi_d, px_d, py_d, pz_d, q1_d, q2_d, q3_d, ul_d, um_d, un_d, w_d, q1_p_d ,q2_p_d ,q3_p_d , ul_ref_d, um_ref_d, un_ref_d, w_ref_d, q1_p_ref_d , q2_p_ref_d , q3_p_ref_d )

    # Rotational Matrix
    a = 0
    b = 0
    l2 = 0.44
    l3 = 0.45
    J = calculate_J(x, l2, l3, a, b)
    R_z = rotation_matrix_z(x)

    # Definir la matriz A en CasADi
    A = MX(7, 7)
    A[0, :] = [-0.4662,  0.0287, -0.0096,  0.0520, -0.2731,  0.3828, -0.0839]
    A[1, :] = [-0.0991, -0.1861,  0.5414, -0.0649, -0.2958,  0.0016, -0.0078]
    A[2, :] = [ 0.0052,  0.0086, -3.4996,  0.0084, -0.0062,  0.0137,  0.0093]
    A[3, :] = [-0.0551,  0.1256, -0.7545, -8.8799, -0.0504, -0.4081,  0.2048]
    A[4, :] = [ 0.0887, -0.0670, -0.0913, -0.0346, -6.2092, -0.2675,  0.1461]
    A[5, :] = [ 0.2260, -0.3115,  1.1240,  0.1200, -0.0909, -9.4262,  0.9687]
    A[6, :] = [-0.0797, -0.1204,  0.1032, -0.0652, -0.3551,  0.4772, -11.0373]

    # Definir la matriz B en CasADi
    B = MX(7, 7)
    B[0, :] = [ 0.5969,  0.0879, -0.0662, -0.0423,  0.2227, -0.3088,  0.0355]
    B[1, :] = [ 0.0976,  0.4041, -0.5105,  0.0698,  0.2486, -0.0069,  0.0258]
    B[2, :] = [-0.0136, -0.0107,  3.5092, -0.0097,  0.0080, -0.0107, -0.0112]
    B[3, :] = [ 0.0675, -0.1475,  0.8577,  8.9287,  0.1352,  0.4226, -0.1536]
    B[4, :] = [-0.0657,  0.0422, -0.0572,  0.0514,  6.3038,  0.3033, -0.1579]
    B[5, :] = [-0.1058,  0.0743, -0.8774, -0.1278,  0.2452,  8.9981, -0.9936]
    B[6, :] = [-0.1820,  0.0812, -0.2088,  0.0730,  0.3007, -0.5067, 10.9660]



    h_p = J@ vertcat(ul, um, un, uw, q1_p, q2_p, q3_p)
    p_p = R_z@vertcat(ul, um, un)
    q_p = vertcat(q1_p, q2_p, q3_p)
    v_p = A@vertcat(ul, um, un, uw, q1_p, q2_p, q3_p) + B@u

   
    f_expl = vertcat(h_p, p_p ,q_p, v_p)

    f_system = Function('system',[x, u], [f_expl])
     # Acados Model
    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    model.p = p


    return model, f_system


def f_system_simple_model_quat():
    # Name of the system
    model_name = 'Drone_ode'
    # Dynamic Values of the system

    
    # set up states & controls
    # Position
    nx = MX.sym('nx') 
    ny = MX.sym('ny')
    nz = MX.sym('nz')
    hpsi = MX.sym('psi')
    qw = MX.sym('qw')
    qx = MX.sym('qx')
    qy = MX.sym('qy')
    qz = MX.sym('qz')
    px = MX.sym('px')
    py = MX.sym('py')
    pz = MX.sym('pz')
    ul = MX.sym('ul')
    um = MX.sym('um')
    un = MX.sym('un')
    uw = MX.sym('w')
    q1 = MX.sym('q1')
    q2 = MX.sym('q2')
    q3 = MX.sym('q3')
    q1_p = MX.sym('q1_p')
    q2_p = MX.sym('q2_p')
    q3_p = MX.sym('q3_p')


    # General vector of the states
    x = vertcat(nx, ny, nz, hpsi, qw, qx, qy, qz, px, py, pz, q1, q2, q3,  ul, um, un, uw, q1_p, q2_p, q3_p)

    # Action variables
    ul_ref = MX.sym('ul_ref')
    um_ref = MX.sym('um_ref')
    un_ref = MX.sym('un_ref')
    w_ref = MX.sym('w_ref')
    q1_p_ref = MX.sym('q1_p_ref')
    q2_p_ref = MX.sym('q2_p_ref')
    q3_p_ref = MX.sym('q3_p_ref')

    # General Vector Action variables
    u = vertcat(ul_ref,um_ref,un_ref,w_ref, q1_p_ref, q2_p_ref, q3_p_ref)

    # Variables to explicit function
    nx_p = MX.sym('nx_p')
    ny_p = MX.sym('ny_p')
    nz_p = MX.sym('nz_p')
    psi_p = MX.sym('psi_p')
    qw_p = MX.sym('qw_p')
    qx_p = MX.sym('qx_p')
    qy_p = MX.sym('qy_p')
    qz_p = MX.sym('qz_p')
    ul_p = MX.sym('ul_p')
    um_p = MX.sym('um_p')
    un_p = MX.sym('un_p')
    w_p = MX.sym('w_p')
    px_p = MX.sym('px_p')
    py_p = MX.sym('py_p')
    pz_p = MX.sym('pz_p')
    q1_pp = MX.sym('q1_pp')
    q2_pp = MX.sym('q2_pp')
    q3_pp = MX.sym('q3_pp')

    # general vector X dot for implicit function
    xdot = vertcat(nx_p, ny_p, nz_p, psi_p, qw_p,qx_p,qy_p,qz_p, px_p, py_p, pz_p, q1_p, q2_p, q3_p,  ul_p, um_p, un_p, w_p, q1_pp, q2_pp, q3_pp)

    # Ref system as a external value
    nx_d = MX.sym('nx_d')
    ny_d = MX.sym('ny_d')
    nz_d = MX.sym('nz_d')
    psi_d = MX.sym('psi_d')

    qw_d = MX.sym('qw_d')
    qx_d = MX.sym('qx_d')
    qy_d = MX.sym('qy_d')
    qz_d = MX.sym('qz_d')
    px_d = MX.sym('px_d')
    py_d = MX.sym('py_d')
    pz_d = MX.sym('pz_d')
    q1_d = MX.sym('q1_d')
    q2_d = MX.sym('q2_d')
    q3_d = MX.sym('q3_d')
    ul_d= MX.sym('ul_d')
    um_d= MX.sym('um_d')
    un_d = MX.sym('un_d')
    w_d = MX.sym('w_d')
    q1_p_d = MX.sym('q1_p_d')
    q2_p_d = MX.sym('q2_p_d')
    q3_p_d = MX.sym('q3_p_d')


    ul_ref_d= MX.sym('ul_ref_d')
    um_ref_d= MX.sym('um_ref_d')
    un_ref_d = MX.sym('un_ref_d')
    w_ref_d = MX.sym('w_ref_d')
    q1_p_ref_d = MX.sym('q1_p_ref_d')
    q2_p_ref_d = MX.sym('q2_p_ref_d')
    q3_p_ref_d = MX.sym('q3_p_ref_d')

    alpha =  MX.sym('alpha')
    beta =  MX.sym('beta')

    
    p = vertcat(nx_d, ny_d, nz_d, psi_d, qw_d, qx_d, qy_d, qz_d, px_d, py_d, pz_d, q1_d, q2_d, q3_d, ul_d, um_d, un_d, w_d, q1_p_d ,q2_p_d ,q3_p_d , ul_ref_d, um_ref_d, un_ref_d, w_ref_d, q1_p_ref_d , q2_p_ref_d , q3_p_ref_d, alpha , beta )

    # Rotational Matrix
    a = 0
    b = 0
    l2 = 0.44
    l3 = 0.45
    J = calculate_J_quat(x, l2, l3, a, b)
    R_z = rotation_matrix_z_quat(x)

    # Definir la matriz A en CasADi
    A = MX(7, 7)
    A[0, :] = [-0.4662,  0.0287, -0.0096,  0.0520, -0.2731,  0.3828, -0.0839]
    A[1, :] = [-0.0991, -0.1861,  0.5414, -0.0649, -0.2958,  0.0016, -0.0078]
    A[2, :] = [ 0.0052,  0.0086, -3.4996,  0.0084, -0.0062,  0.0137,  0.0093]
    A[3, :] = [-0.0551,  0.1256, -0.7545, -8.8799, -0.0504, -0.4081,  0.2048]
    A[4, :] = [ 0.0887, -0.0670, -0.0913, -0.0346, -6.2092, -0.2675,  0.1461]
    A[5, :] = [ 0.2260, -0.3115,  1.1240,  0.1200, -0.0909, -9.4262,  0.9687]
    A[6, :] = [-0.0797, -0.1204,  0.1032, -0.0652, -0.3551,  0.4772, -11.0373]

    # Definir la matriz B en CasADi
    B = MX(7, 7)
    B[0, :] = [ 0.5969,  0.0879, -0.0662, -0.0423,  0.2227, -0.3088,  0.0355]
    B[1, :] = [ 0.0976,  0.4041, -0.5105,  0.0698,  0.2486, -0.0069,  0.0258]
    B[2, :] = [-0.0136, -0.0107,  3.5092, -0.0097,  0.0080, -0.0107, -0.0112]
    B[3, :] = [ 0.0675, -0.1475,  0.8577,  8.9287,  0.1352,  0.4226, -0.1536]
    B[4, :] = [-0.0657,  0.0422, -0.0572,  0.0514,  6.3038,  0.3033, -0.1579]
    B[5, :] = [-0.1058,  0.0743, -0.8774, -0.1278,  0.2452,  8.9981, -0.9936]
    B[6, :] = [-0.1820,  0.0812, -0.2088,  0.0730,  0.3007, -0.5067, 10.9660]

    # Evolucion quat
    quat = vertcat(qw, qx, qy, qz)
    
    p_x = 0
    q = 0
    r = uw

    S = vertcat(
        horzcat(0, -p_x, -q, -r),
        horzcat(p_x, 0, r, -q),
        horzcat(q, -r, 0, p_x),
        horzcat(r, q, -p_x, 0)
    )


    h_p = J@ vertcat(ul, um, un, uw, q1_p, q2_p, q3_p)
    quat_p = 1/2*S @ quat
    p_p = R_z@vertcat(ul, um, un)
    q_p = vertcat(q1_p, q2_p, q3_p)
    v_p = A@vertcat(ul, um, un, uw, q1_p, q2_p, q3_p) + B@u

   
    f_expl = vertcat(h_p,quat_p, p_p ,q_p, v_p)

    # Define f_x and g_x
    f_x = Function('f_x', [x], [f_expl])
    g_x = Function('g_x', [x, u], [jacobian(f_expl, u)])

    f_system = Function('system',[x, u], [f_expl])
     # Acados Model
    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    model.p = p


    return model, f_system


def f_system_simple_model_quat_no_vale():
    # Name of the system
    model_name = 'Drone_ode'
    # Dynamic Values of the system

    chi = [0.6756,    1.0000,    0.6344,    1.0000,    0.4080,    1.0000,    1.0000,    1.0000,    0.2953,    0.5941,   -0.8109,    1.0000,    0.3984,    0.7040,    1.0000,    0.9365,    1.0000, 1.0000,    0.9752]# Position
    
    # set up states & controls
    # Position
    nx = MX.sym('nx') 
    ny = MX.sym('ny')
    nz = MX.sym('nz')
    qw = MX.sym('qw')
    qx = MX.sym('qx')
    qy = MX.sym('qy')
    qz = MX.sym('qz')
    ul = MX.sym('ul')
    um = MX.sym('um')
    un = MX.sym('un')
    w = MX.sym('w')

    # General vector of the states
    x = vertcat(nx, ny, nz, qw, qx, qy, qz, ul, um, un, w)

    # Action variables
    ul_ref = MX.sym('ul_ref')
    um_ref = MX.sym('um_ref')
    un_ref = MX.sym('un_ref')
    w_ref = MX.sym('w_ref')

    # General Vector Action variables
    u = vertcat(ul_ref,um_ref,un_ref,w_ref)

    # Variables to explicit function
    nx_p = MX.sym('nx_p')
    ny_p = MX.sym('ny_p')
    nz_p = MX.sym('nz_p')
    qw_p = MX.sym('qw_p')
    qx_p = MX.sym('qx_p')
    qy_p = MX.sym('qy_p')
    qz_p = MX.sym('qz_p')
    ul_p = MX.sym('ul_p')
    um_p = MX.sym('um_p')
    un_p = MX.sym('un_p')
    w_p = MX.sym('w_p')

    # general vector X dot for implicit function
    xdot = vertcat(nx_p,ny_p,nz_p,qw_p,qx_p,qy_p,qz_p,ul_p,um_p,un_p,w_p)

    # Ref system as a external value
    nx_d = MX.sym('nx_d')
    ny_d = MX.sym('ny_d')
    nz_d = MX.sym('nz_d')
    qw_d = MX.sym('qw_d')
    qx_d = MX.sym('qx_d')
    qy_d = MX.sym('qy_d')
    qz_d = MX.sym('qz_d')
    ul_d = MX.sym('ul_d')
    um_d= MX.sym('um_d')
    un_d = MX.sym('un_d')
    w_d = MX.sym('w_d')

    ul_ref_d= MX.sym('ul_ref_d')
    um_ref_d= MX.sym('um_ref_d')
    un_ref_d = MX.sym('un_ref_d')
    w_ref_d = MX.sym('w_ref_d')

    nx_obs = MX.sym('nx_obs')
    ny_obs = MX.sym('ny_obs')
    
    p = vertcat(nx_d, ny_d, nz_d, qw_d, qx_d, qy_d, qz_d, ul_d, um_d, un_d, w_d, ul_ref_d, um_ref_d, un_ref_d, w_ref_d)

    # Rotational Matrix
    a = 0
    b = 0
    
    M = calc_M(chi,a,b)
    C = calc_C(chi,a,b, w)
    G = calc_G()

    # Crea una lista de MX con los componentes del cuaternión
    quat = [qw, qx, qy, qz]

    # Obtener la matriz de rotación
    J = QuatToRot(quat)

    # Evolucion quat
    p_x = 0
    q = 0
    r = w

    S = vertcat(
        horzcat(0, -p_x, -q, -r),
        horzcat(p_x, 0, r, -q),
        horzcat(q, -r, 0, p_x),
        horzcat(r, q, -p_x, 0)
    )

    quat_p = 1/2*S @ quat


    # Crear matriz A
    A_1 = horzcat(MX.zeros(3, 7), J, MX.zeros(3, 1))
    A_2 = horzcat(MX.zeros(4, 3), 1/2*S, MX.zeros(4, 4))
    A_3 = horzcat(MX.zeros(4, 7), -mtimes(inv(M), C))
   
    A = vertcat(A_1, A_2, A_3)

    # Crear matriz B
    B_top = MX.zeros(7, 4)
    B_bottom = inv(M)
    B = vertcat(B_top, B_bottom)

    f_expl = MX.zeros(11, 1)
    f_expl = A @ x + B @ u 

    f_x = A @ x 
    g_x = B

    f_system = Function('system',[x, u], [f_expl])
     # Acados Model
    f_impl = xdot - f_expl

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.name = model_name
    model.p = p

    return model, f_system

def odometry_call_back(odom_msg):
    global x_real, y_real, z_real, qx_real, qy_real, qz_real, qw_real, vx_real, vy_real, vz_real, wx_real, wy_real, wz_real
    # Read desired linear velocities from node
    x_real = odom_msg.pose.pose.position.x 
    y_real = odom_msg.pose.pose.position.y
    z_real = odom_msg.pose.pose.position.z
    vx_real = odom_msg.twist.twist.linear.x
    vy_real = odom_msg.twist.twist.linear.y
    vz_real = odom_msg.twist.twist.linear.z

    qx_real = odom_msg.pose.pose.orientation.x
    qy_real = odom_msg.pose.pose.orientation.y
    qz_real = odom_msg.pose.pose.orientation.z
    qw_real = odom_msg.pose.pose.orientation.w

    wx_real = odom_msg.twist.twist.angular.x
    wy_real = odom_msg.twist.twist.angular.y
    wz_real = odom_msg.twist.twist.angular.z
    return None

def get_odometry_simple():
    global x_real, y_real, z_real, qx_real, qy_real, qz_real, qw_real, vx_real, vy_real, vz_real, wx_real, wy_real, wz_real
    
    quaternion = [qx_real, qy_real, qz_real, qw_real ] # cuaternión debe estar en la convención "xyzw",
    r_quat = R.from_quat(quaternion)
    euler =  r_quat.as_euler('zyx', degrees = False)
    psi = euler[0]

    J = np.zeros((3, 3))
    J[0, 0] = np.cos(psi)
    J[0, 1] = -np.sin(psi)
    J[1, 0] = np.sin(psi)
    J[1, 1] = np.cos(psi)
    J[2, 2] = 1

    J_inv = np.linalg.inv(J)
    v = np.dot(J_inv, [vx_real, vy_real, vz_real])
 
    ul_real = v[0]
    um_real = v[1]
    un_real = v[2]

    x_state = [x_real,y_real,z_real,psi,ul_real,um_real,un_real, wz_real]

    return x_state

def get_odometry_simple_quat():

    quaternion = [qx_real, qy_real, qz_real, qw_real ] # cuaternión debe estar en la convención "xyzw",
    r_quat = R.from_quat(quaternion)
    euler =  r_quat.as_euler('zyx', degrees = False)
    psi = euler[0]

    J = np.zeros((3, 3))
    J[0, 0] = np.cos(psi)
    J[0, 1] = -np.sin(psi)
    J[1, 0] = np.sin(psi)
    J[1, 1] = np.cos(psi)
    J[2, 2] = 1

    J_inv = np.linalg.inv(J)
    v = np.dot(J_inv, [vx_real, vy_real, vz_real])
 
    ul_real = v[0]
    um_real = v[1]
    un_real = v[2]

    x_state = [x_real,y_real,z_real,qw_real,qx_real,qy_real,qz_real,ul_real,um_real,un_real, wz_real]

    return x_state

def send_velocity_control(u, vel_pub, vel_msg):
    # velocity message

    vel_msg.header.frame_id = "base_link"
    vel_msg.header.stamp = rospy.Time.now()
    vel_msg.twist.linear.x = u[0]
    vel_msg.twist.linear.y = u[1]
    vel_msg.twist.linear.z = u[2]
    vel_msg.twist.angular.x = 0
    vel_msg.twist.angular.y = 0
    vel_msg.twist.angular.z = u[3]

    # Publish control values
    vel_pub.publish(vel_msg)

def pub_odometry_sim(state_vector, odom_sim_pub, odom_sim_msg):

    quaternion = euler_to_quaternion(0, 0, state_vector[3])
    u = [state_vector[4],state_vector[5],state_vector[6]]
    
    v = FLUtoENU(u, quaternion)

    odom_sim_msg.header.frame_id = "odo"
    odom_sim_msg.header.stamp = rospy.Time.now()
    odom_sim_msg.pose.pose.position.x = state_vector[0]
    odom_sim_msg.pose.pose.position.y = state_vector[1]
    odom_sim_msg.pose.pose.position.z = state_vector[2]
    odom_sim_msg.pose.pose.orientation.x = quaternion[1]
    odom_sim_msg.pose.pose.orientation.y = quaternion[2]
    odom_sim_msg.pose.pose.orientation.z = quaternion[3]
    odom_sim_msg.pose.pose.orientation.w = quaternion[0]
    odom_sim_msg.twist.twist.linear.x = v[0]
    odom_sim_msg.twist.twist.linear.y = v[1]
    odom_sim_msg.twist.twist.linear.z = v[2]
    odom_sim_msg.twist.twist.angular.x = 0
    odom_sim_msg.twist.twist.angular.y = 0
    odom_sim_msg.twist.twist.angular.z = state_vector[7]

    # Publish the message
    odom_sim_pub.publish(odom_sim_msg)

def pub_odometry_sim_quat(state_vector, odom_sim_pub, odom_sim_msg):
    
    quaternion = [state_vector[3], state_vector[4], state_vector[5], state_vector[6]]

    u = [state_vector[7],state_vector[8],state_vector[9]]
    
    v = FLUtoENU(u, quaternion)

    odom_sim_msg.header.frame_id = "odo"
    odom_sim_msg.header.stamp = rospy.Time.now()
    odom_sim_msg.pose.pose.position.x = state_vector[0]
    odom_sim_msg.pose.pose.position.y = state_vector[1]
    odom_sim_msg.pose.pose.position.z = state_vector[2]
    odom_sim_msg.pose.pose.orientation.w = state_vector[3]
    odom_sim_msg.pose.pose.orientation.x = state_vector[4]
    odom_sim_msg.pose.pose.orientation.y = state_vector[5]
    odom_sim_msg.pose.pose.orientation.z = state_vector[6]
    odom_sim_msg.twist.twist.linear.x = v[0]
    odom_sim_msg.twist.twist.linear.y = v[1]
    odom_sim_msg.twist.twist.linear.z = v[2]
    odom_sim_msg.twist.twist.angular.x = 0
    odom_sim_msg.twist.twist.angular.y = 0
    odom_sim_msg.twist.twist.angular.z = state_vector[10]

    
    # Publish the message
    odom_sim_pub.publish(odom_sim_msg)    

def calc_M(chi, a, b):
    

    M = MX.zeros(4, 4)
    M[0,0] = chi[0]
    M[0,1] = 0
    M[0,2] = 0
    M[0,3] = b * chi[1]
    M[1,0] = 0
    M[1,1] = chi[2]
    M[1,2] = 0
    M[1,3] = a* chi[3]
    M[2,0] = 0
    M[2,1] = 0
    M[2,2] = chi[4]
    M[2,3] = 0
    M[3,0] = b*chi[5]
    M[3,1] = a* chi[6]
    M[3,2] = 0
    M[3,3] = chi[7]*(a**2+b**2) + chi[8]
    
    return M

def calc_C(chi, a, b, w):
    
    C = MX.zeros(4, 4)
    C[0,0] = chi[9]
    C[0,1] = w*chi[10]
    C[0,2] = 0
    C[0,3] = a * w * chi[11]
    C[1,0] = w*chi[12]
    C[1,1] = chi[13]
    C[1,2] = 0
    C[1,3] = b * w * chi[14]
    C[2,0] = 0
    C[2,1] = 0
    C[2,2] = chi[15]
    C[2,3] = 0
    C[3,0] = a *w* chi[16]
    C[3,1] = b * w * chi[17]
    C[3,2] = 0
    C[3,3] = chi[18]

    return C

def calc_G():
    G = MX.zeros(4, 1)
    G[0, 0] = 0
    G[1, 0] = 0
    G[2, 0] = 0
    G[3, 0] = 0

    return G


def rotation_matrix_z(x):
    # Extraer el ángulo psi de x
    psi = x[3]
    
    # Definir la matriz de rotación en Z usando CasADi
    R_z = MX(3, 3)
    R_z[0, 0] = cos(psi)
    R_z[0, 1] = -sin(psi)
    R_z[0, 2] = 0
    R_z[1, 0] = sin(psi)
    R_z[1, 1] = cos(psi)
    R_z[1, 2] = 0
    R_z[2, 0] = 0
    R_z[2, 1] = 0
    R_z[2, 2] = 1
    
    return R_z

def rotation_matrix_z_quat(x):
    
    # Extraer las variables de cuaterniones simbólicos
    qw = x[4]
    qx = x[5]
    qy = x[6]
    qz = x[7]

    # Calcular el ángulo psi usando atan2 con CasADi
    psi = atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
    
    # Definir la matriz de rotación en Z usando CasADi
    R_z = MX(3, 3)
    R_z[0, 0] = cos(psi)
    R_z[0, 1] = -sin(psi)
    R_z[0, 2] = 0
    R_z[1, 0] = sin(psi)
    R_z[1, 1] = cos(psi)
    R_z[1, 2] = 0
    R_z[2, 0] = 0
    R_z[2, 1] = 0
    R_z[2, 2] = 1
    
    return R_z

def calculate_J(x, l2, l3, a, b):
    # Extraer las variables de estado de x
    nx, ny, nz, psi, px, py, pz, q1, q2, q3 = x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9]
    ul, um, un, uw = x[10], x[11], x[12], x[13]
    q1_p, q2_p, q3_p = x[14], x[15], x[16]

    # Crear una matriz CasADi MX 4x7 para J
    J = MX.zeros(4, 7)

    # Definir cada elemento de la matriz J
    J[0, 0] = cos(psi)
    J[0, 1] = -sin(psi)
    J[0, 2] = 0
    J[0, 3] = -sin(psi + q1) * (l3 * cos(q2 + q3) + l2 * cos(q2)) - b * cos(psi) - a * sin(psi)
    J[0, 4] = -sin(psi + q1) * (l3 * cos(q2 + q3) + l2 * cos(q2))
    J[0, 5] = -cos(psi + q1) * (l3 * sin(q2 + q3) + l2 * sin(q2))
    J[0, 6] = -l3 * cos(psi + q1) * sin(q2 + q3)

    J[1, 0] = sin(psi)
    J[1, 1] = cos(psi)
    J[1, 2] = 0
    J[1, 3] = cos(psi + q1) * (l3 * cos(q2 + q3) + l2 * cos(q2)) + a * cos(psi) - b * sin(psi)
    J[1, 4] = cos(psi + q1) * (l3 * cos(q2 + q3) + l2 * cos(q2))
    J[1, 5] = -sin(psi + q1) * (l3 * sin(q2 + q3) + l2 * sin(q2))
    J[1, 6] = -l3 * sin(psi + q1) * sin(q2 + q3)

    J[2, 0] = 0
    J[2, 1] = 0
    J[2, 2] = 1
    J[2, 3] = 0
    J[2, 4] = 0
    J[2, 5] = -l3 * cos(q2 + q3) - l2 * cos(q2)
    J[2, 6] = -l3 * cos(q2 + q3)

    J[3, 0] = 0
    J[3, 1] = 0
    J[3, 2] = 0
    J[3, 3] = 1
    J[3, 4] = 0
    J[3, 5] = 0
    J[3, 6] = 0

    return J

def calculate_J_quat(x, l2, l3, a, b):
    # Extraer las variables de estado de x
    qw = x[4]
    qx = x[5]
    qy = x[6]
    qz = x[7]

   # Calcular el ángulo psi usando arctan2 con CasADi
    psi = atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))


    q1 = x[11]
    q2 = x[12] 
    q3 = x[13]

    # Crear una matriz CasADi MX 4x7 para J
    J = MX.zeros(4, 7)

    # Definir cada elemento de la matriz J
    J[0, 0] = cos(psi)
    J[0, 1] = -sin(psi)
    J[0, 2] = 0
    J[0, 3] = -sin(psi + q1) * (l3 * cos(q2 + q3) + l2 * cos(q2)) - b * cos(psi) - a * sin(psi)
    J[0, 4] = -sin(psi + q1) * (l3 * cos(q2 + q3) + l2 * cos(q2))
    J[0, 5] = -cos(psi + q1) * (l3 * sin(q2 + q3) + l2 * sin(q2))
    J[0, 6] = -l3 * cos(psi + q1) * sin(q2 + q3)

    J[1, 0] = sin(psi)
    J[1, 1] = cos(psi)
    J[1, 2] = 0
    J[1, 3] = cos(psi + q1) * (l3 * cos(q2 + q3) + l2 * cos(q2)) + a * cos(psi) - b * sin(psi)
    J[1, 4] = cos(psi + q1) * (l3 * cos(q2 + q3) + l2 * cos(q2))
    J[1, 5] = -sin(psi + q1) * (l3 * sin(q2 + q3) + l2 * sin(q2))
    J[1, 6] = -l3 * sin(psi + q1) * sin(q2 + q3)

    J[2, 0] = 0
    J[2, 1] = 0
    J[2, 2] = 1
    J[2, 3] = 0
    J[2, 4] = 0
    J[2, 5] = -l3 * cos(q2 + q3) - l2 * cos(q2)
    J[2, 6] = -l3 * cos(q2 + q3)

    J[3, 0] = 0
    J[3, 1] = 0
    J[3, 2] = 0
    J[3, 3] = 1
    J[3, 4] = 0
    J[3, 5] = 0
    J[3, 6] = 0

    return J



def f_d(x, u, ts, f_sys):
    k1 = f_sys(x, u)
    k2 = f_sys(x+(ts/2)*k1, u)
    k3 = f_sys(x+(ts/2)*k2, u)
    k4 = f_sys(x+(ts)*k3, u)
    x = x + (ts/6)*(k1 +2*k2 +2*k3 +k4)

    
    num = x.size()[0]  # Obtener el número de componentes en x automáticamente
    aux_x = np.array(x[:,0]).reshape((num,))
    return aux_x

def QuatToRot(quat):
    # Quaternion to Rotational Matrix
    q = vertcat(*quat)  # Convierte la lista de cuaterniones en un objeto MX
    
    # Calcula la norma 2 del cuaternión
    q_norm = norm_2(q)
    
    # Normaliza el cuaternión dividiendo por su norma
    q_normalized = q / q_norm

    q_hat = MX.zeros(3, 3)

    q_hat[0, 1] = -q_normalized[3]
    q_hat[0, 2] = q_normalized[2]
    q_hat[1, 2] = -q_normalized[1]
    q_hat[1, 0] = q_normalized[3]
    q_hat[2, 0] = -q_normalized[2]
    q_hat[2, 1] = q_normalized[1]

    Rot = MX.eye(3) + 2 * q_hat @ q_hat + 2 * q_normalized[0] * q_hat

    return Rot




def euler_to_quaternion(roll, pitch, yaw):
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    return [qw, qx, qy, qz]

def FLUtoENU(u, quaternion):
    # Cuaterniones (qw, qx, qy, qz)
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    R_aux = R.from_quat([qx, qy, qz, qw])
    # Obtiene la matriz de rotación
    Rot = R_aux.as_matrix()
    v = Rot@u 

    return v

def quaternionMultiply(q1, q2):
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    
    scalarPart = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    vectorPart = vertcat(w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                         w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2)
    
    q_result = vertcat(scalarPart, vectorPart)
    return q_result

def quaternion_error(q_real, quat_d):
    norm_q = norm_2(q_real)
   
    
    q_inv = vertcat(q_real[0], -q_real[1], -q_real[2], -q_real[3]) / norm_q
    
    q_error = quaternionMultiply(q_inv, quat_d)
    return q_error


def log_cuaternion_casadi(q):
 

    # Descomponer el cuaternio en su parte escalar y vectorial
    q_w = q[0]
    q_v = q[1:]

    q = if_else(
        q_w < 0,
        -q,  # Si q_w es negativo, sustituir q por -q
        q    # Si q_w es positivo o cero, dejar q sin cambios
    )

    # Actualizar q_w y q_v después de cambiar q si es necesario
    q_w = q[0]
    q_v = q[1:]
    
    # Calcular la norma de la parte vectorial usando CasADi
    norm_q_v = norm_2(q_v)

    print(norm_q_v)
    
    # Calcular el ángulo theta
    theta = atan2(norm_q_v, q_w)
    
    log_q = 2 * q_v * theta / norm_q_v
    
    return log_q


def publish_matrix(matrix_data, topic_name='/nombre_del_topico'):
    """
    Publica una matriz en un tópico ROS.

    Args:
        matrix_data (numpy.ndarray): La matriz a publicar.
        topic_name (str): El nombre del tópico ROS en el que se publicará la matriz.
    """
    # Inicializa el nodo ROS si aún no está inicializado
   

    # Crea una instancia del mensaje Float64MultiArray
    matrix_msg = Float64MultiArray()

    # Convierte la matriz NumPy en una lista plana
    matrix_data_flat = matrix_data.flatten().tolist()

    # Asigna los datos de la matriz al mensaje
    matrix_msg.data = matrix_data_flat

    # Crea un publicador para el tópico deseado
    matrix_publisher = rospy.Publisher(topic_name, Float64MultiArray, queue_size=10)

    # Publica el mensaje en el tópico
    matrix_publisher.publish(matrix_msg)