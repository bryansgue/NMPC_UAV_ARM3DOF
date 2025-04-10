from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
from acados_template import AcadosModel
import scipy.linalg
import numpy as np
import time
import matplotlib.pyplot as plt
from casadi import Function
from casadi import MX
from casadi import reshape
from casadi import vertcat
from casadi import horzcat
from casadi import cos
from casadi import sin
from casadi import solve
from casadi import inv
from casadi import mtimes
from casadi import jacobian
from casadi import atan2
from scipy.optimize import bisect
from scipy.integrate import quad
import rospy
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
#from c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython
from geometry_msgs.msg import TwistStamped
import math
from scipy.io import savemat
import os
from scipy.interpolate import CubicSpline

# CARGA FUNCIONES DEL PROGRAMA
from fancy_plots import plot_pose, plot_error, plot_time
from Functions_SimpleModel import quaternion_error, log_cuaternion_casadi, euler_to_quaternion
from Functions_SimpleModel import f_d, odometry_call_back, get_odometry_simple, send_velocity_control, pub_odometry_sim
import P_UAV_simple

# Definir el valor global
value = 3

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

    vx =  MX.sym('vx')
    vy =  MX.sym('vy')
    vz =  MX.sym('vz')

    
    p = vertcat(nx_d, ny_d, nz_d, psi_d, qw_d, qx_d, qy_d, qz_d, px_d, py_d, pz_d, q1_d, q2_d, q3_d, ul_d, um_d, un_d, w_d, q1_p_d ,q2_p_d ,q3_p_d , ul_ref_d, um_ref_d, un_ref_d, w_ref_d, q1_p_ref_d , q2_p_ref_d , q3_p_ref_d, alpha , beta, vx, vy, vz )

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


    return model, f_system, f_x, g_x


def create_ocp_solver_description(x0, N_horizon, t_horizon, zp_max, zp_min, phi_max, phi_min, theta_max, theta_min, psi_max, psi_min) -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model, f_system,  f_x, g_x = f_system_simple_model_quat()
    ocp.model = model
    ocp.p = model.p
    nx = model.x.size()[0]
    nu = model.u.size()[0]

    var_aditional = 5
    ny = nx + nu + var_aditional

    # set dimensions
    ocp.dims.N = N_horizon

    Q_mat = MX.zeros(3, 3)
    Q_mat[0, 0] = 1
    Q_mat[1, 1] = 1
    Q_mat[2, 2] = 1

    R_mat = MX.zeros(7, 7)
    R_mat[0, 0] = 0.05 # ul
    R_mat[1, 1] = 0.05 # um
    R_mat[2, 2] = 0.05 # un
    R_mat[3, 3] = 0.05 # uw
    R_mat[4, 4] = 0.5 # q1
    R_mat[5, 5] = 1 # q2
    R_mat[6, 6] = 1 # q3

    K_mat = MX.zeros(3, 3)
    K_mat[0, 0] = 1
    K_mat[1, 1] = 1
    K_mat[2, 2] = 1

    Q_sec = MX.zeros(3, 3)
    Q_sec[0, 0] = 0
    Q_sec[1, 1] = 0.05
    Q_sec[2, 2] = 0.05
    
    ocp.parameter_values = np.zeros(ny)

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    error_pose = ocp.p[0:3] - model.x[0:3]
    error_internal = ocp.p[11:14] - model.x[11:14]
   

    quat_error = quaternion_error(model.x[4:8], ocp.p[4:8])

    alpha = ocp.p[28]
    beta = ocp.p[29]


    sd = ocp.p[0:3]
    e_t = (sd - model.x[0:3])

    #Vector tangente
    sd_p = ocp.p[30:33]



    # Calcular Log(q)
    log_q = log_cuaternion_casadi(quat_error)

    error_h = alpha* error_pose.T @ Q_mat @error_pose
    Error_q = beta *error_internal.T @ Q_sec @ error_internal
    actitud_cost = log_q.T @ K_mat @ log_q 


    

    ocp.model.cost_expr_ext_cost = error_h  + Error_q  + model.u.T @ R_mat @ model.u + actitud_cost
    ocp.model.cost_expr_ext_cost_e = error_h  + Error_q  + actitud_cost

    # set constraints
    # Valores en grados
    lbx_degrees = np.array([-15, 15, -70]) #min
    ubx_degrees = np.array([15, 90, 0]) #MAX

    # Convertir de grados a radianes
    lbx_radians = lbx_degrees * np.pi / 180
    ubx_radians = ubx_degrees * np.pi / 180

    # Asignar los valores convertidos a las restricciones
    ocp.constraints.lbx = lbx_radians
    ocp.constraints.ubx = ubx_radians
    ocp.constraints.idxbx = np.array([11, 12, 13])


    ocp.constraints.x0 = x0

    # set options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"  # 'GAUSS_NEWTON', 'EXACT'
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"  # SQP_RTI, SQP
    
    #ocp.solver_options.tol = 1e-4

    # set prediction horizon
    ocp.solver_options.tf = t_horizon

    return ocp


def CDBrazo3DOF(xu, yu, zu, psi, l1, l2, l3, a, b, c, q1, q2, q3):
    hx = xu + a * np.cos(psi) - b * np.sin(psi) + np.cos(psi + q1) * (l3 * np.cos(q2 + q3) + l2 * np.cos(q2))
    hy = yu + a * np.sin(psi) + b * np.cos(psi) + np.sin(psi + q1) * (l3 * np.cos(q2 + q3) + l2 * np.cos(q2))
    hz = zu + c - l1 - l3 * np.sin(q2 + q3) - l2 * np.sin(q2)

    cinematica3DOF = np.array([hx, hy, hz])
    return cinematica3DOF


# Parameters for adaptive weights
alpha_0 = 1
k_alpha = 1

beta_0 = 1
k_beta = 2

def compute_curvature(h_d, t_s):
    """
    Compute the curvature of the desired trajectory.
    
    Args:
        h_d (numpy.ndarray): Desired trajectory positions (3xN array for x, y, z).
        t_s (float): Time step.

    Returns:
        numpy.ndarray: Curvature values over time.
    """
    # Compute velocity and acceleration numerically
    h_dot = np.gradient(h_d, t_s, axis=1)
    h_ddot = np.gradient(h_dot, t_s, axis=1)
    
    # Cross product of velocity and acceleration
    cross_product = np.cross(h_dot.T, h_ddot.T).T
    
    # Compute curvature
    curvature = np.linalg.norm(cross_product, axis=0) / (np.linalg.norm(h_dot, axis=0)**3 + 1e-8)
    return curvature

def adaptive_weights(curvature):
    """
    Compute adaptive weights alpha and beta based on curvature.

    Args:
        curvature (numpy.ndarray): Curvature values over time.

    Returns:
        tuple: Arrays for alpha and beta over time.
    """
    # Alpha centered around 1, small variation
    alpha = alpha_0 * (1 + (k_alpha / (1 + k_alpha)) * (curvature - 1))
    
    # Beta scales between 0 and 1
    beta = beta_0 * (1 - np.exp(-k_beta * curvature))

    return alpha, beta


def manage_ocp_solver(model, ocp):
    """
    Maneja la creación o uso del solver OCP.

    Args:
        model: El modelo utilizado para definir el solver.
        ocp: La descripción del OCP a gestionar.

    Returns:
        acados_ocp_solver: La instancia del solver creado o cargado.
    """
    # Definir el nombre del archivo JSON
    solver_json = 'acados_ocp_' + model.name + '.json'

    # Comprobar si el archivo JSON del solver ya existe
    if os.path.exists(solver_json):
        # Preguntar al usuario qué desea hacer
        respuesta = input(f"El solver {solver_json} ya existe. ¿Deseas usar el existente (U) o generarlo nuevamente (G)? [U/G]: ").strip().upper()
        
        if respuesta == 'U':
            print(f"Usando el solver existente: {solver_json}")
            # Crear el solver directamente desde el archivo existente
            return AcadosOcpSolver.create_cython_solver(solver_json)
        elif respuesta == 'G':
            print(f"Regenerando y reconstruyendo el solver: {solver_json}")
            # Generar y construir el solver
            AcadosOcpSolver.generate(ocp, json_file=solver_json)
            AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
            return AcadosOcpSolver.create_cython_solver(solver_json)
        else:
            raise ValueError("Opción no válida. No se realizó ninguna acción.")
    else:
        print(f"El solver {solver_json} no existe. Generando y construyendo...")
        # Generar y construir el solver
        AcadosOcpSolver.generate(ocp, json_file=solver_json)
        AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
        return AcadosOcpSolver.create_cython_solver(solver_json)


def trayectoria(t):

    def xd(t):
        return 7 * np.sin(value * 0.04 * t) + 3

    def yd(t):
        return 7 * np.sin(value * 0.08 * t)

    def zd(t):
        return 1.5 * np.sin(value * 0.08 * t) + 6

    def xd_p(t):
        return 7 * value * 0.04 * np.cos(value * 0.04 * t)

    def yd_p(t):
        return 7 * value * 0.08 * np.cos(value * 0.08 * t)

    def zd_p(t):
        return 1.5 * value * 0.08 * np.cos(value * 0.08 * t)

    return xd, yd, zd, xd_p, yd_p, zd_p

def calculate_positions_and_arc_length(xd, yd, zd, xd_p, yd_p, zd_p, t_range, t_max):

    
    def r(t):
        """ Devuelve el punto en la trayectoria para el parámetro t usando las funciones de trayectoria. """
        return np.array([xd(t), yd(t), zd(t)])

    def r_prime(t):
        """ Devuelve la derivada de la trayectoria en el parámetro t usando las derivadas de las funciones de trayectoria. """
        return np.array([xd_p(t), yd_p(t), zd_p(t)])

    def integrand(t):
        """ Devuelve la norma de la derivada de la trayectoria en el parámetro t. """
        return np.linalg.norm(r_prime(t))

    def arc_length(tk, t0=0):
        """ Calcula la longitud de arco desde t0 hasta tk usando las derivadas de la trayectoria. """
        length1, _ = quad(integrand, t0, (t0 + tk) / 2, limit=50)
        length2, _ = quad(integrand, (t0 + tk) / 2, tk, limit=50)
        length = length1 + length2
        length, _ = quad(integrand, t0, tk, limit=100)
        return length

    def find_t_for_length(theta, t0=0):
        """ Encuentra el parámetro t que corresponde a una longitud de arco theta. """
        func = lambda t: arc_length(t, t0) - theta
        return bisect(func, t0, t_max)

    # Generar las posiciones y longitudes de arco
    positions = []
    arc_lengths = []
    
    for tk in t_range:
        theta = arc_length(tk)
        arc_lengths.append(theta)
        point = r(tk)
        positions.append(point)

    arc_lengths = np.array(arc_lengths)
    positions = np.array(positions).T  # Convertir a array 2D (3, N)

    # Crear splines cúbicos para la longitud de arco con respecto al tiempo
    spline_t = CubicSpline(arc_lengths, t_range)
    spline_x = CubicSpline(t_range, positions[0, :])
    spline_y = CubicSpline(t_range, positions[1, :])
    spline_z = CubicSpline(t_range, positions[2, :])

    # Función que retorna la posición dado un valor de longitud de arco
    def position_by_arc_length(s):
        t_estimated = spline_t(s)  # Usar spline para obtener la estimación precisa de t
        return np.array([spline_x(t_estimated), spline_y(t_estimated), spline_z(t_estimated)])

    return arc_lengths, positions, position_by_arc_length


def main(vel_pub, vel_msg, odom_sim_pub, odom_sim_msg):
    # Initial Values System
    # Simulation Time
    t_final = 30
    # Sample time
    frec= 30
    t_s = 1/frec
    # Prediction Time
    N_horizont = frec
    t_prediction = N_horizont/frec

    # Nodes inside MPC
    N = np.arange(0, t_prediction + t_s, t_s)
    N_prediction = N.shape[0]

    # Time simulation
    t = np.arange(0, t_final + t_s, t_s)

    # Sample time vector
    delta_t = np.zeros((1, t.shape[0] - N_prediction), dtype=np.double)
    t_sample = t_s*np.ones((1, t.shape[0] - N_prediction), dtype=np.double)

    # Vector Initial conditions
    x = np.zeros((21, t.shape[0]+1-N_prediction), dtype = np.double)
    x_sim = np.zeros((8, t.shape[0]+1-N_prediction), dtype = np.double)


    # Longitud total del eslabón
    l1 = 0.15 
    l2 = 0.44
    l3 = 0.45

    L = l1 + l2 + l3  # Asegúrate de definir l1, l2 y l3 antes
    # Factor de aumento de frecuencia
    freq_multiplier = 4  # Cambia este valor para ajustar la frecuencia (2 es el doble de la frecuencia original)

    

    # Obtener las funciones de trayectoria y sus derivadas
    xd, yd, zd, xd_p, yd_p, zd_p = trayectoria(t)
    # Calcular posiciones parametrizadas en longitud de arco
    t_finer = np.linspace(0, t_final, len(t))  # Duplicar el tiempo y generar más puntos

    arc_lengths, pos_ref, position_by_arc_length= calculate_positions_and_arc_length(xd, yd, zd, xd_p, yd_p, zd_p, t_finer , t_max=t_final)
    dp_ds = np.gradient(pos_ref, arc_lengths, axis=1)

    # Evaluar las derivadas en cada instante
    xd_p_vals = xd_p(t)
    yd_p_vals = yd_p(t)
    

    # Calcular psid y su derivada
    psid = np.arctan2(yd_p_vals, xd_p_vals)
    psidp = np.gradient(psid, t_s)

    #quaternion = euler_to_quaternion(0, 0, psid) 
    quatd= np.zeros((4, t.shape[0]), dtype = np.double)


    # Calcular los cuaterniones utilizando la función euler_to_quaternion para cada psid
    for i in range(t.shape[0]):
        quaternion = euler_to_quaternion(0, 0, psid[i])  # Calcula el cuaternión para el ángulo de cabeceo en el instante i
        quatd[:, i] = quaternion  # Almacena el cuaternión en la columna i de 'quatd'


    # Reference Signal of the system
    ref = np.zeros((28, t.shape[0]), dtype = np.double)
    ref[0,:] = pos_ref[0, :]  # px_d 
    ref[1,:] = pos_ref[1, :]  # py_d
    ref[2,:] = pos_ref[2, :]  # pz_d  
    ref[3,:] = 0
    ref[4,:] = quatd[0, :] 
    ref[5,:] = quatd[1, :]  
    ref[6,:] = quatd[2, :]  
    ref[7,:] = quatd[3, :]  

    ref[11,:] = np.deg2rad(0)
    ref[12,:] = np.deg2rad(60)
    ref[13,:] =  np.deg2rad(-60)

    
    
    # Initial Control values
    u_control = np.zeros((7, t.shape[0]-N_prediction), dtype = np.double)
    #u_control = np.zeros((4, t.shape[0]), dtype = np.double)

    # Limits Control values
    zp_ref_max = 3
    phi_max = 3
    theta_max = 3
    psi_max = 2

    zp_ref_min = -zp_ref_max
    phi_min = -phi_max
    theta_min = -theta_max
    psi_min = -psi_max
    
    # Simulation System
    ros_rate = 30  # Tasa de ROS en Hz
    rate = rospy.Rate(ros_rate)  # Crear un objeto de la clase rospy.Rate

    #P_UAV_simple.main(vel_pub, vel_msg )

    # Condiciones Iniciales
    px_0 = 0
    py_0 = 0 
    pz_0 = 5
    psi_0 = np.deg2rad(30)  # orientación psi en radianes (ejemplo de 30 grados)
    l1 = 0.15 
    l2 = 0.44 
    l3 = 0.45
    a = 0.0  # parámetro a del modelo
    b = 0.0  # parámetro b del modelo
    c = -0.03
    q1_0 = np.deg2rad(0)  # ángulo q1 en radianes (ejemplo de 45 grados)
    q2_0 = np.deg2rad(60)  # ángulo q2 en radianes (ejemplo de 60 grados)
    q3_0 = np.deg2rad(-15)  # ángulo q3 en radianes (ejemplo de 30 grados)

    qw,qx,qy,qz = euler_to_quaternion(0,0,psi_0 )

    # Llamar a la función para calcular la posición del extremo del brazo
    hx_0, hy_0, hz_0 = CDBrazo3DOF(px_0, py_0, pz_0, psi_0, l1, l2, l3, a, b, c, q1_0, q2_0, q3_0)

    #INICIALIZA LECTURA DE ODOMETRIA
    for k in range(0, 10):
        # Read Real data
        x[:, 0] = [hx_0,hy_0,hz_0,psi_0, qw, qx ,qy ,qz, px_0 ,py_0 ,pz_0,q1_0,q2_0,q3_0,0,0,0,0,0,0,0]
        # Loop_rate.sleep()
        rate.sleep() 
        print("Init System")

    # Create Optimal problem
    model, f, f_x, g_x = f_system_simple_model_quat()

    ocp = create_ocp_solver_description(x[:,0], N_prediction, t_prediction, zp_ref_max, zp_ref_min, phi_max, phi_min, theta_max, theta_min, psi_max, psi_min)
    
    acados_ocp_solver = manage_ocp_solver(model, ocp)

    nx = ocp.model.x.size()[0]
    nu = ocp.model.u.size()[0]

    simX = np.ndarray((nx, N_prediction+1))
    simU = np.ndarray((nu, N_prediction))

    # Initial States Acados
    for stage in range(N_prediction + 1):
        acados_ocp_solver.set(stage, "x", x[:, 0])
    for stage in range(N_prediction):
        acados_ocp_solver.set(stage, "u", np.zeros((nu,)))

    # Errors of the system
    Error = np.zeros((3, t.shape[0]-N_prediction), dtype = np.double)

    print("HERE OK")
    time.sleep(0.1)  # Pausa el script por 5 segundos
    print("Pausa finalizada.")



    for k in range(0, t.shape[0]-N_prediction):
              
        Error[:,k] = ref[0:3, k] - x[0:3, k]

        # Control Law Section
        acados_ocp_solver.set(0, "lbx", x[:,k])
        acados_ocp_solver.set(0, "ubx", x[:,k])

        # SET REFERENCES
        #values = [alpha_vals[k],beta_vals[k]]
        values = [1,0.1, dp_ds [0, k], dp_ds [1, k], dp_ds [2, k]]


        for j in range(N_prediction):
            yref = ref[:,k+j]
            acados_ocp_solver.set(j, "p", np.hstack([yref,values ]))

        yref_N = ref[:,k+N_prediction]
        acados_ocp_solver.set(N_prediction, "p",  np.hstack([yref_N, values ]))
        
        
        # Get Computational Time
        tic = time.time()
        status = acados_ocp_solver.solve()
        toc_solver = time.time()- tic

        # get solution
        for i in range(N_prediction):
            simX[:,i] = acados_ocp_solver.get(i, "x")
            simU[:,i] = acados_ocp_solver.get(i, "u")
        simX[:,N_prediction] = acados_ocp_solver.get(N_prediction, "x")

        publish_matrix(simX[0:3, 0:N_prediction], '/Prediction')
        print(simX[0:3, 0:N_prediction].shape)

        u_control[:, k] = acados_ocp_solver.get(0, "u")
        #u_control[:, k] = [0.0, 0.0, 0, 00.1, 0.1, 0, 0]
        send_velocity_control(u_control[:, k], vel_pub, vel_msg)

        # System Evolution
        opcion = "Sim"  # Valor que quieres evaluar

        if opcion == "Real":
            x[:, k+1] = get_odometry_simple()
        elif opcion == "Sim":
            x[:, k+1] = f_d(x[:, k], u_control[:, k], t_s, f)
            #pub_odometry_sim(x[:, k+1], odom_sim_pub, odom_sim_msg)
        else:
            print("Opción no válida")
        
        delta_t[:, k] = toc_solver
        
        
        print("x:", " ".join("{:.2f}".format(value) for value in np.round(x[0:12, k], decimals=2)))
        
        rate.sleep() 
        toc = time.time() - tic 
        
        
    send_velocity_control([0, 0, 0, 0], vel_pub, vel_msg)

    fig1 = plot_pose(x, ref, t)
    fig1.savefig("1_pose.png")
    fig2 = plot_error(Error, t)
    fig2.savefig("2_error_pose.png")
    fig3 = plot_time(t_sample, delta_t , t)
    fig3.savefig("3_Time.png")

    #For MODEL TESTS
    #x_data = {"states_MPC": x, "label": "x"}
    #ref_data = {"ref_MPC": ref, "label": "ref"}
    #t_data = {"t_MPC": t, "label": "time"}

    # Ruta que deseas verificar
    # Ruta que deseas verificar
    pwd = "/home/bryansgue/Doctoral_Research/Matlab/MPCC_UAV_Manipulator"

    # Verificar si la ruta no existe
    if not os.path.exists(pwd) or not os.path.isdir(pwd):
        print(f"La ruta {pwd} no existe. Estableciendo la ruta local como pwd.")
        pwd = os.getcwd()  # Establece la ruta local como pwd

    #SELECCION DEL EXPERIMENTO
   
    experiment_number = 1
    name_file = "UAV_1Arm3DOF_" + str(experiment_number) + ".mat"
    
    save = True
    if save==True:
        savemat(os.path.join(pwd, name_file), {
            'x_states': x,
            'x_ref': ref,
            'time': t,
            'u_control': u_control})

    return None


    print(f'Mean iteration time with MLP Model: {1000*np.mean(delta_t):.1f}ms -- {1/np.mean(delta_t):.0f}Hz)')



if __name__ == '__main__':
    try:
        # Node Initialization
        rospy.init_node("Acados_controller",disable_signals=True, anonymous=True)

        # SUCRIBER
        velocity_subscriber = rospy.Subscriber("/dji_sdk/odometry", Odometry, odometry_call_back)
        
        # PUBLISHER
        velocity_message = TwistStamped()
        velocity_publisher = rospy.Publisher("/m100/velocityControl", TwistStamped, queue_size=10)

        odometry_sim_msg = Odometry()
        odom_sim_pub = rospy.Publisher('/dji_sdk/odometry', Odometry, queue_size=10)
    
        
    

        main(velocity_publisher, velocity_message, odom_sim_pub, odometry_sim_msg)
    except(rospy.ROSInterruptException, KeyboardInterrupt):
        print("\nError System")
        send_velocity_control([0, 0, 0, 0], velocity_publisher, velocity_message)
        pass
    else:
        print("Complete Execution")
        pass