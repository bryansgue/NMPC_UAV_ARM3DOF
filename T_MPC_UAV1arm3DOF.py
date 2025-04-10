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

import rospy
from scipy.spatial.transform import Rotation as R
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
#from c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython
from geometry_msgs.msg import TwistStamped
import math
from scipy.io import savemat
import os

# CARGA FUNCIONES DEL PROGRAMA
from fancy_plots import plot_pose, plot_error, plot_time
from Functions_SimpleModel import f_system_simple_model
from Functions_SimpleModel import f_d, odometry_call_back, get_odometry_simple, send_velocity_control, pub_odometry_sim
import P_UAV_simple


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

def create_ocp_solver_description(x0, N_horizon, t_horizon, zp_max, zp_min, phi_max, phi_min, theta_max, theta_min, psi_max, psi_min) -> AcadosOcp:
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    model, f_system = f_system_simple_model()
    ocp.model = model
    ocp.p = model.p
    nx = model.x.size()[0]
    nu = model.u.size()[0]
    ny = nx + nu

    # set dimensions
    ocp.dims.N = N_horizon

    Q_mat = MX.zeros(4, 4)
    Q_mat[0, 0] = 1
    Q_mat[1, 1] = 1
    Q_mat[2, 2] = 1
    Q_mat[3, 3] = 1

    R_mat = MX.zeros(7, 7)
    R_mat[0, 0] = 0.1
    R_mat[1, 1] = 0.1
    R_mat[2, 2] = 0.1
    R_mat[3, 3] = 0.1
    R_mat[4, 4] = 0.1
    R_mat[5, 5] = 0.1
    R_mat[6, 6] = 0.1
    
    ocp.parameter_values = np.zeros(ny)

    ocp.cost.cost_type = "EXTERNAL"
    ocp.cost.cost_type_e = "EXTERNAL"

    error_pose = ocp.p[0:4] - model.x[0:4]
    ocp.model.cost_expr_ext_cost = error_pose.T @ Q_mat @error_pose  + model.u.T @ R_mat @ model.u 
    ocp.model.cost_expr_ext_cost_e = error_pose.T @ Q_mat @ error_pose

    # set constraints
    # Valores en grados
    lbx_degrees = np.array([-30, 15, -45]) #min
    ubx_degrees = np.array([30, 90, 0]) #MAX

    # Convertir de grados a radianes
    lbx_radians = lbx_degrees * np.pi / 180
    ubx_radians = ubx_degrees * np.pi / 180

    # Asignar los valores convertidos a las restricciones
    ocp.constraints.lbx = lbx_radians
    ocp.constraints.ubx = ubx_radians
    ocp.constraints.idxbx = np.array([7, 8, 9])


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
    x = np.zeros((17, t.shape[0]+1-N_prediction), dtype = np.double)
    x_sim = np.zeros((8, t.shape[0]+1-N_prediction), dtype = np.double)



    #TAREA DESEADA
    value = 5
    xd = lambda t: 4 * np.sin(value*0.04*t) + 3
    yd = lambda t: 4 * np.sin(value*0.08*t)
    zd = lambda t: 2 * np.sin(value*0.08*t) + 6
    xdp = lambda t: 4 * value * 0.04 * np.cos(value*0.04*t)
    ydp = lambda t: 4 * value * 0.08 * np.cos(value*0.08*t)
    zdp = lambda t: 2 * value * 0.08 * np.cos(value*0.08*t)

    hxd = xd(t)
    hyd = yd(t)
    hzd = zd(t)
    hxdp = xdp(t)
    hydp = ydp(t)
    hzdp = zdp(t)

    psid = np.arctan2(hydp, hxdp)
    psidp = np.gradient(psid, t_s)

    # Reference Signal of the system
    ref = np.zeros((12, t.shape[0]), dtype = np.double)
    ref[0,:] = hxd 
    ref[1,:] = hyd
    ref[2,:] = hzd  
    ref[3,:] = psid 
    ref[4,:] = 0
    ref[5,:] = 0 
    ref[6,:] = 0 
    ref[7,:] = 0 
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

    # Llamar a la función para calcular la posición del extremo del brazo
    hx_0, hy_0, hz_0 = CDBrazo3DOF(px_0, py_0, pz_0, psi_0, l1, l2, l3, a, b, c, q1_0, q2_0, q3_0)

    #INICIALIZA LECTURA DE ODOMETRIA
    for k in range(0, 10):
        # Read Real data
        x[:, 0] = [hx_0,hy_0,hz_0,psi_0,px_0 ,py_0 ,pz_0,q1_0,q2_0,q3_0,0,0,0,0,0,0,0]
        # Loop_rate.sleep()
        rate.sleep() 
        print("Init System")

    # Create Optimal problem
    model, f = f_system_simple_model()

    ocp = create_ocp_solver_description(x[:,0], N_prediction, t_prediction, zp_ref_max, zp_ref_min, phi_max, phi_min, theta_max, theta_min, psi_max, psi_min)
    #acados_ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp_" + ocp.model.name + ".json", build= True, generate= True)

    solver_json = 'acados_ocp_' + model.name + '.json'
    AcadosOcpSolver.generate(ocp, json_file=solver_json)
    AcadosOcpSolver.build(ocp.code_export_directory, with_cython=True)
    acados_ocp_solver = AcadosOcpSolver.create_cython_solver(solver_json)
    #acados_ocp_solver = AcadosOcpSolverCython(ocp.model.name, ocp.solver_options.nlp_solver_type, ocp.dims.N)

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
        values = [0,0,0,0,0,0,0,0,0,0,0,0]

        for j in range(N_prediction):
            yref = ref[:,k+j]
            acados_ocp_solver.set(j, "p", np.hstack([yref, values]))

        yref_N = ref[:,k+N_prediction]
        acados_ocp_solver.set(N_prediction, "p",  np.hstack([yref_N, values]))
        
        
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
    pwd = "/home/bryansgue/Doctoral_Research/Matlab/Aerial_Manipulator"

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
            'time': t,})

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