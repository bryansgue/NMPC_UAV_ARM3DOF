
## 🛰️ NMPC for UAV with 3-DOF Manipulator Arm

<img src="NMPC_UAV_3DOFarm.gif" width="600"/>


This repository contains Python code for simulating and evaluating a **Nonlinear Model Predictive Control (NMPC)** framework applied to a **UAV equipped with a 3-DOF robotic arm**, using **CasADi 3.5.5** and **ACADOS**.

The control approach integrates a **data-driven model (DMDc)** for high-dimensional system dynamics and supports two formulations for attitude representation:
- **Quaternions**: handled in `T_MPC_UAV1arm3DOF_quat.py`
- **Euler angles**: handled in `T_MPC_UAV1arm3DOF.py`

---

## 📁 Project Structure

```bash

├── T_MPC_UAV1arm3DOF_quat.py         # Main NMPC script with quaternion-based attitude control
├── T_MPC_UAV1arm3DOF.py              # NMPC variant using Euler angles
├── mpcc_uav_manipulator.py.py        # MPC configuration and class definition
├── Functions_SimpleModel.py          # Basic UAV-manipulator model functions
├── Functions_DinamicControl.py       # NMPC cost functions and dynamic setup
├── P_UAV_simple.py                   # UAV dynamics and parameters
├── fancy_plots.py                    # Custom plotting utilities
├── acados_ocp_Drone_ode.json         # ACADOS OCP configuration
├── c_generated_code/                 # ACADOS auto-generated C code
├── 1_pose.png / 2_error_pose.png / 3_Time.png  # Sample figures
├── NMPC_UAV_3DOFarm.gif              # Animation of trajectory tracking
└── README.md                         # This documentation file
```


## ⚙️ Requirements

Make sure you have the following installed:

- Python 3.7+
- [CasADi 3.5.5](https://web.casadi.org/)
- [ACADOS](https://github.com/acados/acados) (compiled and available in your environment)
- NumPy, Matplotlib, SciPy
- (Optional) LaTeX for plotting with fancy labels

You can install Python dependencies via:

```bash
pip install casadi==3.5.5 numpy matplotlib scipy
```

---

## 🚀 How to Run

### Quaternion-based NMPC:
```bash
python3 T_MPC_UAV1arm3DOF_quat.py
```

### Euler-based NMPC:
```bash
python3 T_MPC_UAV1arm3DOF.py
```

Output includes:
- Trajectory tracking performance
- Attitude and joint errors
- Real-time simulation visualizations

---

## 🧠 Control Highlights

- Full-body model combining UAV and manipulator
- Uses **Dynamic Mode Decomposition with control (DMDc)** to identify a linear approximation for NMPC
- Attitude represented via **quaternions on Lie groups**, mapped to tangent space via logarithmic map
- Adaptive cost weighting strategy for balancing primary (trajectory) and secondary (internal configuration) objectives
- Integrated with **ACADOS** for real-time optimal control

---

## 📈 Visualization

Use `fancy_plots.py` to generate high-quality figures:

```bash
python3 fancy_plots.py
```

This script can recreate figures like:
- Position vs. reference
- Error convergence
- Joint trajectories

---

## 📬 Contact

> Developed as part of doctoral research by **Bryan S. Guevara** and collaborators.  
For questions or collaboration, please contact:  
📧 `bryansgue@gmail.com`

---

## 📜 License

This project is open-source and available under the MIT License.
```


