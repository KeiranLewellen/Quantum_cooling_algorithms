import numpy as np
from numpy.linalg import norm
from numpy.random import choice
import matplotlib.pyplot as plt
import math
# from scipy.integrate import quad, dblquad, tplquad, nquad
# from scipy.linalg import expm


# Object definitions

s_z = np.array([[1, 0], [0, -1]])
s_x = np.array([[0, 1], [1, 0]])
s_y = np.array([[0, -1j], [1j, 0]])
I = np.array([[1, 0], [0, 1]])


# Functions for time evolution

def evolve_pauli(pauli, coeff, dt):
    return np.cos(coeff*dt)*I - 1j*np.sin(coeff*dt)*pauli


def kron_n(mat_list):
    mat = mat_list[0]
    for i in range(len(mat_list) - 1):
        mat = np.kron(mat, mat_list[i + 1])
    return mat


def U_dt_list(pauli_list, coeff, dt):
    N = len(pauli_list)
    I_list = []
    for i in range(N):
        I_list.append(I)
    I_N = kron_n(I_list)
    pauli_N = kron_n(pauli_list)
    U = np.cos(coeff*dt)*I_N - 1j*np.sin(coeff*dt)*pauli_N
    return U


def mul(mat_list):
    mat = mat_list[0]
    for i in range(len(mat_list)-1):
        mat = np.matmul(mat, mat_list[i + 1])
    return mat


# Functions for system and bath time evolution

def U_dt_sum_sys(mat, coeff, dt, N_B, N_S):
    U = np.diag(np.ones(2 ** (N_B + N_S)))
    for i in range(N_S):
        U_i_list = []
        U_i_record = []
        for j in range(N_S + N_B):
            U_i_list.append(I)
            U_i_record.append(0)
        U_i_list[N_B + i] = mat
        U_i_record[N_B + i] = 1
        # print(U_i_record)
        U_i = U_dt_list(U_i_list, coeff, dt)
        U = np.matmul(U, U_i)
    return U


def U_dt_sum_sys_double(mat1, mat2, diff, coeff, dt, N_B, N_S):
    U = np.diag(np.ones(2 ** (N_B + N_S)))
    for i in range(N_S):
        U_i_list = []
        U_i_record = []
        for j in range(N_B + N_S):
            U_i_list.append(I)
            U_i_record.append(0)
        U_i_list[N_B + i] = mat1
        U_i_list[N_B + ((i + diff) % N_S)] = mat2
        U_i_record[N_B + i] = 1
        U_i_record[N_B + ((i + diff) % N_S)] = 1
        # print(U_i_record)
        U_i = U_dt_list(U_i_list, coeff, dt)
        U = np.matmul(U, U_i)
    return U


def U_dt_sum_bath(mat, coeff, dt, N_B, N_S):
    U = np.diag(np.ones(2 ** (N_B + N_S)))
    for i in range(N_B):
        U_i_list = []
        U_i_record = []
        for j in range(N_B + N_S):
            U_i_list.append(I)
            U_i_record.append(0)
        U_i_list[i] = mat
        U_i_record[i] = 1
        # print(U_i_record)
        U_i = U_dt_list(U_i_list, coeff, dt)
        U = np.matmul(U, U_i)
    return U


def U_dt_sum_coupling(mat1, mat2, coeff, dt, N_B, N_S):
    U = np.diag(np.ones(2 ** (N_B + N_S)))
    for i in range(N_B):
        U_i_list = []
        U_i_record = []
        for j in range(N_B + N_S):
            U_i_list.append(I)
            U_i_record.append(0)
        U_i_list[N_B + i] = mat1
        U_i_list[i] = mat2
        U_i_record[N_B + i] = 1
        U_i_record[i] = 2
        # print(U_i_record)
        U_i = U_dt_list(U_i_list, coeff, dt)
        U = np.matmul(U, U_i)
    return U


# Defines functions that takes the quantum ising model to its ground state


def B(t, T, B_i, B_f):
    return B_i - ((B_i - B_f)*t)/T


def g(t, T, g_max):
    if t < T/4:
        return (g_max/(T/4))*t
    if t >= T/4 and t <= 3*T/4:
        return g_max
    if t > 3*T/4:
        return g_max*(1 - (t-3*T/4)/(T/4))


def evolve_ising_coupled(sys, dt, T, N_B, N_S, J, h_x):
    m = int(T/dt)
    omega = 2*np.pi/T
    sys_dt_i = sys
    sys_t_list = []
    U_sJ = U_dt_sum_sys_double(s_x, s_x, 1, -J/2, dt / 2, N_B, N_S)
    for time_step in range(m + 1):
        # print(time_step)
        t = time_step*dt
        U_sx = U_dt_sum_sys(s_z, (-h_x/2)*np.cos(omega*t), dt, N_B, N_S)
        U_dt = mul([U_sJ, U_sx, U_sJ])
        sys_dt_i = np.dot(U_dt, sys_dt_i)
        sys_t_list.append(sys_dt_i)
    return sys_t_list


# Defines functions to measure the system and calculate the energy

def measure_qubit(sys, k, N_B, N_S):
    if not N_B == 0:
        print("ERROR must have N_B = 0")
    block_num = 2 ** (k + 1)
    block_len = 2 ** (N_S - (k + 1))
    measured_zero = np.zeros(2**N_S, dtype=np.complex64)
    measured_one = np.zeros(2**N_S, dtype=np.complex64)
    state_list = np.zeros(2**N_S)
    for i in range(2**N_S):
        if int(math.floor(i/block_len)) % 2 == 0:
            measured_zero[i] = sys[i][0]
        if int(math.floor(i/block_len)) % 2 == 1:
            measured_one[i] = sys[i][0]
            state_list[i] = 1
    # print("k:", k)
    # print("state_list:", state_list)
    p_zero = 0
    p_one = 0
    for i in range(2 ** N_S):
        p_zero += norm(measured_zero[i]) ** 2
        p_one += norm(measured_one[i]) ** 2

    print("total probability: ", p_zero + p_one)
    if not p_zero == 0:
        measured_zero = measured_zero/np.sqrt(p_zero)
    if not p_one == 0:
        measured_one = measured_one/np.sqrt(p_one)
    return p_zero, measured_zero, measured_one


def measure_random(sys, p, N_B, N_S):
    measurement_arr = np.zeros(N_S)
    for k in range(N_S):
        measure_or_not = choice(2, p=[p, 1 - p])
        if measure_or_not == 0:
            p_zero, measured_zero, measured_one = measure_qubit(sys, k, N_B, N_S)
            measure_outcome = choice(2, p=[p_zero, 1 - p_zero])
            if measure_outcome == 0:
                sys = np.reshape(measured_zero, (len(measured_zero), 1))
                measurement_arr[k] = 0
            if measure_outcome == 0:
                sys = np.reshape(measured_one, (len(measured_one), 1))
                measurement_arr[k] = 1
        if measure_or_not == 1:
            measurement_arr[k] = -1
    return sys, measurement_arr


def evolve_ising_ground_state(sys, dt, T, N_B, N_S, J, h_x, p, steps):
    sys_step_list = [sys]
    measurement_list = []
    sys_step = sys
    for i in range(steps):
        print("Step " + str(i))
        sys_list = evolve_ising_coupled(sys_step, dt, T, N_B, N_S, J, h_x)
        sys_evolved = sys_list[-1]
        sys_step, measurement_arr = measure_random(sys_evolved, p, N_B, N_S)
        sys_step_list.append(sys_step)
        measurement_list.append(measurement_arr)
    measurement_list = np.stack(measurement_list)
    return sys_step_list, measurement_list


# Makes plot of the system's energy

def remove_dots(string):
    string_out = ""
    for i in string:
        if not i == ".":
            string_out += i
    return string_out


def plot(energy_list, title, name):
    trial_num = range(len(energy_list))
    fig, ax = plt.subplots()
    ax.plot(trial_num, energy_list)
    ax.set(xlabel="Number of cycles", ylabel="Energy of system qubits", title=title)
    fig.savefig("Graphs/" + name)
    with open("Graphs/Energy_list," + name + ".npy", 'wb') as f:
        np.savez(f, *energy_list)


def ground_state_energy(H):
    energies, vectors = np.linalg.eig(H)
    GS_energy = min(energies)
    return GS_energy


# Parameter definitions

# H = -J/2 sum s_x s_x - h_x/2 cos(omega t) sum s_z
# we measure each qubit in s_z basis with probability p

J = 1   # coefficient of s_x*s_x
h_x = 1    # coefficient of s_z

N_S = 5
run = 1

T = 6  # floquet period
dt = 0.06   # trotter step time
m = T/dt    # number of trotter steps per floquet period
steps = 50   # number of floquet periods evolved for

p = 0.5  # probability of measuring each qubit


N_B = 0  # This is always zero and we should clean it up to just remove it

# Defines system

sys = np.zeros(2 ** (N_B + N_S))
sys_start_num = 1
sys[sys_start_num] = 1
sys = np.array([sys]).T


# Evolves system
print("Start")


print("J = " + str(J))
print("h_x = " + str(h_x))


sys_step_list, measurement_list = evolve_ising_ground_state(sys, dt, T, N_B, N_S, J, h_x, p, steps)
print("_________________")


# sys_step_list is a list of the system value stored as a list of column vectors
# measurement list is an array where the rows are the measurements (-1 means no measurement occurred)


'''
with open("Wave_functions/Wave_function_Ising_model,NS_" + str(N_S) + ",NB_" + str(N_B) + ",T_" + remove_dots(str(T))
          + ",sys_start_" + str(sys_start_num) + ",run_" + str(run) + '.npy', 'wb') as f:
    np.savez(f, *sys_step_list)
'''
print("measurement outcomes: ", measurement_list)
print("final state:", sys_step_list[-1])

'''
plot(energy_list, "Demagnetization of Ising model: N_S = " + str(N_S) + ", N_B = " + str(N_B) + ", T = " + str(T),
     "Demagnetization_of_Ising_model,NS_" + str(N_S) + ",NB_" + str(N_B) + ",J_" + remove_dots(str(J)) +
     ",h_x_" + remove_dots(str(h_x)) + ",h_z_" + remove_dots(str(h_z)) + ",sys_start_" + str(sys_start_num) + ",run_" + str(run))
'''





