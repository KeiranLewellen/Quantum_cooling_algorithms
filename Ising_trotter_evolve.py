import numpy as np
from numpy.linalg import norm
from numpy.random import choice
import matplotlib.pyplot as plt
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


def evolve_ising_coupled(sys, dt, T, N_B, N_S, J, h_x, h_z, g, g_max, B, B_i, B_f):
    m = int(T/dt)
    sys_dt_i = sys
    sys_t_list = []
    bath_energy_list = []
    total_energy_list = []
    sys_energy_list = []
    int_energy_list = []
    U_sz = U_dt_sum_sys(s_z, -h_z, dt/2, N_B, N_S)
    U_sx = U_dt_sum_sys(s_x, -h_x, dt, N_B, N_S)
    U_sJ = U_dt_sum_sys_double(s_z, s_z, 1, -J, dt / 2, N_B, N_S)
    H_S = Hamiltonian(N_B, N_S, J, h_x, h_z)
    for time_step in range(m + 1):
        # print(time_step)
        t = time_step*dt
        H_B = Hamiltonian_bath(N_B, N_S, B, t, T, B_i, B_f)
        H_int = Hamiltonian_int(N_B, N_S, g, t, T, g_max)
        H = H_B + H_S + H_int
        U_bB = U_dt_sum_bath(s_z, -B(t, T, B_i, B_f), dt/2, N_B, N_S)
        U_c = U_dt_sum_coupling(s_y, s_y, g(t, T, g_max), dt, N_B, N_S)
        U_dt = mul([U_sJ, U_sz, U_bB, U_c, U_sx, U_bB, U_sz, U_sJ])
        sys_dt_i = np.dot(U_dt, sys_dt_i)
        sys_t_list.append(sys_dt_i)
        bath_energy = energy([sys_dt_i], H_B)[0]
        sys_energy = energy([sys_dt_i], H_S)[0]
        int_energy = energy([sys_dt_i], H_int)[0]
        total_energy = energy([sys_dt_i], H_S + H_B + H_int)[0]
        bath_energy_list.append(bath_energy)
        sys_energy_list.append(sys_energy)
        int_energy_list.append(int_energy)
        total_energy_list.append(total_energy)
    return sys_t_list, bath_energy_list, sys_energy_list, int_energy_list, total_energy_list


# Defines functions to measure the system and calculate the energy

def ket(num, N):
    ket = str(bin(num))[2:]
    for i in range(N - len(ket)):
        ket = "0" + ket
    ket = "|" + ket + ">"
    return ket


def prob_list_bath(sys, N_B, N_S):
    prob_list = []
    for i in range(2 ** N_B):
        prob = 0
        for j in range(2 ** N_S):
            prob += norm(sys[(2 ** N_S)*i + j]) ** 2
        prob_list.append(prob)
    return prob_list


def measure_bath(sys, N_B, N_S):
    prob_list = prob_list_bath(sys, N_B, N_S)
    print(prob_list)
    measure_num = choice(2 ** N_B, p=prob_list)
    measurement = ket(measure_num, N_B)
    sys_measured = sys[measure_num*(2 ** N_S): (measure_num + 1)*(2 ** N_S)]
    sys_measured = np.concatenate((sys_measured, np.array([np.zeros(((2**N_B)-1)*(2 ** N_S))]).T), axis=0)
    sys_measured = (prob_list[measure_num] ** (-0.5))*sys_measured
    # print(sys[-8:-1])
    # print(sys[0:8])
    # print(sys_measured[0:8])
    print(measurement)
    return measurement, sys_measured


def evolve_ising_ground_state(sys, dt, T, N_B, N_S, J, h_x, h_z, g, g_max, B, B_i, B_f, steps):
    sys_step_list = [sys]
    measurement_list = []
    sys_step = sys
    energy_sys = []
    energy_bath = []
    energy_tot = []
    energy_int = []
    H = Hamiltonian(N_B, N_S, J, h_x, h_z)
    print(energy([sys], H)[0])
    for i in range(steps):
        print("Step " + str(i))
        sys_list, bath_energy_list, sys_energy_list, int_energy_list, total_energy_list = evolve_ising_coupled(sys_step, dt, T, N_B, N_S, J, h_x, h_z, g, g_max, B, B_i, B_f)
        energy_sys.append(sys_energy_list)
        energy_bath.append(bath_energy_list)
        energy_int.append(int_energy_list)
        energy_tot.append(total_energy_list)
        sys_evolved = sys_list[-1]
        measurement, sys_step = measure_bath(sys_evolved, N_B, N_S)
        sys_step_list.append(sys_step)
        print(energy([sys_step], H)[0])
        measurement_list.append(measurement)
    return sys_step_list, measurement_list, energy_sys, energy_bath, energy_int, energy_tot


# Defines the Hamiltonian of the system and bath as well as Hamiltonians of parts of the system

def Hamiltonian(N_B, N_S, J, h_x, h_z):
    H = np.diag(np.zeros(2 ** (N_B + N_S)))
    for i in range(N_S):
        H_i_list = []
        H_i_record = []
        for j in range(N_S + N_B):
            H_i_list.append(I)
            H_i_record.append(0)
        H_i_list[N_B + i] = s_z
        H_i_record[N_B + i] = -h_z
        # print(H_i_record)
        H_i = -h_z*kron_n(H_i_list)
        H += H_i
    for i in range(N_S):
        H_i_list = []
        H_i_record = []
        for j in range(N_S + N_B):
            H_i_list.append(I)
            H_i_record.append(0)
        H_i_list[N_B + i] = s_x
        H_i_record[N_B + i] = -h_x
        # print(H_i_record)
        H_i = -h_x*kron_n(H_i_list)
        H += H_i
    for i in range(N_S):
        H_i_list = []
        H_i_record = []
        for j in range(N_B + N_S):
            H_i_list.append(I)
            H_i_record.append(0)
        H_i_list[N_B + i] = s_z
        H_i_list[N_B + ((i + 1) % N_S)] = s_z
        H_i_record[N_B + i] = -J
        H_i_record[N_B + ((i + 1) % N_S)] = -J
        # print(H_i_record)
        H_i = -J * kron_n(H_i_list)
        H += H_i
    return H


def Hamiltonian_z(N_B, N_S, J, h_x, h_z):
    H = np.diag(np.zeros(2 ** (N_B + N_S)))
    for i in range(N_S):
        H_i_list = []
        H_i_record = []
        for j in range(N_S + N_B):
            H_i_list.append(I)
            H_i_record.append(0)
        H_i_list[N_B + i] = s_z
        H_i_record[N_B + i] = -h_z
        # print(H_i_record)
        H_i = -h_z*kron_n(H_i_list)
        H += H_i
    return H


def Hamiltonian_x(N_B, N_S, J, h_x, h_z):
    H = np.diag(np.zeros(2 ** (N_B + N_S)))
    for i in range(N_S):
        H_i_list = []
        H_i_record = []
        for j in range(N_S + N_B):
            H_i_list.append(I)
            H_i_record.append(0)
        H_i_list[N_B + i] = s_x
        H_i_record[N_B + i] = -h_x
        # print(H_i_record)
        H_i = -h_x*kron_n(H_i_list)
        H += H_i
    return H


def Hamiltonian_J(N_B, N_S, J, h_x, h_z):
    H = np.diag(np.zeros(2 ** (N_B + N_S)))
    for i in range(N_S):
        H_i_list = []
        H_i_record = []
        for j in range(N_B + N_S):
            H_i_list.append(I)
            H_i_record.append(0)
        H_i_list[N_B + i] = s_z
        H_i_list[N_B + ((i + 1) % N_S)] = s_z
        H_i_record[N_B + i] = -J
        H_i_record[N_B + ((i + 1) % N_S)] = -J
        # print(H_i_record)
        H_i = -J * kron_n(H_i_list)
        H += H_i
    return H


def Hamiltonian_bath(N_B, N_S, B, t, T, B_i, B_f):
    H = np.diag(np.zeros(2 ** (N_B + N_S)))
    B = B(t, T, B_i, B_f)
    for i in range(N_B):
        H_i_list = []
        H_i_record = []
        for j in range(N_S + N_B):
            H_i_list.append(I)
            H_i_record.append(0)
        H_i_list[i] = s_z
        H_i_record[i] = -B
        # print(H_i_record)
        H_i = -B*kron_n(H_i_list)
        H += H_i
    # print(H)
    return H


def Hamiltonian_int(N_B, N_S, g, t, T, g_max):
    H = np.diag(np.zeros(2 ** (N_B + N_S)).view(np.complex64))
    g = g(t, T, g_max)
    for i in range(N_B):
        H_i_list = []
        H_i_record = []
        for j in range(N_S + N_B):
            H_i_list.append(I)
            H_i_record.append(0)
        H_i_list[i] = s_y
        H_i_record[i] = -g
        H_i_list[N_B + i] = s_y
        H_i_record[N_B + i] = -g
        # print(H_i_record)
        H_i = -g*kron_n(H_i_list)
        H += H_i
    # print(H)
    return H


def energy(sys_list, H):
    energy_list = []
    for sys in sys_list:
        energy = mul([sys.T.conjugate(), H, sys])
        if energy.imag > 0.00001:
            return False
        else:
            energy_list.append(float(energy.real))
    return energy_list


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

# Ising model paramters
J = 1
h_x = 1
h_z = 0.2

# Magnetic field and coupling parameters
B_i = 5
B_f = 0.7
g_max = 0.5

# System qubit number, bath qubit number 
N_S = 5
N_B = 2
run = 5

T = 6
dt = 0.06
m = T/dt
steps = 50


# Defines system

sys = np.zeros(2 ** (N_B + N_S))
sys_start_num = 1
sys_start = ket(sys_start_num, N_S)
sys[sys_start_num] = 1
sys = np.array([sys]).T


# Evolves system
print("Start")

H = Hamiltonian(N_B, N_S, J, h_x, h_z)
GS_energy = ground_state_energy(H)
print("The exact ground state energy is: " + str(GS_energy))

print("J = " + str(J))
print("h_x = " + str(h_x))
print("h_z = " + str(h_z))
sys_step_list, measurement_list, energy_sys, energy_bath, energy_int, energy_tot = evolve_ising_ground_state(sys, dt, T, N_B, N_S, J, h_x, h_z, g, g_max, B, B_i, B_f, steps)
print("_________________")

H = Hamiltonian(N_B, N_S, J, h_x, h_z)
energy_list = energy(sys_step_list, H)

'''
with open("Wave_functions/Wave_function_Ising_model,NS_" + str(N_S) + ",NB_" + str(N_B) + ",T_" + remove_dots(str(T))
          + ",sys_start_" + str(sys_start_num) + ",run_" + str(run) + '.npy', 'wb') as f:
    np.savez(f, *sys_step_list)
'''
print(sys_start)
print(measurement_list)
print(energy_list)

'''
plot(energy_list, "Demagnetization of Ising model: N_S = " + str(N_S) + ", N_B = " + str(N_B) + ", T = " + str(T),
     "Demagnetization_of_Ising_model,NS_" + str(N_S) + ",NB_" + str(N_B) + ",J_" + remove_dots(str(J)) +
     ",h_x_" + remove_dots(str(h_x)) + ",h_z_" + remove_dots(str(h_z)) + ",sys_start_" + str(sys_start_num) + ",run_" + str(run))
'''





