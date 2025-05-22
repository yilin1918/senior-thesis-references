# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:48:30 2023

@author: Zz
"""



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.fftpack import fft,ifft

##### this version fixed the bug of normalization. now the normalization is done with an ultra-low steady-state frequency (1Hz in model)
##### this requires the experiment to be done starting from an ultra-low frequency.
##### this requirement is actually very reasonable, because the highest sensitivity occurs at around 0.75 theta, relying on the normalization from with steady state
##### also, we use the fft function in python, and include any waveform of laser


# h = 20e6  # W/m2/K
# coef = [1, 1, h*1e-10 ]
# w = [2 * np.pi * 100000]   ## 2pif Hz
# kz = 3


### --------------- metadata --------------
# h_12 = 10e6    ## 10, 200;
# h_23 = 10e6
# coef = [h_12*1e-10, ]#h_23*1e-10 ]

###################
layer =   ['Air',         'Graphene',            'Au',              'Cr',             'SiO2',              'Si']
# l =     [20e-6,   1e-10,   6.5e-9,     1e-10,     41.67e-9,      1e-10,    8.33e-9,    1e-10,     100e-9,     1e-10,       2000e-6]  # thickness m
# alfa_r =  [0.026,   1e-20,   kr_Gr,      coef[0],    314,        18.69e6*e-10,  93.7,    50e6*1e-10,   1.36,     436e6*1e-10,    140]
# alfa_z =  [0.026,   1e-20,   kz_Gr,      coef[0],    314,        18.69e6*e-10,  93.7,    500e6*1e-10,   1.36,     436e6*1e-10,    140]
# rho =    [1.205,   1,     1950,       1,      19300,       1,       7190,      1,      2200,      1,        2329]  # density  kg/m3
# c =      [1007,   1,     710,       1,      128,        1,       448,      1,       743,      1,        706] # heat capacity J/Kg/K
###################

# w = [2 * np.pi * 100000]   ## 2pi*f Hz

spot_radius = 1e-6  ## 1/e radius, meter

# ### the following list follow the index of "layer"
p_laser = [0, 0.35, 0.281, 0, 0.069]
tl_list = [0, 18e-9, 18.9e-9, 0, 850e-9]
heat_source_layer_list = [1, 2, 4]
###############
p_laser = [0, 0.26, 0.181, 0.117, 0, 0]
tl_list = [0, 18e-9, 18.9e-9, 12.7e-9, 0, 0]
heat_source_layer_list = [1, 2, 3]
###############

SIMPSON_STEPS = 10   ## 100 does not produce much difference
NUM_SINS = 10    ### !!!!!!!!  ## 100 does not produce much difference

# # tl_layer2 = 14e-9
# # tl_layer3 = 850e-9

radial_upper_limit = 10

corrected_tl_sub_pow = 2


# f_hz = np.logspace(1, 7, num = 30, base = 10)

try:
    kr_Gr  # 如果 `kr_Gr` 已在 `main.py` 里定义，就不修改
except NameError:
    kr_Gr = 2300 # 只有在未定义时才设默认值

try:
    kz_Gr  # 如果 `kr_Gr` 已在 `main.py` 里定义，就不修改
except NameError:
    kz_Gr = 0.3  # 只有在未定义时才设默认值

### calculate the T_top_substrate from substrate heat source
def two_materials_anisotropic_twoDir_heat_source_T_top(w, coef, heat_source_layer, output_temperature_layer):   ## the layer number 3 is for substrate layer

    ### --------------- metadata --------------
    # h_12 = 10e6    ## 10, 200;
    # h_23 = 10e6
    # coef = [h_12*1e-10, ]#h_23*1e-10 ]

    # layer =   ['Air',            'Graphene',               'SiO2',                     'Si'      ]
    l_int = 1e-10
    #####       0                   1                         2                         3                             4
    layer =   ['Air',         'Graphene',            'Au',              'Cr',             'SiO2',              'Si']
    l =     [20e-6,   1e-10,   6.5e-9,     1e-10,     41.67e-9,      1e-10,    8.33e-9,    1e-10,     100e-9,     1e-10,      2000e-6]  # thickness m
    alfa_r =  [0.026,   1e-20,   kr_Gr,      coef[0],    200,      1000e6*1e-10,    93.7,    50e6*1e-10,   1.36,     436e6*1e-10,    140]
    alfa_z =  [0.026,   1e-20,   kz_Gr,      coef[0],    200,      1000e6*1e-10,    93.7,    50e6*1e-10,   1.36,     436e6*1e-10,    140]
    rho =    [1.205,   1,     1950,       1,      19300,       1,       7190,      1,      2200,      1,        2329]  # density  kg/m3
    c =      [1007,   1,     710,       1,      128,        1,       448,      1,       743,      1,        706] # heat capacity J/Kg/K


    # w = [2 * np.pi * 100000]   ## 2pi*f Hz

    # spot_radius = 2.85e-6  ## 1/e radius, meter

    ### the following list follow the index of "layer"
    # p_laser = [0, 0.014, 0, 0.077]
    # tl_list = [0, 14e-9, 0, 850e-9]
    # heat_source_layer_list = [1,3]


    # SIMPSON_STEPS = 200
    # NUM_SINS = 200    ### !!!!!!!!

    # tl_layer2 = 14e-9
    # tl_layer3 = 850e-9

    # radial_upper_limit = 10

    # corrected_tl_sub_pow = 2#.4

    # f_hz = np.logspace(1, 7, num = 30, base = 10)
    ### -------------- end of  metadata --------------


    ### feature: heat source is embedded, get the temperature of heat source
    #np.set_printoptions(precision=15)
    w0 = spot_radius     ## laser spot radius /m
    tl = tl_list[heat_source_layer]     ## penetration depth

    w1 = w0  # the 1/e2 radius of the pump beam
    spot = w1

    layerAboveAu = heat_source_layer * 2 # consider the interface, 6  ## air, air/2D interface, 2D, 2D/Si interface
    ### here the layer above gold acutually serves as the layer above heat source for this function
    # layerUnderAu = 0   ## but acutally layer and layeraboveAu together is enough
    ### layer above gold and layer under gold together determines the position of heat source
    target_layer_index = output_temperature_layer * 2   ## the layer whose temperature we want to predict in this function, only needed if the target layer is different from the heat source layer
    ## MoS2 hBN Si
    # alfa_z = [kz[0],    1e-20,   kz[1],     coef[0], kz[2],     coef[1],    kz[3]]  # coef(1)]
    # alfa_r = [kr[0],    1e-20,   kr[1],     coef[0], kr[2],     coef[1],    kr[3]]  # coef(2)]
    # rho =    [rhoo[0],  1,       rhoo[1],   1,       rhoo[2],   1,          rhoo[3]]  # 2530]  # density  kg/m3
    # c =      [cp[0],    1,       cp[1],     1,       cp[2],     1,          cp[3]]  # 130]  # heat capacity J/Kg/K
    # L =      [l[0],     1e-10,   l[1],      1e-10,   l[2],      1e-10,      l[3]]  # 90e-6]  # thickness m

    A_0 = p_laser[heat_source_layer]  ## seems to control the magnitude of signal, laser power
    #time_steps = 1000

    simpson_steps = SIMPSON_STEPS
    up = radial_upper_limit / np.sqrt(w0 * w0 + w1 * w1)     ## upper limit for radial spreading
    pace = up / simpson_steps

    m = np.arange(0, up + pace, pace)

    even_indices = np.arange(2, simpson_steps, 2)
    odd_indices = np.arange(3, simpson_steps-1, 2)

    k = m
    K, W = np.meshgrid(k, w)
    layers = len(alfa_z)
    M11 = np.zeros((1, len(k), layers), dtype=complex)    ###  assign dtype!! otherwise only real part is stored
    M12 = np.zeros((1, len(k), layers), dtype=complex)
    M21 = np.zeros((1, len(k), layers), dtype=complex)
    M22 = np.zeros((1, len(k), layers), dtype=complex)



    ### start from the 6th layer, i.e. the substrate layer
    Q1 = np.sqrt((alfa_r[layerAboveAu] * K * K + rho[layerAboveAu] * c[layerAboveAu] * 1j * W) / alfa_z[layerAboveAu])
    #print(Q1)
    for n in range(layerAboveAu, layerAboveAu + 1):    ## the top layer needs to adopt the volumetric heating model
        Q = np.sqrt((alfa_r[n] * K * K + rho[n] * c[n] * 1j * W) / alfa_z[n])

        threshold = 15      ## cutoff, need to be tested, to decide above which value the result can converge, to avoid overflow of cosh and sinh
        temp = Q * l[n]
        # print(np.shape(temp))
        # print(temp)
        for i in range(np.shape(temp)[1]):
            if np.abs(temp[0,i].real) > threshold or np.abs(temp[0,i].imag) > threshold:
                temp[0,i] = np.sign(temp[0,i].real) * threshold + np.sign(temp[0,i].imag) * threshold * 1j
        # temp = [np.sign(i.real) * threshold + np.sign(i.imag) * threshold * 1j for i in temp if np.abs(i.real) > threshold or np.abs(i.imag) > threshold]


        M11[:, :, n] = np.cosh(temp)
        M12[:, :, n] = -np.sinh(temp) / (alfa_z[n] * Q) + tl * np.cosh(temp) / alfa_z[n] - tl / alfa_z[n] * np.exp(-l[n]/tl)
        M21[:, :, n] = -alfa_z[n] * Q * np.sinh(temp)
        M22[:, :, n] = np.cosh(temp) - Q * tl * np.sinh(temp) - np.exp(-l[n]/tl)




    ## treat it as if there are still layers under the substrate
    for n in range(layerAboveAu + 1, layers):   ## the 2nd and 3rd layers are treated as usual
        Q = np.sqrt((alfa_r[n] * K * K + rho[n] * c[n] * 1j * W) / alfa_z[n])

        threshold = 15      ## need to be tested, to decide above which value to assign the temp as a fixed number, to avoid overflow of cosh and sinh
        temp = Q * l[n]
        # print(np.shape(temp))
        # print(temp)
        for i in range(np.shape(temp)[1]):
            if np.abs(temp[0,i].real) > threshold or np.abs(temp[0,i].imag) > threshold:
                temp[0,i] = np.sign(temp[0,i].real) * threshold + np.sign(temp[0,i].imag) * threshold * 1j
        # temp = [np.sign(i.real) * threshold + np.sign(i.imag) * threshold * 1j for i in temp if np.abs(i.real) > threshold or np.abs(i.imag) > threshold]


        M11[:, :, n] = np.cosh(temp)
        M12[:, :, n] = -np.sinh(temp) / (alfa_z[n] * Q)
        M21[:, :, n] = -alfa_z[n] * Q * np.sinh(temp)
        M22[:, :, n] = np.cosh(temp)

    ## calculate the matrix of the layers above gold
    for n in range(layerAboveAu):   ## the 2nd and 3rd layers are treated as usual
        Q = np.sqrt((alfa_r[n] * K * K + rho[n] * c[n] * 1j * W) / alfa_z[n])

        threshold = 15      ## need to be tested, to decide above which value to assign the temp as a fixed number, to avoid overflow of cosh and sinh
        temp = Q * l[n]
        # print(np.shape(temp))
        # print(temp)
        for i in range(np.shape(temp)[1]):
            if np.abs(temp[0,i].real) > threshold or np.abs(temp[0,i].imag) > threshold:
                temp[0,i] = np.sign(temp[0,i].real) * threshold + np.sign(temp[0,i].imag) * threshold * 1j
        # temp = [np.sign(i.real) * threshold + np.sign(i.imag) * threshold * 1j for i in temp if np.abs(i.real) > threshold or np.abs(i.imag) > threshold]


        M11[:, :, n] = np.cosh(temp)
        M12[:, :, n] = -np.sinh(temp) / (alfa_z[n] * Q)
        M21[:, :, n] = -alfa_z[n] * Q * np.sinh(temp)
        M22[:, :, n] = np.cosh(temp)

        # Q = np.sqrt((alfa_r[n] * K * K + rho[n] * c[n] * 1j * W) / alfa_z[n])
        # M11[:, :, n] = 1
        # M12[:, :, n] = -np.tanh(Q * L[n]) / (alfa_z[n] * Q)
        # M21[:, :, n] = -alfa_z[n] * Q * np.tanh(Q * L[n])
        # M22[:, :, n] = 1

    ### consider the top layer above the substrate
    # if layerAboveAu > 0:
    A1 = M11[:, :, 0]
    B1 = M12[:, :, 0]
    C1 = M21[:, :, 0]
    D1 = M22[:, :, 0]

    for j in range(layerAboveAu - 1):
        A1, B1, C1, D1 = ABCD_func(A1, B1, C1, D1, M11[:, :, j + 1], M12[:, :, j + 1],
                                  M21[:, :, j + 1], M22[:, :, j + 1])

    # else:
    #     A1 = 0
    #     B1 = 0
    #     C1 = 0
    #     D1 = 1


    A2 = M11[:, :, layers - 1]
    B2 = M12[:, :, layers - 1]
    C2 = M21[:, :, layers - 1]
    D2 = M22[:, :, layers - 1]

    for j in range(layers - layerAboveAu - 1):     ## multiply all the M matrix together to get the final A B C D
        A2, B2, C2, D2 = ABCD_func(A2, B2, C2, D2, M11[:, :, layers - j - 2], M12[:, :, layers - j - 2],
                                  M21[:, :, layers - j - 2], M22[:, :, layers - j - 2])

    # G = (A_0 / (2 * np.pi)) * K * (-D1 * D2 / (D1 * C2 + D2 * C1)) * np.exp(
    #     -K * K * (w0 ** 2 + w1 ** 2) / 8)
    # G = (A_0 / (2 * np.pi)) * K * (- D2 / C2 ) * np.exp(
    #     -K * K * (w0 ** 2 + w1 ** 2) / 8)
    # G = (A_0 / (2 * np.pi)) / (1-np.exp(-L[0]/tl)) * K * (- D2 / C2 ) * (1/(1- np.abs(Q1)**2 * tl**2)) * np.exp(
    #     -K * K * (w0 ** 2 + w1 ** 2) / 8)
    # G = (A_0 / (2 * np.pi))  * K * (- D2 / C2 ) * (1/(1- (np.abs(Q1)**2) * (tl**2))) * np.exp(
    #     -K * K * (w0 ** 2 + w0 ** 2 ) / 4)
    if layerAboveAu == target_layer_index:
        G = A_0  * K * (- D1 * D2 ) / (C1 * D2 * np.exp(-K * K * (w0 ** 2) / 4) + C2 * D1 * (2 * np.pi) * (1- (np.abs(Q1)**2) * (tl**corrected_tl_sub_pow))) * np.exp(
            -K * K * (w0 ** 2 + w0 ** 2 ) / 4)

    if layerAboveAu > target_layer_index:  ## Target layer is on top of the heat source layer
        ## calculate the matrix for T_target, multiply the matrix starting from the target layer to the heat source layer
        A3 = M11[:, :, target_layer_index]
        B3 = M12[:, :, target_layer_index]
        C3 = M21[:, :, target_layer_index]
        D3 = M22[:, :, target_layer_index]

        for j in range(layerAboveAu - target_layer_index - 1):
            A3, B3, C3, D3 = ABCD_func(A3, B3, C3, D3, M11[:, :, j + target_layer_index + 1], M12[:, :, j + target_layer_index + 1],
                                      M21[:, :, j + target_layer_index + 1], M22[:, :, j + target_layer_index + 1])

        G = A_0  * K * (A3 - B3 * C1 / D1 ) * (- D1 * D2 ) / (C1 * D2 * np.exp(-K * K * (w0 ** 2) / 4) + C2 * D1 * (2 * np.pi) * (1- (np.abs(Q1)**2) * (tl**corrected_tl_sub_pow))) * np.exp(
            -K * K * (w0 ** 2 + w0 ** 2 ) / 4)

    if layerAboveAu < target_layer_index:  ## Target layer is under the heat source layer

        ## calculate the matrix for T_target, multiply the matrix starting from the target layer to the heat source layer
        A3 = M11[:, :, target_layer_index-1]   ## target layer under the heat source, multiply from target_layer_index-1
        B3 = M12[:, :, target_layer_index-1]
        C3 = M21[:, :, target_layer_index-1]
        D3 = M22[:, :, target_layer_index-1]

        for j in range(target_layer_index - layerAboveAu - 1):
            A3, B3, C3, D3 = ABCD_func(A3, B3, C3, D3, M11[:, :, target_layer_index - j - 2], M12[:, :, target_layer_index - j - 2],
                                      M21[:, :, target_layer_index - j - 2], M22[:, :, target_layer_index - j - 2])

        G = A_0  * K * (A3 - B3 * C2 / D2 ) * (- D1 * D2 ) / (C1 * D2 * np.exp(-K * K * (w0 ** 2) / 4) + C2 * D1 * (2 * np.pi) * (1- (np.abs(Q1)**2) * (tl**corrected_tl_sub_pow))) * np.exp(
            -K * K * (w0 ** 2 + w0 ** 2 ) / 4)



    ## simpson integration
    delT_2Dsimp = (pace / 3) * (G[:, 0] + 2 * np.sum(G[:, odd_indices-1], axis=1) +
                                4 * np.sum(G[:, even_indices-1], axis=1) + G[:, simpson_steps - 1])

    phase = np.angle(delT_2Dsimp) #* 180 / np.pi
    amp = np.abs(delT_2Dsimp)

    phase_amp = np.column_stack((phase, amp))
    return phase_amp, spot











def ABCD_func(M11n, M12n, M21n, M22n, M11n_1, M12n_1, M21n_1, M22n_1):
    A = M11n * M11n_1 + M12n * M21n_1
    B = M11n * M12n_1 + M12n * M22n_1
    C = M21n * M11n_1 + M22n * M21n_1
    D = M21n * M12n_1 + M22n * M22n_1
    return A, B, C, D








def ave_temp_rise_all_layer(f_hz, coef):      ## f_hz is a list of frequency
    if isinstance(f_hz, float):     ## convert f_hz to a one element array if it is a single number, to comform to the function
        f_hz = np.array([f_hz])
    # h = 10e6  # W/m2/K
    # coef = [0.1, h*1e-10 ]    ## initial guess for kz and h
    # f_hz = np.linspace(1e3, 1e8, 100)
    # w = [2 * np.pi * f_hz]   ## 2pif Hz
    # w = 2 * np.pi * f_hz
    #print(f_hz)
    # print('kz = %.15f W/m/K, h = %.15f MW/m2/K' %(coef[0], coef[1]*1e10/1e6))
    # ave_active_2D = np.zeros_like(f_hz)
    ave_active_T_all_layer_all_freq_list = np.zeros([int(np.shape(f_hz)[0]), int(len(layer)-1)])
    T_all_layers_min_all_freq_list = np.zeros([int(np.shape(f_hz)[0]), int(len(layer)-1)])

    for e in range(int(np.shape(f_hz)[0])):
        w = [2 * np.pi * f_hz[e]]   ## 2pif Hz

        ########################################################################################################################################
        # for each layer (except for air), calculate all the solutions from all the heat sources -----
        num_sins = NUM_SINS
        N = num_sins
        L = 1 / (w[0] / (2 * np.pi))      ## period
        dx = L / (N - 1)
        x = np.arange(0, 2 * L + dx, dx)    ## how many periods to plot, the size of x is also the size of the outputs

        T_all_layers_list = np.zeros([int(np.shape(x)[0]), int(len(layer)-1)])   ## define an array tp store the temperature solutions of all the layers  ## define an array tp store the temperature solutions of all the layers
        ave_active_T_all_layers_list = np.zeros(int(len(layer)-1))
        T_all_layers_min_list = np.zeros(int(len(layer)-1))

        for z in range(len(layer)-1): ## calculate each layer's temperature
            ### assign an array for each target layer, to store the solutions for all the heat sources
            T_layer_z = np.zeros(int(np.shape(x)[0]))
            T_layer_z_from_each_hs_list = np.zeros([int(np.shape(x)[0]), int(len(heat_source_layer_list))])

            for y in range(len(heat_source_layer_list)):

                phaseamp = np.zeros((2, int(N/2+2)))
                f= np.zeros_like(x)
                f[:(N // 2 + 1)] = 1
                f[N:(3 * N // 2 + 1)] = 1
                ffs1 = 0
                ffs2 = 0

                for i in range(len(w)):
                    for j in range(1, int(N/2+1)):
                        phaseamp[:, j], spot= two_materials_anisotropic_twoDir_heat_source_T_top(w[i] * j, coef, heat_source_layer_list[y], z+1)
                        Ak = np.sum(f * np.cos(2 * np.pi * j * x / L)) * dx * 2 / L
                        Bk = np.sum(f * np.sin(2 * np.pi * j * x / L)) * dx * 2 / L
                        ffs1 += ( Ak * np.cos(2 * np.pi * j * x / L) + Bk * np.sin(2 * np.pi * j * x / L) )
                        ffs2 += ( Ak * phaseamp[1, j] * np.cos(2 * np.pi * j * x / L + phaseamp[0, j]) + \
                                    phaseamp[1, j] * Bk * np.sin(2 * np.pi * j * x / L + phaseamp[0, j]) )


                ffs2_min = ffs2[int(np.shape(ffs2)[0]/8*3.95)]
                ffs2_max = ffs2[int(np.shape(ffs2)[0]/8)]
                # print(ffs2[int(np.shape(ffs2)[0]/8)])
                # print(ffs2[int(np.shape(ffs2)[0]/8*3)])

                T_layer_z_from_each_hs_list[:,y] = ffs2 #- ffs2_min

            ### sum up the solutions from each heat source
            for i in range(int(np.shape(T_layer_z_from_each_hs_list)[0])):
                T_layer_z[i] = np.sum(T_layer_z_from_each_hs_list[i,:])

            ## assign this layer temperature to the all layer list
            T_all_layers_list[:,z] = T_layer_z

            ave_active_T_all_layers_list[z] = np.average(T_layer_z[0:int(np.shape(T_layer_z)[0]/4)])
            T_all_layers_min_list[z] = T_layer_z[int(np.shape(T_layer_z)[0]/8*3.95)]      ## the minimum temperature in the off-state, if 1Hz, then it is the temperature we need to offset


        for i in range(len(layer)-1):  ## the eth freq, ith layer
            ave_active_T_all_layer_all_freq_list[e,i] = ave_active_T_all_layers_list[i]
            T_all_layers_min_all_freq_list[e,i] = T_all_layers_min_list[i]



    return ave_active_T_all_layer_all_freq_list, T_all_layers_min_all_freq_list




# def normalized_RSC_all_layer(f_hz):    ## f_hz and normalized_RSC are single values, for fitting
#     # coef = [ coef_0, coef_1]
#     #ave_temp_rise_steady_state = ave_temp_rise(1, coef)    ## 1 Hz as steady state
#     #print(f_hz)
#     ave_temp_rise_array = ave_temp_rise_all_layer(f_hz)
#     ave_temp_rise_1Hz = ave_temp_rise_all_layer([1])
#     # ave_temp_rise_1Hz = np.asarray(ave_temp_rise_1Hz)
#     ## need to move upward an amplitude since the profile of superpositioned thermal response is not zero-based
#     print(ave_temp_rise_array)
#     print(ave_temp_rise_1Hz)
#     normalized_rsc = np.zeros([int(np.shape(ave_temp_rise_array)[0]), int(len(layer)-1)])
#     for i in range(len(layer)-1):
#         # normalized_rsc[:,i] = (ave_temp_rise_array[:,i]+ave_temp_rise_array[0,i])/(ave_temp_rise_array[0,i]+ave_temp_rise_array[0,i])
#         for j in range(np.shape(ave_temp_rise_array)[0]):
#             normalized_rsc[j,i] = (ave_temp_rise_array[j,i]+ave_temp_rise_1Hz[0,i])/(ave_temp_rise_1Hz[0,i]+ave_temp_rise_1Hz[0,i])

#     return normalized_rsc



def normalized_RSC_all_layer(f_hz, coef_0):    ## f_hz and normalized_RSC are single values, for fitting
    coef = [ coef_0, ]#coef_1]
    #ave_temp_rise_steady_state = ave_temp_rise(1, coef)    ## 1 Hz as steady state
    #print(f_hz)
    ave_temp_rise_array, T_all_layers_min_all_freq  = ave_temp_rise_all_layer(f_hz, coef)
    ave_temp_rise_1Hz, T_all_layers_min_1Hz = ave_temp_rise_all_layer([1], coef)   ## the frequency to normalize
    # ave_temp_rise_1Hz = np.asarray(ave_temp_rise_1Hz)
    ## need to move upward an amplitude since the profile of superpositioned thermal response is not zero-based
    # print(ave_temp_rise_array)
    # print(ave_temp_rise_1Hz)
    normalized_rsc = np.zeros([int(np.shape(ave_temp_rise_array)[0]), int(len(layer)-1)])
    for i in range(len(layer)-1):
        # normalized_rsc[:,i] = (ave_temp_rise_array[:,i]+ave_temp_rise_array[0,i])/(ave_temp_rise_array[0,i]+ave_temp_rise_array[0,i])
        for j in range(np.shape(ave_temp_rise_array)[0]):
            # normalized_rsc[j,i] = (ave_temp_rise_array[j,i]-T_all_layers_min_1Hz[0,i])/(ave_temp_rise_array[0,i]-T_all_layers_min_1Hz[0,i])
            ### add a shift term to consider the off-state residue laser power
            off_state_laser_power_ratio = 0#.1
            normalized_rsc[j,i] = ((ave_temp_rise_array[j,i]-T_all_layers_min_1Hz[0,i])+((ave_temp_rise_array[0,i]-T_all_layers_min_1Hz[0,i])*off_state_laser_power_ratio))/((ave_temp_rise_array[0,i]-T_all_layers_min_1Hz[0,i])*(1+off_state_laser_power_ratio))

    #### here we specify a layer for output, used in the fitting function
    normalized_rsc_output = normalized_rsc[:,0]   ## the index 0 means graphene

    print('h = %.15f MW/m2/K' %(coef[0]/1e-10/1e6))
    return normalized_rsc_output








# # ####----------calculate the normalized raman shift from experimental data---------


# #freq2 = [1,10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000]  ## kHz   ##5000000000  ## Hz
# #freq2 = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900,]
# #freq2 = [1,10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]#, 2000, 3000]
# #freq2 = [1,10, 20, 30, 40, 50, 60, 70, 80, 90,  200, 300, 400, 500, 600, 700, 800, 900, 1000]#, 2000, 3000]
# #freq2 = [1, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, ]# 2000, 3000, 4000, 5000, 6000]



# freq2 = [100, 120, 160, 200, 240, 300,  400, 500, 600,  700,  1000, 1200, 1600,  2000, 2400, 3000, 4000, 5000, 6000,  7000,  10000, ] ## kHz   ##5000000000  ## Hz
# # freq2 = [1, 4,  10,  50, 100,  400,  1000, 4000, 10000] ## kHz   ##5000000000  ## Hz

# # freq2 = [1, 2, 4, 7, 10, 20,  50,  70, 100, 200, 400, 700, 1000, 2000, 4000, 7000, 10000,] ## kHz   ##5000000000  ## Hz
# #freq2 = [1, 100, 200, 300, 400, 500, 600, 700, 800, 1000, 2000, 3000, ] ## kHz   ##5000000000  ## Hz
# # freq2 = [1, 2,4,7, 10, 20, 30, 40, 50, 60, 70, 100, 200, 300, 400, 500, 600, 700,  1000, 2000, 3000, 4000, 5000, 6000, 7000, 10000,] ## kHz   ##5000000000  ## Hz
# # freq2 = [1, 2, 3, 4, 5, 6, 7, 10, 12, 16, 20, 30, 40, 50, 60, 70, 100, 120, 160, 200, 300, 400, 500, 600, 700,  1000, 1200, 1600, 2000, 3000, 4000, 5000, 6000, 7000, 10000,]
# # freq2 = [1, 10, 50,  90, 200,  400,  700, 1000,  2000, 3000, ]
# freq2_hz = [i*1000 for i in freq2]
# num_freq2 = np.shape(freq2)[0]
# ## parameters------
# w0 = 1580.70 #382.30 #407.71 #382.35 #407.62 #382.27 #407.66 #382.31 #407.66 #382.21 #407.69 #382.23 #382.23 #     ##cm-1  from previous
# #w0 = 130.82
# bias = -181
# voltage_min = -400  ## mV
# voltage_max = 400  ## mV
# laser_power = 100  # %
# objective = 20
# exposure_time = 3 # s
# date = 2024062813
# times = 5
# accums = 1

# map_index = 2

# num_first_freq = 1
# #num_data = np.shape(bias)[0] * np.shape(p_laser)[0]
# ##----data-----loading-----
# peak1 = np.zeros([times+2,num_freq2])  ## additional two for ave and std
# #peak2 = np.zeros([times+2,num_freq2]) ## additional two for ave and std

# for i in range(num_freq2):
#     if i == 0:
#         first_freq_index = 1
#         skiprow = 1 + (first_freq_index-1) * times
#         data_temp = np.loadtxt('Graphene_SiO2_untreated_%dbias_%d_%dkHz_%dmv_%dmv_%d%%_%dx_%ds_%dtimes_%daccums_%d_Map%d.txt' %(bias,freq2[0],freq2[-1],voltage_min,voltage_max,laser_power,objective,exposure_time,times,accums,date,map_index), skiprows = skiprow, max_rows = times, usecols = 1)  ## note for the double % to 转义

#     else:
#         skiprow = 1 + times * (num_first_freq + i-1 )
#         data_temp = np.loadtxt('Graphene_SiO2_untreated_%dbias_%d_%dkHz_%dmv_%dmv_%d%%_%dx_%ds_%dtimes_%daccums_%d_Map%d.txt' %(bias,freq2[0],freq2[-1],voltage_min,voltage_max,laser_power,objective,exposure_time,times,accums,date,map_index), skiprows = skiprow, max_rows = times, usecols = 1)  ## note for the double % to 转义
#     peak1[0:times, i] = data_temp

#     # data_temp = np.loadtxt('Graphite_%dbias_%dmv_%dkHz_%d%%_%dx_%ds_%dtimes_%d_Map1.txt' %(bias,amplitude,freq2[i],p_laser,objective,exposure_time,times,date), skiprows = 1, usecols = 1)  ## note for the double % to 转义
#     # peak1[0:times, i] = data_temp

# ###  ave and std-------------
# start = 0
# end = times

# # peak1[start:end,:] = np.sort(peak1[start:end,:], axis=0)  ## sorting small to big

# # start = 1
# # end = times-1

# for i in range(num_freq2):
#     peak1[times,i] = np.average(peak1[start:end,i]) #+ 0.06/num_freq2 * i
#     peak1[times+1,i] = np.std(peak1[start:end,i])

#     # if i == 0:
#     #     peak1[times,i] = np.average(peak1[start:end,i]) - 0.01
#     #     peak1[times+1,i] = np.std(peak1[start:end,i])

#     # if i == 22 or i == 23 or i == 24:
#     #     peak1[times,i] = np.average(peak1[start:end,i]) + 0.02
#     #     peak1[times+1,i] = np.std(peak1[start:end,i])


# ##----calculate normalized raman shift for each freq--------

# normalized_raman_shift_ave = np.zeros(num_freq2)
# normalized_raman_shift_std = np.zeros(num_freq2)

# for i in range(num_freq2):
#     normalized_raman_shift_ave[i] = (peak1[times,i]-w0)/(peak1[times,0]-w0)
#     normalized_raman_shift_std[i] = peak1[times+1,i]/(peak1[times,0]-w0)/1

# ## if no raman file is provided, the expt data is imported directly -----
# # freq2_hz = [1000,  40000, 200000, 400000, 1000000, 2000000]
# # normalized_raman_shift_ave = [1, 0.85, 0.83, 0.78, 0.75, 0.71, ]
# # normalized_raman_shift_std = [0, 0.02, 0.007, 0.012, 0.01, 0.02,]

# # freq2_hz = [1000, 40000, 200000, 400000, 2000000]
# # normalized_raman_shift_ave = [1, 0.85, 0.79, 0.71, 0.67, ]
# # normalized_raman_shift_std = [0, 0.02, 0.04, 0.03, 0.02,]

# # freq2_hz = [1000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000]
# # normalized_raman_shift_ave = [1, 0.935, 0.85, 0.79, 0.745, 0.71, 0.70, 0.66, 0.65]
# # normalized_raman_shift_std = [0, 0.02, 0.01, 0.016, 0.018, 0.009, 0.02, 0.01, 0.011]


## ---------------- experiment data ---------------
freq2 = [500,
2000,
2500,
3000,
4000,
4200,
7000,
8000,
9000,
] ## kHz   ##5000000000  ## Hz

freq2_hz = [i*1000 for i in freq2]


normalized_raman_shift_ave = [1,
0.967532468,
0.961038641,
0.948051948,
0.922077922,
0.915584416,
0.896103896,
0.883116883,
0.837662338,
]


normalized_raman_shift_std = [0.02942,
0.02942,
0.02942,
0.0,
0.02942,
0,
0,
0.02942,
0.03798,
]



####  ------- fitting --------------------------------
### first pick the data we want to fit
list_use_data =  [0,  1,  2, 3, 4,  5,  6, 7, 8,]# 9]   ### delete some data points
# list_use_data =  [0,  1,  2, 3, 4,  5,  6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,     18, 19, 20]   ### delete some data points

#list_use_data =  [0, 1, 2,   4, 5, 6, 7, 8, 9, 10, ]#11  ]#
#list_use_data =  [0, 1, 2, 3,  4, 5, 6, 7,  8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,  ]
a = np.take(freq2_hz, list_use_data)
b = np.take(normalized_raman_shift_ave, list_use_data)
c = np.take(normalized_raman_shift_std, list_use_data)


# ### if the list_data_use does not start from 0, need to renormalize the raman data
b = b/b[0]

###-------define the frequency fed to the model, same as the picked raman data range-----

f_hz = np.logspace(np.log10(a[0]), np.log10(a[-1]), num = 30, base = 10)

h = 10e6  # W/m2/K
coef = [ h*1e-10 ]    ## initial guess for h

freq2_hz = np.array(freq2_hz)


popt_T, pcov_T = curve_fit(normalized_RSC_all_layer, a, b, p0 = coef[0]) #, bounds=(1e-4, [10,100e6*1e-10]))


#y = [normalized_RSC(i, popt_T[0], popt_T[1]) for i in f_hz]

#y = normalized_RSC(f_hz, coef[0], coef[1])
y = normalized_RSC_all_layer(f_hz, popt_T[0])#, popt_T[1])


mse = np.mean((normalized_raman_shift_ave - normalized_RSC_all_layer(freq2_hz, popt_T[0]))**2)
final_h = popt_T[0] / 1e-10 / 1e6
#####print(f'Final h = {final_h:.6f} MW/m²/K')
#####print('mse = %.15f' %mse)





###---plot---the therotical rsc curve of assigned coef-------
fontsize = 30
lw = 3 # line width
labelsize = 35
legendsize = 35
elinewidth = 2
ax_linewidth = 2
ticksize = 5
tickwidth = 3
linewidth = 2
markersize = 10
markeredgewidth = 3
capsize = 8

# ymin= 0  # y range in plot
# ymax= 60

fig,ax1=plt.subplots(1,1, figsize=(12,10))

#plt.plot(x, ffs1 * 186.955 * 2, 'k', linewidth=0.5)
#plt.plot(x, ffs1 * 1.94, 'k', linewidth=0.5)
ax1.plot(f_hz, y, 'k', linewidth=lw, label = 'Best fit')
ax1.errorbar(a, b, yerr = np.abs(c), fmt = 'o', color = 'k', markersize = markersize, markeredgewidth = markeredgewidth, capsize = capsize, label = 'Experiment')



freq_RSC_err = np.column_stack((a,b,c))
np.savetxt('freq_RSC_err.txt', freq_RSC_err)


plus_minus = 0.2
y_plus = normalized_RSC_all_layer(f_hz, popt_T[0]*(1+plus_minus))
y_minus = normalized_RSC_all_layer(f_hz, popt_T[0]*(1-plus_minus))
ax1.plot(f_hz, y_plus, 'g--', linewidth=lw, label = '+%d%%' %(plus_minus*100))
ax1.plot(f_hz, y_minus, 'r--', linewidth=lw, label = '-%d%%' %(plus_minus*100))



#ax1.set_xlim(-0.01,1)
ax1.set_ylim(0.5,1.1)
ax1.set_ylabel(' Normalized Temperature rise ',fontsize=labelsize, labelpad = 15)
ax1.set_xlabel(' Frequency (Hz)',fontsize=labelsize, labelpad = 15)
ax1.set_xscale('log')
ax1.tick_params(labelsize = labelsize, size = ticksize, width = tickwidth, pad = 10)
ax1.spines['bottom'].set_linewidth(ax_linewidth)
ax1.spines['top'].set_linewidth(ax_linewidth)
ax1.spines['left'].set_linewidth(ax_linewidth)
ax1.spines['right'].set_linewidth(ax_linewidth)

ax1.legend(frameon = False, fontsize = legendsize)
#ax1.set_title("kz = %.3f W/m/K, h = %.3f MW/m$^2$/K, MSE = %.8f"  %(popt_T[0], popt_T[1]*1e10/1e6, mse), fontsize = fontsize, pad = 20)
# ax1.set_title("kz = %.3f W/m/K, h = %.3f MW/m$^2$/K, MSE = %.8f"  %(popt_T[0], h*1e10/1e6, mse), fontsize = fontsize, pad = 20)

# plt.savefig('fitting_kz_with_fixed_h_%dWm2K' %(h*1e10))
ax1.set_title("h = %.3f MW/m2/K, MSE = %.8f"  %(popt_T[0]/1e-10/1e6, mse), fontsize = fontsize, pad = 30)
plt.tight_layout()

plt.savefig('plots20/fitting_%d' %(popt_T[0]/1e-10/1e6*1000*np.random.rand()), bbox_inches = 'tight')
# plt.show()
