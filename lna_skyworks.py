# %%

import skrf as rf
import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

sqabs = lambda x: np.square(np.absolute(x))  # noqa: E731

def calc_circle(c, r):
    theta = np.linspace(0, 2*np.pi, 1000)
    return c + r*np.exp(1.0j*theta)

def plot_noise_circles(lna, rn, gamma_opt, fmin):
    plt.close()
    ax = plt.axes()
    for nf_added in [0, 0.01, 0.02, 0.05, 0.1]:
        nf = 10**(nf_added/10) * fmin

        N = (nf - fmin)*abs(1+gamma_opt)**2/(4*rn)
        c_n = gamma_opt/(1+N)
        r_n = 1/(1-N)*np.sqrt(N**2 + N*(1-abs(gamma_opt)**2))

        n = rf.Network(name=str(nf_added), s=calc_circle(c_n, r_n))
        n.plot_s_smith()
    ax.set_title(lna.name)
    plt.show()

def plot_source_circles(lna, dec):
    plt.close()
    delta = lna.s11.s*lna.s22.s - lna.s12.s*lna.s21.s
    rs = np.absolute((lna.s12.s * lna.s21.s)/(sqabs(lna.s11.s) - sqabs(delta)))
    cs = np.conj(lna.s11.s - delta*np.conj(lna.s22.s))/(sqabs(lna.s11.s) - sqabs(delta))
    
    for i, f in enumerate(lna.f):
        # decimate it a little
        if i % dec != 0:
            continue
        n = rf.Network(name=str(f/1.e+9), s=calc_circle(cs[i][0, 0], rs[i][0, 0]))
        n.plot_s_smith()
    plt.show()

def plot_load_circles(lna, dec):
    plt.close()
    delta = lna.s11.s*lna.s22.s - lna.s12.s*lna.s21.s
    rl = np.absolute((lna.s12.s * lna.s21.s)/(sqabs(lna.s22.s) - sqabs(delta)))
    cl = np.conj(lna.s22.s - delta*np.conj(lna.s11.s))/(sqabs(lna.s22.s) - sqabs(delta))
    
    for i, f in enumerate(lna.f):
        # decimate it a little
        if i % dec != 0:
            continue
        n = rf.Network(name=str(f/1.e+9), s=calc_circle(cl[i][0, 0], rl[i][0, 0]))
        n.plot_s_smith()
    plt.show()

def plot_source_circle_at(lna, freq):
    plt.close()
    idx_freq = rf.util.find_nearest_index(lna.f, freq)
    delta = lna.s11.s*lna.s22.s - lna.s12.s*lna.s21.s
    rs = np.absolute((lna.s12.s * lna.s21.s)/(sqabs(lna.s11.s) - sqabs(delta)))
    cs = np.conj(lna.s11.s - delta*np.conj(lna.s22.s))/(sqabs(lna.s11.s) - sqabs(delta))
    
    n = rf.Network(name=str(lna.f[idx_freq]/1.e+6)+"MHz, Source", s=calc_circle(cs[idx_freq][0, 0], rs[idx_freq][0, 0]))
    n.plot_s_smith()
    plt.show()

def plot_load_circle_at(lna, freq):
    plt.close()
    idx_freq = rf.util.find_nearest_index(lna.f, freq)
    delta = lna.s11.s*lna.s22.s - lna.s12.s*lna.s21.s
    rl = np.absolute((lna.s12.s * lna.s21.s)/(sqabs(lna.s22.s) - sqabs(delta)))
    cl = np.conj(lna.s22.s - delta*np.conj(lna.s11.s))/(sqabs(lna.s22.s) - sqabs(delta))
    
    n = rf.Network(name=str(lna.f[idx_freq]/1.e+6)+"MHz, Load", s=calc_circle(cl[idx_freq][0, 0], rl[idx_freq][0, 0]))
    n.plot_s_smith()
    plt.show()

def analyse_lna(lna):
    idx_924mhz = rf.util.find_nearest_index(lna.f, 924.e+6)
    rn = lna.rn[idx_924mhz]/50
    gamma_opt = lna.g_opt[idx_924mhz]
    fmin = lna.nfmin[idx_924mhz]
    z_opt = (1 + gamma_opt) / (1 - gamma_opt)
    plot_noise_circles(lna, rn, gamma_opt, fmin)
    print(f"Noise Resistance: {rn}")
    print(f"Gamma_opt: {gamma_opt} | Mag: {abs(gamma_opt)} | Arg: {np.degrees(np.angle(gamma_opt))}")
    print(f"Optimal source impedance: {50*z_opt} | Normalised: {z_opt}")

def match_gamma_opt_highpass(lna, freq, system=50):
    idx_freq = rf.util.find_nearest_index(lna.f, freq)
    
    gamma_opt = lna.g_opt[idx_freq]
    zs = 50 * (1 + gamma_opt) / (1 - gamma_opt)
    return matched_l_network_highpass(np.conj(zs), system, freq) # match to the conjugate

def match_gamma_opt_lowpass(lna, freq, system=50):
    idx_freq = rf.util.find_nearest_index(lna.f, freq)
    
    gamma_opt = lna.g_opt[idx_freq]
    zs = 50 * (1 + gamma_opt) / (1 - gamma_opt)
    return matched_l_network_lowpass(np.conj(zs), system, freq) # match to the conjugate

def matched_l_network_highpass(source, system, freq):
    ys = 1 / source
    x_temp = np.sqrt(system/np.real(ys) - system*system)
    bl = -1j * x_temp / (system*system + x_temp*x_temp) - 1j*np.imag(ys)
    ind = -1 / (2 * np.pi * freq * np.imag(bl))
    ys = ys + bl
    zs = 1 / ys
    xc = np.imag(zs)
    cap = 1 / (2 * np.pi * freq * xc)
    return ind, cap

def matched_l_network_lowpass(source, system, freq):
    ys = 1 / source
    x_temp = -np.sqrt(system/np.real(ys) - system*system)
    bc = -1j * x_temp / (system*system + x_temp*x_temp) - 1j*np.imag(ys)
    cap = 1 * np.imag(bc) / (2 * np.pi * freq)
    ys = ys + bc
    zs = 1 / ys
    xl = np.imag(zs)
    ind = -xl / (2 * np.pi * freq)
    return cap, ind

def matched_pi_network_highpass(lna, intermediate, freq):
    ind1, cap1 = match_gamma_opt_highpass(lna, freq, intermediate)
    ind2, cap2 = matched_l_network_highpass(50, intermediate, freq)
    cap = cap1 * cap2 / (cap1 + cap2)
    return ind2, cap, ind1

def matched_pi_network_lowpass(lna, intermediate, freq):
    capl, ind1 = match_gamma_opt_lowpass(lna, freq, intermediate)
    caps, ind2 = matched_l_network_lowpass(50, intermediate, freq)
    ind = ind1 + ind2
    return caps, ind, capl

def pi_network_impedance_highpass(inds, cap, indl, freq):
    xls = 2j * np.pi * freq * inds
    xll = 2j * np.pi * freq * indl
    xc = -1j / (2 * np.pi * freq * cap)
    z1 = (50 * xls) / (50 + xls)
    z2 = z1 + xc
    z3 = (z2 * xll) / (z2 + xll)
    return z3

def pi_network_impedance_lowpass(caps, ind, capl, freq):
    xcs = -1j / (2 * np.pi * freq * caps)
    xcl = -1j / (2 * np.pi * freq * capl)
    xl = 2j * np.pi * freq * ind
    z1 = (50 * xcs) / (50 + xcs)
    z2 = z1 + xl
    z3 = (z2 * xcl) / (z2 + xcl)
    return z3

def series_cap_shunt_l(cap, ind):
    l_react = 2j * np.pi * 924e6 * ind / 50
    c_react = -1j / (2 * np.pi * 924e6 * cap) / 50
    lane1 = 1 + c_react
    total = lane1 * l_react / (lane1 + l_react)
    return total

def series_l_shunt_cap(ind, cap):
    l_react = 2j * np.pi * 924e6 * ind / 50
    c_react = -1j / (2 * np.pi * 924e6 * cap) / 50
    lane1 = 1 + l_react
    total = lane1 * c_react / (lane1 + c_react)
    return total

def calc_matching_network_vals(z1, z2):
    flipped = np.real(z1) < np.real(z2)
    if flipped:
        z2, z1 = z1, z2

    # cancel out the imaginary parts of both input and output impedances
    z1_par = 0.0
    if abs(np.imag(z1)) > 1e-6:
        # parallel something to cancel out the imaginary part of
        # z1's impedance
        z1_par = 1/(-1j*np.imag(1/z1))
        z1 = 1/(1./z1 + 1/z1_par)
    z2_ser = 0.0
    if abs(np.imag(z2)) > 1e-6:
        z2_ser = -1j*np.imag(z2)
        z2 = z2 + z2_ser

    Q = np.sqrt((np.real(z1) - np.real(z2))/np.real(z2))
    x1 = -1.j * np.real(z1)/Q
    x2 = 1.j * np.real(z2)*Q

    x1_tot = 1/(1/z1_par + 1/x1)
    x2_tot = z2_ser + x2
    if flipped:
        print(flipped)
        return x2_tot, x1_tot
    else:
        return x1_tot, x2_tot

def match(lna):
    idx_924mhz = rf.util.find_nearest_index(lna.f, 924.e+6)
    
    delta = lna.s11.s*lna.s22.s - lna.s12.s*lna.s21.s
    rl = np.absolute((lna.s12.s * lna.s21.s)/(sqabs(lna.s22.s) - sqabs(delta)))
    cl = np.conj(lna.s22.s - delta*np.conj(lna.s11.s))/(sqabs(lna.s22.s) - sqabs(delta))

    gamma_s = lna.g_opt[idx_924mhz]
    gamma_l = np.conj(lna.s22.s - lna.s21.s*gamma_s*lna.s12.s/(1-lna.s11.s*gamma_s))
    gamma_l = gamma_l[idx_924mhz, 0, 0]
    is_gamma_l_stable = np.absolute(gamma_l - cl[idx_924mhz]) > rl[idx_924mhz]
    if np.absolute(0 - cl[idx_924mhz]) < rl[idx_924mhz] and lna.s22.s[idx_924mhz] < 1:
        is_gamma_l_stable = not is_gamma_l_stable
    
    print(gamma_l)
    print(np.abs(lna.s22.s)[idx_924mhz])
    print(is_gamma_l_stable)

    z_l = rf.s2z(np.array([[[gamma_l]]]))[0,0,0]
    # note that we're matching against the conjugate;
    # this is because we want to see z_l from the BJT side
    # if we plugged in z the matching network would make
    # the 50 ohms look like np.conj(z) to match against it, so
    # we use np.conj(z_l) so that it'll look like z_l from the BJT's side
    z_par, z_ser = calc_matching_network_vals(np.conj(z_l), 50)
    return z_l, z_par, z_ser

rf.stylely()

# %%

lna = rf.Network("SKY67150_wNoise.s2p")
analyse_lna(lna)
print(match(lna))
# plot_source_circles(lna, 100)
# plot_load_circles(lna, 100)
plot_source_circle_at(lna, 9e9)
print(np.abs(lna.s11.s)[rf.util.find_nearest_index(lna.f, 9e9)])
plot_load_circle_at(lna, 0)
print(np.abs(lna.s22.s)[rf.util.find_nearest_index(lna.f, 0)])
print(match(lna))
ind, cap = match_gamma_opt_highpass(lna, 924e6)
print(series_cap_shunt_l(cap, ind))

# %%

ind, cap = match_gamma_opt_highpass(lna, 924e6)
print([ind, cap])
print(series_cap_shunt_l(cap, ind))
cap, ind = match_gamma_opt_lowpass(lna, 924e6)
print([ind, cap])
print(series_l_shunt_cap(ind, cap))

# %%

ind1, cap, ind2 = matched_pi_network_highpass(lna, 9, 924e6)
print([ind1, cap, ind2])
print(pi_network_impedance_highpass(ind1, cap, ind2, 924e6))

cap1, ind, cap2 = matched_pi_network_lowpass(lna, 9, 924e6)
print([cap1, ind, cap2])
print(pi_network_impedance_lowpass(cap1, ind, cap2, 924e6))

# %%

lna = rf.Network("GRF207X_208X_Spars_Noise/GRF2070_2080_5V_70mA_25C.s2p")
analyse_lna(lna)
print(match(lna))
plot_source_circles(lna, 5)
plot_load_circles(lna, 5)
# plot_source_circle_at(lna, 9e9)
# print(np.abs(lna.s11.s)[rf.util.find_nearest_index(lna.f, 9e9)])
# plot_load_circle_at(lna, 1)
# print(np.abs(lna.s22.s)[rf.util.find_nearest_index(lna.f, 1)])
ind, cap = match_gamma_opt_highpass(lna, 924e6)
print(f"inductance: {ind}")
print(f"capacitance: {cap}")
print(series_cap_shunt_l(cap, ind))
print(series_l_shunt_cap(ind, ind))

# %%

lna = rf.Network("PMA2-33LN+_S2P/PMA2-33LN+_AP180688_CE7590__CE7600_S_paramaters_U1.s2p")
analyse_lna(lna)

# %%

lna = rf.Network("QPL9547_SN1_5V_de-embedded_2.s2p")
analyse_lna(lna)
print(match(lna))
plot_source_circles(lna, 10)
plot_load_circles(lna, 10)
ind, cap = match_gamma_opt_highpass(lna, 924e6)
print(series_cap_shunt_l(cap, ind))

# %%

series_cap_shunt_l(3.3e-12, 12e-9)

# %%