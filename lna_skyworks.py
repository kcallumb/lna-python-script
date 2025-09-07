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

def plot_t_network_input_margins(lna, caps, ind, capl, freq):
    idx_freq = rf.util.find_nearest_index(lna.f, freq)
    
    rn = lna.rn[idx_freq]/50
    gamma_opt = lna.g_opt[idx_freq]
    fmin = lna.nfmin[idx_freq]
    
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
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # caps_temp = cap + 
                pass
    zs = t_network_impedance_highpass(caps, ind, capl, freq)
    gamma_s = (zs - 50) / (zs + 50)

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
    print(f"---{lna.name}---")
    print(f"Gain: {lna.s21.s_db[idx_924mhz][0, 0]:.3f}dB")
    print(f"Isolation: {lna.s12.s_db[idx_924mhz][0, 0]:.3f}dB")
    print(f"Noise Resistance: {rn}ohms")
    print(f"Minimum Noise Figure: {lna.nfmin_db[idx_924mhz]}dB")
    print(f"Gamma_opt: {gamma_opt:.3f} | Mag: {abs(gamma_opt):.3f} | Arg: {np.degrees(np.angle(gamma_opt)):.3f}")
    print(f"Optimal source impedance: {50*z_opt:.3f} | Normalised: {z_opt:.3f}")

def find_gamma_l(lna, zs, freq):
    idx_freq = rf.util.find_nearest_index(lna.f, freq)
    gamma_s = (50 - zs) / (50 + zs)
    s11 = lna.s11.s[idx_freq][0, 0]
    s21 = lna.s21.s[idx_freq][0, 0]
    s12 = lna.s12.s[idx_freq][0, 0]
    s22 = lna.s22.s[idx_freq][0, 0]
    gamma_l = np.conj(s22 + (s21 * gamma_s * s12) / (1 - s11 * gamma_s))
    return gamma_l

def match_gamma_opt_highpass(lna, freq, system=50):
    idx_freq = rf.util.find_nearest_index(lna.f, freq)
    
    gamma_opt = lna.g_opt[idx_freq]
    zs = 50 * (1 + gamma_opt) / (1 - gamma_opt)
    return matched_l_network_highpass(np.conj(zs), system, freq) # match to the conjugate

def match_gamma_opt_conj_highpass(lna, freq, system=50):
    idx_freq = rf.util.find_nearest_index(lna.f, freq)
    
    gamma_opt = lna.g_opt[idx_freq]
    zs = 50 * (1 + gamma_opt) / (1 - gamma_opt)
    return matched_l_network_highpass(system, np.conj(zs), freq)

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

def matched_t_network_highpass(lna, intermediate, freq):
    ind1, cap1 = match_gamma_opt_conj_highpass(lna, freq, intermediate)
    ind2, cap2 = matched_l_network_highpass(intermediate, 50, freq)
    ind = ind1 * ind2 / (ind1 + ind2)
    return cap2, ind, cap1

def matched_pi_output_highpass(zl, intermediate, freq):
    ind1, cap1 = matched_l_network_highpass(np.conj(zl), intermediate, freq)
    ind2, cap2 = matched_l_network_highpass(50, intermediate, freq)
    cap = cap1 * cap2 / (cap1 + cap2)
    return ind1, cap, ind2

def matched_t_output_highpass(zl, intermediate, freq):
    ind1, cap1 = matched_l_network_highpass(intermediate, np.conj(zl), freq)
    ind2, cap2 = matched_l_network_highpass(intermediate, 50, freq)
    ind = ind1 * ind2 / (ind1 + ind2)
    return cap1, ind, cap2

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

def t_network_impedance_highpass(caps, ind, capl, freq):
    xcs = -1j / (2 * np.pi * freq * caps)
    xcl = -1j / (2 * np.pi * freq * capl)
    xl = 2j * np.pi * freq * ind
    z1 = 50 + xcs
    z2 = (z1 * xl) / (z1 + xl)
    z3 = z2 + xcl
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

def shunt_l_series_cap(cap, ind):
    l_react = 2j * np.pi * 924e6 * ind / 50
    c_react = -1j / (2 * np.pi * 924e6 * cap) / 50
    par = 1 * l_react / (1 + l_react)
    total = par + c_react
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

# %%

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

lna = rf.Network("GRF207X_208X_Spars_Noise/GRF2070_2080_5V_70mA_25C.s2p")
analyse_lna(lna)
print(match(lna))

# %%

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

lna = rf.Network("QPL9547_SN1_5V_de-embedded_2.s2p")
analyse_lna(lna)

# %%

print(match(lna))
plot_source_circles(lna, 10)
plot_load_circles(lna, 10)
ind, cap = match_gamma_opt_highpass(lna, 924e6)
print(series_cap_shunt_l(cap, ind))

# %%

# No noise details in s-params file, no reverse isolation data,
# Not amazing gain or NF at 900MHz
lna = rf.Network("PMA2-33LN+_S2P/PMA2-33LN+_AP180688_CE7590__CE7600_S_paramaters_U1.s2p")
analyse_lna(lna)

# %%

ind, cap = match_gamma_opt_highpass(lna, 924e6)
print([ind, cap])
print(series_cap_shunt_l(cap, ind))

# %%

ind1, cap, ind2 = matched_pi_network_highpass(lna, 8.5, 924e6)
print([ind1, cap, ind2])
ind1, ind2 = round(ind1, 10), round(ind2, 10)
cap = round(cap, 13)
print([ind1, cap, ind2])
zs = pi_network_impedance_highpass(ind1, cap, ind2, 924e6)
print(f"zs: {zs}")
gamma_l = find_gamma_l(lna, zs, 924e6)
print(f"gamma_l: {gamma_l}")
print(f"S22*: {np.conj(lna.s22.s[rf.util.find_nearest_index(lna.f, 924.e+6)])}")
zl = 50*(1 + gamma_l) / (1 - gamma_l)
print(f"zl_opt: {zl}")
ind3, cap2, ind4 = matched_pi_output_highpass(zl, 6.35, 924e6)
print([ind3, cap2, ind4])
ind3, ind4 = round(ind3, 10), round(ind4, 10)
cap2 = round(cap2, 13)
print([ind3, cap2, ind4])
zl = pi_network_impedance_highpass(ind4, cap2, ind3, 924e6)
print(f"zl: {zl}")
print(match(lna)) # obsolete but good to reference against
print([ind1, ind2, ind3, ind4, cap, cap2])

# %%

ind, cap = match_gamma_opt_highpass(lna, 924e6, 50)
print([ind, cap])
ind, cap = round(ind, 9), round(cap, 13)
print([ind, cap])
zs = 50*series_cap_shunt_l(cap, ind)
print(zs)
gamma_l = find_gamma_l(lna, zs, 924e6)
print(f"gamma_l: {gamma_l}")
print(f"S22*: {np.conj(lna.s22.s[rf.util.find_nearest_index(lna.f, 924.e+6)])}")
zl = 50*(1 + gamma_l) / (1 - gamma_l)
ind2, cap2 = matched_l_network_highpass(zl, 50, 924e6)
print([ind2, cap2])
ind2, cap2 = round(ind2, 9), round(cap2, 13)
print([ind2, cap2])
print(50*series_cap_shunt_l(cap2, ind2))
print([ind, ind2, cap, cap2])

# %%

ind, cap = match_gamma_opt_highpass(lna, 924e6, 50)
print([ind, cap])
ind, cap = round(ind, 9), round(cap, 13)
print([ind, cap])
zs = 50*series_cap_shunt_l(cap, ind)
print(zs)
cap1, ind, cap2 = matched_t_network_highpass(lna, 150, 924e6)
cap1, cap2 = round(cap1, 13), round(cap2, 13)
ind = round(ind, 10)
print([cap1, ind, cap2])
zs = t_network_impedance_highpass(cap2, ind, cap1, 924e6)
print(zs)

# %%

ind, cap = match_gamma_opt_highpass(lna, 924e6, 50)
print([ind, cap])
ind, cap = round(ind, 9), round(cap, 13)
print([ind, cap])
zs = 50*series_cap_shunt_l(cap, ind)
print(zs)
gamma_l = find_gamma_l(lna, zs, 924e6)
print(f"gamma_l: {gamma_l}")
print(f"S22*: {np.conj(lna.s22.s[rf.util.find_nearest_index(lna.f, 924.e+6)])}")
zl = 50*(1 + gamma_l) / (1 - gamma_l)
print(zl)
cap1, ind2, cap2 = matched_t_output_highpass(zl, 150, 924e6)
print([cap1, ind2, cap2])
cap1, ind2, cap2 = round(cap1, 13), round(ind2, 9), round(cap2, 13)
print([cap1, ind2, cap2])
zl = t_network_impedance_highpass(cap2, ind2, cap1, 924e6)
print(zl)
zl = t_network_impedance_highpass(cap2+2e-13, ind2, cap1, 924e6)
print(zl)
zl = t_network_impedance_highpass(cap2-2e-13, ind2, cap1, 924e6)
print(zl)
zl = t_network_impedance_highpass(cap2, ind2+2e-10, cap1, 924e6)
print(zl)
zl = t_network_impedance_highpass(cap2, ind2, cap1+2e-13, 924e6)
print(zl)
zl = t_network_impedance_highpass(cap2, ind2, cap1-2e-13, 924e6)
print(zl)

# %%

zs = 50 + -1j / (2 * np.pi * 924e6 * 68e-12)
print(zs)
gamma_l = find_gamma_l(lna, zs, 924e6)
print(f"gamma_l: {gamma_l}")
print(f"S22*: {np.conj(lna.s22.s[rf.util.find_nearest_index(lna.f, 924.e+6)])}")
zl = 50*(1 + gamma_l) / (1 - gamma_l)
print(zl)
cap1, ind2, cap2 = matched_t_output_highpass(zl, 150, 924e6)
print([cap1, ind2, cap2])
cap1, ind2, cap2 = round(cap1, 13), round(ind2, 9), round(cap2, 13)
print([cap1, ind2, cap2])
zl = t_network_impedance_highpass(cap2, ind2, cap1, 924e6)
print(zl)
zl = t_network_impedance_highpass(cap2+2e-13, ind2, cap1, 924e6)
print(zl)
zl = t_network_impedance_highpass(cap2-2e-13, ind2, cap1, 924e6)
print(zl)
zl = t_network_impedance_highpass(cap2, ind2+2e-10, cap1, 924e6)
print(zl)
zl = t_network_impedance_highpass(cap2, ind2, cap1+2e-13, 924e6)
print(zl)
zl = t_network_impedance_highpass(cap2, ind2, cap1-2e-13, 924e6)
print(zl)
print([cap1, ind2, cap2])

# %%

ind, cap = match_gamma_opt_highpass(lna, 924e6, 50)
print([ind, cap])
ind, cap = round(ind, 9), round(cap, 13)
print([ind, cap])
zs = 50*series_cap_shunt_l(cap, ind)
print(f"zs: {zs}")
zs = zs + -1j / (2 * np.pi * 924e6 * 68e-12)
print(f"zs: {zs}")
zs = zs / 50
print(f"zs: {zs}")
gamma_l = find_gamma_l(lna, zs, 924e6)
print(f"gamma_l: {gamma_l}")
print(f"S22*: {np.conj(lna.s22.s[rf.util.find_nearest_index(lna.f, 924.e+6)])}")
zl = 50*(1 + gamma_l) / (1 - gamma_l)
print(zl)
cap1, ind2, cap2 = matched_t_output_highpass(zl, 170, 924e6)
print([cap1, ind2, cap2])
cap1, ind2, cap2 = round(cap1, 13), round(ind2, 9), round(cap2, 13)
print([cap1, ind2, cap2])
zl = t_network_impedance_highpass(cap2, ind2, cap1, 924e6)
print(zl)
zl = t_network_impedance_highpass(cap2+2e-13, ind2, cap1, 924e6)
print(zl)
zl = t_network_impedance_highpass(cap2-2e-13, ind2, cap1, 924e6)
print(zl)
zl = t_network_impedance_highpass(cap2, ind2+2e-10, cap1, 924e6)
print(zl)
zl = t_network_impedance_highpass(cap2, ind2, cap1+2e-13, 924e6)
print(zl)
zl = t_network_impedance_highpass(cap2, ind2, cap1-2e-13, 924e6)
print(zl)
print([cap1, ind2, cap2])

# %%

# ind, cap = match_gamma_opt_highpass(lna, 924e6, 50)
idx_freq = rf.util.find_nearest_index(lna.f, 924e6)
gamma_opt = lna.g_opt[idx_freq]
zs = 50 * (1 + gamma_opt) / (1 - gamma_opt)
zs += 15j
ind, cap = matched_l_network_highpass(np.conj(zs), 50, 924e6) # match to the conjugate
print([ind, cap])
ind, cap = round(ind, 9), round(cap, 13)
print([ind, cap])
zs = 50*series_cap_shunt_l(cap, ind)
print(f"zs: {zs}")
gamma_l = find_gamma_l(lna, zs, 924e6)
print(f"gamma_l: {gamma_l}")
print(f"S22*: {np.conj(lna.s22.s[rf.util.find_nearest_index(lna.f, 924.e+6)])}")
zl = 50*(1 + gamma_l) / (1 - gamma_l)
print(zl)
ind2, cap2 = matched_l_network_highpass(zl, 50, 924e6)
print([ind2, cap2])
ind2, cap2 = round(ind2, 9), round(cap2, 13)
print([ind2, cap2])
print(50*series_cap_shunt_l(cap2, ind2))
print([ind, ind2, cap, cap2])

# %%

zs = 50 + -2j / (2 * np.pi * 924e6 * 68e-12)
print(f"zs: {zs}")
gamma_l = find_gamma_l(lna, zs, 924e6)
print(f"gamma_l: {gamma_l}")
print(f"S22*: {np.conj(lna.s22.s[rf.util.find_nearest_index(lna.f, 924.e+6)])}")
zl = 50*(1 + gamma_l) / (1 - gamma_l)
print(f"zl_opt: {zl}")
ind2, cap2 = matched_l_network_highpass(np.conj(zl + 15), 50, 924e6)
print([ind2, cap2])
ind2, cap2 = round(ind2, 9), round(cap2, 13)
print([ind2, cap2])
print(50*series_cap_shunt_l(cap2, ind2))
print([ind, ind2, cap, cap2])
zl = pi_network_impedance_highpass(ind4+2e-10, cap2, ind3, 924e6)
print(f"zl: {zl}")
zl = pi_network_impedance_highpass(ind4, cap2+2e-13, ind3, 924e6)
print(f"zl: {zl}")
zl = pi_network_impedance_highpass(ind4, cap2, ind3+2e-10, 924e6)
print(f"zl: {zl}")
zl = pi_network_impedance_highpass(ind4-2e-10, cap2, ind3, 924e6)
print(f"zl: {zl}")
zl = pi_network_impedance_highpass(ind4, cap2-2e-13, ind3, 924e6)
print(f"zl: {zl}")
zl = pi_network_impedance_highpass(ind4, cap2, ind3-2e-10, 924e6)
print(f"zl: {zl}")
print([ind3, ind4, cap2])

# %%

xl = 2j * np.pi * 924e6 * 47e-6
xc1 = -1j / (2 * np.pi * 924e6 * 1e-6)
xc2 = -1j / (2 * np.pi * 924e6 * 200e-12)

zp = xl + xc1 * xc2 / (xc1 + xc2)
print (zp)

# %%

dc_block = rf.Network("Tests/DUT-Single-Merged-68pF-600MHz-1.2GHz.s2p")
z = dc_block.s11.z[rf.util.find_nearest_index(dc_block.f, 924.e+6)][0, 0]
print(z)
print(50*series_cap_shunt_l(3.3e-12, 12e-9))
print(50*series_cap_shunt_l(3.6e-12, 12e-9))

# %%

# ground = rf.Circuit.Ground(lna.f, "ground")
# port_lna = rf.Circuit.Port(lna.f, name = 'port_lna', z0=50)
# port_source = rf.Circuit.Port(lna.f, name = 'port_source', z0=50)
cap = rf.Network("Passives/GJM1555C1H5R3WB01_DC0V_25degC_series.s2p").interpolate(lna.f)
ind = rf.Network("Passives/LQW15AN16NG80_shunt.s2p").interpolate(lna.f)
media = rf.DefinedGammaZ0(lna.f, z0=50)
res = media.resistor(R=50)
# connections = [
#     [(ind, 1), (ground, 0)],
#     [(port_lna, 0), (ind, 0), (cap, 0)]
# ]
print(lna)

combined_input = ind ** cap ** res
print(combined_input)
combined_input.plot_s_db()
idx_924mhz = rf.util.find_nearest_index(lna.f, 924.e+6)
s11 = lna.s11.s[idx_924mhz][0, 0]
print(f"S11: {s11:.3f}dB")
z = lna.z[idx_924mhz][0, 0]
print(f"Z: {z:.3f}dB")

# %%

matcher = rf.Network("Tests/DUT-L-Merged-16nH-5.3pF-600MHz-1.2GHz.s2p")
dc_block = rf.Network("Tests/DUT-Single-Merged-68pF-600MHz-1.2GHz.s2p")
print(matcher)
dc_block.plot_s_smith()
idx_924mhz = rf.util.find_nearest_index(matcher.f, 924.e+6)
print(f"Z: {matcher.s11.z[idx_924mhz][0, 0]}")
print(f"Z: {dc_block.s11.z[idx_924mhz][0, 0]}")
#%%
combined = dc_block ** dc_block
print(f"Z: {matcher.s11.z[idx_924mhz][0, 0]}")
print(f"Z: {dc_block.s11.z[idx_924mhz][0, 0]}")
print(f"Z: {combined.s11.z[idx_924mhz][0, 0]}")
print(f"S11: {matcher.s11.s[idx_924mhz][0, 0]}")
print(f"S11: {dc_block.s11.s[idx_924mhz][0, 0]}")
print(f"S11: {combined.s11.s[idx_924mhz][0, 0]}")
print(f"Loss: {matcher.s21.s_db[idx_924mhz][0, 0]}")
print(f"Loss: {dc_block.s21.s_db[idx_924mhz][0, 0]}")
print(f"Loss: {combined.s21.s_db[idx_924mhz][0, 0]}")

idx_freq = rf.util.find_nearest_index(lna.f, 924.e+6)
rn = lna.rn[idx_freq]/50
gamma_opt = lna.g_opt[idx_freq]
fmin = lna.nfmin[idx_freq]

plt.close()
ax = plt.axes()
for nf_added in [0.01, 0.02, 0.05]:
    nf = 10**(nf_added/10) * fmin

    N = (nf - fmin)*abs(1+gamma_opt)**2/(4*rn)
    c_n = gamma_opt/(1+N)
    r_n = 1/(1-N)*np.sqrt(N**2 + N*(1-abs(gamma_opt)**2))

    n = rf.Network(name=str(nf_added), s=calc_circle(c_n, r_n))
    n.plot_s_smith()
ax.set_title(lna.name)
matcher.s11[idx_924mhz-5:idx_924mhz+5].plot_s_smith()
dc_block.s11[idx_924mhz-5:idx_924mhz+5].plot_s_smith()
combined.s11[idx_924mhz-5:idx_924mhz+5].plot_s_smith()

gamma = combined.s11.z[idx_924mhz]
print(gamma)

# %%