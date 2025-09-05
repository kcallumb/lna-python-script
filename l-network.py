# %%
# Load networks from file

import skrf as rf

dc_block = rf.Network('Tests/DUT-Single-Merged-68pF-600MHz-1.2GHz.s2p')
l_network = rf.Network('Tests/DUT-L-Merged-16nH-5.3pF-600MHz-1.2GHz.s2p')

print(dc_block)
print(l_network)

# %%
# Combine networks
network = dc_block ** l_network
network.name = 'T network'
print(network)

# %%
# Get impedance at 924MHz from s-parameters

# Find closest index to 924MHz
i = rf.util.find_nearest_index(network.f, 924e6)

# Find Z_11 at that index
Z = network.z[i]
Z_lna = Z[1, 1]

print(Z_lna)

# %%
