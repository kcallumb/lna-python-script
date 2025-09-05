#!/usr/bin/python3

"""
    Takes two S2P files, a forward and reverse configuration of the same
    network, and merges them into a single file. For recording S12 and S22
    on VNAs with a single output port.
"""

FORWARD_S2P = 'Tests/DUT-Single-68pF-600MHz-1.2GHz.s2p'
REVERSE_S2P = 'Tests/DUT-Single-Reversed-68pF-600MHz-1.2GHz.s2p'
OUTPUT_FILENAME = 'Tests/DUT-Single-Merged-68pF-600MHz-1.2GHz.s2p'

with open(FORWARD_S2P) as forward:
    forward_lines = forward.readlines()

with open(REVERSE_S2P) as reverse:
    reverse_lines = reverse.readlines()

with open(OUTPUT_FILENAME, 'w') as output:
    # Write header and comment.
    output.write('!Stimulus Real(S11) Imag(S11) Real(S21) Imag(S21) Real(S12) '
                 + 'Imag(S12) Real(S22) Imag(S22)\n')
    output.write('# HZ S RI R 50\n')

    for i, forward_line in enumerate(forward_lines[1:]):
        # Add the forward components, S11 and S12.
        line = forward_line.strip('\n').split(' ')
        line = line[:5]

        # Add the reverse components, S21 and S22.
        # The order of these must be swapped before being appended.
        reverse_line = reverse_lines[i + 1].strip('\n').split(' ')
        line += reverse_line[3:5]
        line += reverse_line[1:3]
        output.write(' '.join(line) + '\n')
