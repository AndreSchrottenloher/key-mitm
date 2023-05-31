#!/usr/bin/python3
# -*- coding: utf-8 -*-

#=========================================================================
#Copyright (c) 2023

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.

#=========================================================================

#This project has been partially supported by ERC-ADG-ALGSTRONGCRYPTO (project 740972),
#and partially supported by the French Agence Nationale de la Recherche through 
#the DeCrypt project under Contract ANR-18-CE39-0007, and through the France 2030 
#program under grant agreement No. ANR-22-PETQ-0008 PQ-TLS.

#=========================================================================

# REQUIREMENT LIST
#- Python 3.x with x >= 2
#- the SCIP solver, see https://scip.zib.de/ OR the Gurobi solver
#- pyscipopt (https://github.com/SCIP-Interfaces/PySCIPOpt) OR gurobipy

#=========================================================================

# Author: André Schrottenloher & Marc Stevens
# Date: March 2023
# Version: 1

#=========================================================================
"""
This file includes the code that generates the models, finds the attacks
given in the paper (including the automatic generation of LateX / Tikz pictures,
that help in visualizing the attack paths).

To use it, run:

python3 attacks.py attack_name

Where "attack_name" is one of the specified cases. To test the code, run:

python3 attacks.py test

This will check that the expected time complexity is found when solving all
the models (this will also take some time). 

Some of the attacks generate large pictures, which are automatically "simplified".
To generate the full pictures instead, run:

python3 attacks.py attack_name full

For more details on each specific attacks, see the paper. 
"""

from util import Cipher, rotate_right, rotate_left
from util import write_and_compile
from modeling import CLASSICAL_GAD, CLASSICAL_PARALLEL, CLASSICAL_PREIMAGE
from modeling import QUANTUM_GAD, QUANTUM_PARALLEL, QUANTUM_SIMON, QUANTUM_PREIMAGE
from modeling import find_mitm_attack


def make_future_constraints(nrounds=10):
    """Constraints for the FUTURE block cipher"""
    permbits = [
        0, 1, 2, 3, 52, 53, 54, 55, 40, 41, 42, 43, 28, 29, 30, 31, 16, 17, 18,
        19, 4, 5, 6, 7, 56, 57, 58, 59, 44, 45, 46, 47, 32, 33, 34, 35, 20, 21,
        22, 23, 8, 9, 10, 11, 60, 61, 62, 63, 48, 49, 50, 51, 36, 37, 38, 39,
        24, 25, 26, 27, 12, 13, 14, 15
    ]

    cipher = Cipher(state_size=64, key_register_sizes=[64, 64])
    for r in range(nrounds):
        for i in range(64):
            cipher.add_key_bit(i, i, register=(r % 2))  # start with key 0
        if r % 2 == 0:
            cipher.key_rotate_left(5, register=0)
        else:
            cipher.key_rotate_left(5, register=1)
        #  SC- MC
        if r < nrounds - 1:
            for i in range(4):
                cipher.sbox(16 * i, 16, super_sbox=True)
        else:
            # last round: no MC
            for i in range(16):
                cipher.sbox(4 * i, 4, super_sbox=False, display_name=False)
        # SR
        cipher.permute_inv(permbits)
    # last round: key addition
    for i in range(64):
        cipher.add_key_bit(i, i, register=(nrounds % 2))
    return cipher


def make_pipo_constraints(nrounds=5, flag="pipo-128"):
    """
    Constraints for the PIPO block cipher. For the linear layer, we used
    an arbitrary permutation that distributes the bits among the S-Boxes. All such
    permutations are structurally equivalent, up to a permutation of the key bits.
    """
    permbits = [
        0, 9, 18, 27, 36, 45, 54, 63, 8, 17, 26, 35, 44, 53, 62, 7, 16, 25, 34,
        43, 52, 61, 6, 15, 24, 33, 42, 51, 60, 5, 14, 23, 32, 41, 50, 59, 4,
        13, 22, 31, 40, 49, 58, 3, 12, 21, 30, 39, 48, 57, 2, 11, 20, 29, 38,
        47, 56, 1, 10, 19, 28, 37, 46, 55
    ]
    if flag == "pipo-128":
        nb_keys = 2
    elif flag == "pipo-256":
        nb_keys = 4
    else:
        raise ValueError("Wrong flag")
    cipher = Cipher(state_size=64, key_register_sizes=[64] * nb_keys)
    for r in range(nrounds):
        for i in range(64):
            cipher.add_key_bit(i, i,
                               register=(r % nb_keys))  # start with key 0
        for i in range(8):
            cipher.sbox(8 * i, 8)  # 8-bit S-Boxes
        cipher.permute_inv(permbits)
    # finish last round with a key addition
    for i in range(64):
        cipher.add_key_bit(i, i, register=(nrounds % nb_keys))
    return cipher


def make_aes_constraints(nrounds=5,
                         flag="aes-2k",
                         no_mc_at_last_round=False,
                         no_sr_at_last_round=False):
    """Generates constraints for an AES variant with simple key-schedule."""
    if flag == "aes-2k":
        nb_keys = 2
    elif flag == "aes-1k":
        nb_keys = 1
    else:
        raise ValueError("Wrong flag")
    permbits = [0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3]
    cipher = Cipher(state_size=16, key_register_sizes=[16] * nb_keys)
    for r in range(nrounds):
        for i in range(16):
            cipher.add_key_bit(i, i, register=(r % nb_keys))
        # SR
        if (not no_sr_at_last_round) or r < nrounds - 1:
            cipher.permute(permbits)

        if (not no_mc_at_last_round) or r < nrounds - 1:
            # SB followed by MC
            for i in range(4):
                cipher.sbox(4 * i, 4, super_sbox=True)
        # no MC at the last round
        else:
            # no_mc_at_last_round = True and r == nrounds-1
            for i in range(16):
                cipher.sbox(i, 1, super_sbox=False, display_name=False)

    for i in range(16):
        cipher.add_key_bit(i, i, register=(nrounds % nb_keys))

    return cipher


def make_saturnin_constraints(nrounds=5, no_mc_at_last_round=False):
    """Generates constraints for Saturnin (block cipher or compression function)."""
    # In Saturnin, the internal state is represented as a matrix
    # and the supernibbles are numbered in the matrix as:
    # 0 1 2 3
    # 4 5 6 7
    # 8 9 10 11
    # 12 13 14 15
    # transposition of supernibbles
    permbits = [0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]
    cipher = Cipher(state_size=16, key_register_sizes=[16])
    shift = 5
    for r in range(nrounds - 1):
        for i in range(16):
            if r % 2 == 0:
                # even rounds: add key
                cipher.add_key_bit(i, i, register=0)
            else:
                # odd rounds: add transposed, shifted key
                cipher.add_key_bit((i + shift) % 16, i, register=0)
        # even rounds: apply Super S-Box on the rows
        if r % 2 == 0:
            for i in range(4):
                cipher.sbox(4 * i, 4, super_sbox=True)
        else:
            # odd rounds: apply Super S-Box on the columns
            cipher.permute(permbits)
            for i in range(4):
                cipher.sbox(4 * i, 4, super_sbox=True)
            cipher.permute(permbits)
    # last round: apply simple s-boxes
    r = nrounds - 1
    for i in range(16):
        if r % 2 == 0:
            # even rounds: add key
            cipher.add_key_bit(i, i, register=0)
        else:
            # odd rounds: add shifted key
            cipher.add_key_bit((i + shift) % 16, i, register=0)
    if no_mc_at_last_round:
        for i in range(16):
            cipher.sbox(i, 1, super_sbox=False, display_name=False)
    else:
        if r % 2 == 0:
            for i in range(4):
                cipher.sbox(4 * i, 4, super_sbox=True)
        else:
            # odd rounds: apply Super S-Box on the columns
            cipher.permute(permbits)
            for i in range(4):
                cipher.sbox(4 * i, 4, super_sbox=True)
            cipher.permute(permbits)
    r = nrounds
    for i in range(16):
        if r % 2 == 0:
            # even rounds: add key
            cipher.add_key_bit(i, i, register=0)
        else:
            # odd rounds: add shifted key
            cipher.add_key_bit((i + shift) % 16, i, register=0)
    return cipher


def make_gift64_constraints(nrounds=5):
    """Generates constraints for Gift-64."""
    permbits64 = [
        0, 17, 34, 51, 48, 1, 18, 35, 32, 49, 2, 19, 16, 33, 50, 3, 4, 21, 38,
        55, 52, 5, 22, 39, 36, 53, 6, 23, 20, 37, 54, 7, 8, 25, 42, 59, 56, 9,
        26, 43, 40, 57, 10, 27, 24, 41, 58, 11, 12, 29, 46, 63, 60, 13, 30, 47,
        44, 61, 14, 31, 28, 45, 62, 15
    ]

    gift64 = Cipher(state_size=64, key_register_sizes=[16] * 8)

    #print(gift64.get_state())
    for r in range(nrounds):
        for i in range(16):
            gift64.sbox(4 * i, 4)

        gift_key = gift64.key_registers
        for i in range(16):
            gift64.add_key_bit(i, 4 * i, register=0)
            gift64.add_key_bit(i, 4 * i + 1, register=1)
        tmp1 = gift_key[7]
        tmp2 = gift_key[6]
        gift_key[7] = rotate_right(gift_key[1], 2)
        gift_key[6] = rotate_right(gift_key[0], 12)
        gift_key[0] = gift_key[2]
        gift_key[1] = gift_key[3]
        gift_key[2] = gift_key[4]
        gift_key[3] = gift_key[5]
        gift_key[4] = tmp2
        gift_key[5] = tmp1
        gift64.permute(permbits64)

    return gift64


def make_present_constraints(nrounds=6, flag="present-80"):
    """Generates constraints for Present."""
    if flag not in ["present-80", "present-128"]:
        raise ValueError("Wrong flag")

    # the key bits in the register are numbered from 0 to 79 / 127 in increasing
    # order, so it corresponds to the reverse of the cipher's specification.
    present = Cipher(
        state_size=64,
        key_register_sizes=([80] if flag == "present-80" else [128]))

    width = 16
    b = 4 * width
    permbits = [
        0, 16, 32, 48, 1, 17, 33, 49, 2, 18, 34, 50, 3, 19, 35, 51, 4, 20, 36,
        52, 5, 21, 37, 53, 6, 22, 38, 54, 7, 23, 39, 55, 8, 24, 40, 56, 9, 25,
        41, 57, 10, 26, 42, 58, 11, 27, 43, 59, 12, 28, 44, 60, 13, 29, 45, 61,
        14, 30, 46, 62, 15, 31, 47, 63
    ]

    # first key addition
    for i in range(b):
        present.add_key_bit(-1 - i, i,
                            register=0)  # only one key register anyway

    for r in range(nrounds):
        # apply S-Box layer
        for i in range(width):
            present.sbox(4 * i, 4)
        # apply bit permutation, except at the last round
        if r < nrounds - 1:
            present.permute(permbits)

        # key schedule
        if flag == "present-80":
            # rotate key register
            present.key_rotate_left(19, register=0)
            # apply S-Box on 4 first bits
            present.key_sbox([-1, -2, -3, -4], register=0)
        else:
            # rotate key register
            present.key_rotate_right(61, register=0)
            # apply S-Box on 4 first bits, and 4 next bits
            present.key_sbox([-1, -2, -3, -4], register=0)
            present.key_sbox([-5, -6, -7, -8], register=0)
        # XOR key (the 64 first bits)
        for i in range(b):
            present.add_key_bit(-1 - i, i, register=0)

    return present


def make_haraka256_constraints(nrounds=5):
    """
    Generates constraints for Haraka-256.
    nrounds -- number of AES rounds (corresponds to half-rounds in Haraka)
    """
    # 1 goes to 13
    permbits_sr = [0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3]
    full_permbits_sr = permbits_sr + [t + 16 for t in permbits_sr]
    permcols_mix = [0, 2, 4, 6, 1, 3, 5,
                    7]  # permutation of the columns during
    # MIX operation. Column 4 goes to 1.
    permbits_mix = sum([[i + 4 * t for i in range(4)] for t in permcols_mix],
                       [])
    cipher = Cipher(state_size=32, key_register_sizes=[])  # no key
    for r in range(nrounds):
        if r % 2 == 0:
            # AES round 1
            # SR
            cipher.permute(full_permbits_sr)
            # SB - MC
            for i in range(8):
                cipher.sbox(4 * i, 4, super_sbox=True)
            # AES round 2
        else:
            cipher.permute(full_permbits_sr)
            # SB - MC
            for i in range(8):
                cipher.sbox(4 * i, 4, super_sbox=True)
            # MIX operation
            cipher.permute(permbits_mix)

    return cipher


def make_haraka512_constraints(nrounds=5):
    """
    Generates constraints for Haraka-512.
    nrounds -- number of AES rounds (corresponds to half-rounds in Haraka)
    """
    # 1 goes to 13
    permbits_sr = [0, 13, 10, 7, 4, 1, 14, 11, 8, 5, 2, 15, 12, 9, 6, 3]
    full_permbits_sr = (permbits_sr + [t + 16 for t in permbits_sr] +
                        [t + 32
                         for t in permbits_sr] + [t + 48 for t in permbits_sr])
    permcols_mix = [5, 9, 12, 0, 7, 11, 14, 2, 4, 8, 13, 1, 6, 10, 15, 3]
    #permcols_mix = [3, 11, 7, 15, 8, 0, 12, 4, 9, 1, 13, 5, 2, 10, 6, 14]
    # MIX operation. Column 3 goes to 0
    permbits_mix = sum([[i + 4 * t for i in range(4)] for t in permcols_mix],
                       [])
    cipher = Cipher(state_size=64, key_register_sizes=[])  # no key
    for r in range(nrounds):
        if r % 2 == 0:
            # AES round 1 # SR
            cipher.permute(full_permbits_sr)
            # SB - MC
            for i in range(16):
                cipher.sbox(4 * i, 4, super_sbox=True)
            # AES round 2 # SR
        else:
            cipher.permute(full_permbits_sr)
            # SB - MC
            for i in range(16):
                cipher.sbox(4 * i, 4, super_sbox=True)
            # MIX operation
            cipher.permute(permbits_mix)
    wrapping_columns = [2, 3, 6, 7, 8, 9, 12, 13]
    cipher.set_partial_wrapping(
        sum([[i + 4 * t for i in range(4)] for t in wrapping_columns], []))

    return cipher


_HELP = """
Finding MITM attacks with our MILP tool. This code includes all the attacks
given in the paper. Simply run:

python3 attacks.py attack_name

It will run the optimization and find the attack. A LaTeX document (standalone)
will be created in a new folder "results" in the current folder. It will then
be compiled. The parameters of the MITM path are printed, and included as 
comments in the LaTeX document.

By default the solver used is Gurobi, but this can be modified in the code
(like the parameters of the search). Not all optimizations will terminate in 
a reasonable time, but the solver can be stopped by CTRL+C.

Some of the attacks generate large pictures, which are automatically "simplified".
To generate the full pictures instead, run:

python3 attacks.py attack_name full

"attack_name" should be one of these, which form a superset of the attacks
discussed in the paper:
* aes-preimage
    -> quantum preimage attack on a 7-round compression function based on AES, recovering
    a result of [SS22]
* saturnin-preimage-classical
    -> classical preimage attack on 7 super-round SATURNIN
* saturnin-preimage-quantum
    -> quantum version of this attack
* aes-1k
    -> classical key-recovery attack on a 7-round AES with repeating key (no key schedule)
* aes-1k-smalldata
    -> same, but with reduced data complexity
* aes-2k
    -> classical key-recovery attack on a 10-round AES with two alternating keys (no key schedule)
* future
    -> key-recovery attack on 10-round FUTURE
* saturnin-classical
    -> classical key-recovery attack on 6.5 super-round SATURNIN
* saturnin-quantum
    -> quantum key-recovery attack on 6.5 super-round SATURNIN
* gitf-64-gad
    -> key-recovery attack on 15-round Gift, recovering a result of Sasaki (IWSEC 2018)
* present-80-lowdata
    -> low-data key-recovery attack on 9-round Present
* present-80-lowdata-quantum
    -> same, in the quantum setting
* pipo-128
    -> 10-round attack on PIPO-128 (= 11-round attack on FLY)
* pipo-256
    -> 18-round attack on PIPO-256
* haraka-256
    -> recovering previous results on preimage attacks on Haraka-256
* haraka-512-11
    -> improving the time complexity of previous works (see [SS22]) for the
    5.5-round (11 AES rounds) Haraka-512
* haraka-512-13
    -> preimage attack on 6.5-round (13 AES rounds) Haraka-512
* saturnin-simon
    -> Grover-meet-Simon attack on 5.5 super-round SATURNIN
* gift-64-simon
    -> Grover-meet-Simon attack on 15-round Gift
* test
    -> will run all the searches and test that they find the expected time
    complexity.

"""


class AttackParams:
    """ Parameters of a search. """

    def __init__(self):
        # parameters for the searches
        self.cut_forward, self.cut_backward = [], []
        self.forward_hint, self.backward_hint = [], []
        self.forward_zero, self.backward_zero = [], []
        self.not_bwd_key, self.not_fwd_key, self.shared_key = [], [], []
        self.minimize_data = False
        self.data_limit, self.memory_limit = None, None
        self.covered_rounds = [
        ]  # don't forget that in key-recovery cases, the covered round
        # can be easily set to be the encryption function (single cell spanning a whole round)
        self.computation_model = CLASSICAL_PARALLEL
        self.file_name = None
        self.round_bits = False
        self.time_target = None
        self.optimize_with_mem = True

        #============ picture parameters
        self.key_prev = []  # changes the way the key bits are displayed (at
        # the top or at the bottom of an edge)
        self.simplified = False  # simplifies the picture, just to get a broad overview


def _parameters(attack):
    """
    The parameters of each of our models.
    """
    p = AttackParams()

    if attack == "aes-preimage":
        # recover a quantum preimage attack on 7-round AES, by fixing the key
        p.computation_model, p.covered_rounds = QUANTUM_PREIMAGE, [7]
        p.shared_key = [('$k_{%i}$' % i)
                        for i in range(16)]  # no degree of freedom in the key
        cipher = make_aes_constraints(nrounds=7,
                                      flag="aes-1k",
                                      no_mc_at_last_round=True,
                                      no_sr_at_last_round=False)
        cipher.preimage_attack = True

    if attack == "saturnin-preimage-classical":
        # classical preimage attack on 7 super-rounds of Saturnin
        p.computation_model, p.covered_rounds = CLASSICAL_PREIMAGE, [1]
        cipher = make_saturnin_constraints(nrounds=7,
                                           no_mc_at_last_round=False)
        cipher.preimage_attack = True
        p.key_prev = [1, 3, 5, 7]  # to make the drawing nicer
        # some hints (just to accelerate convergence)
        p.covered_rounds = [1, 2, 3, 4, 5]
        p.cut_forward, p.cut_backward = [1], [0, 1, 5, 6, 7]
        # some hints (just to recover the same path, and not a symmetric one)
        p.not_fwd_key = [
            '$k_{5}$', '$k_{6}$', '$k_{7}$', '$k_{8}$', '$k_{9}$', '$k_{10}$'
        ]
        p.not_bwd_key = [
            '$k_{0}$', '$k_{4}$', '$k_{12}$', '$k_{13}$', '$k_{14}$',
            '$k_{15}$'
        ]

    if attack == "saturnin-preimage-quantum":
        # quantum preimage attack on 7 super-rounds of Saturnin
        # Also demonstrates a classical attack with small memory (2^32)
        p.computation_model, p.covered_rounds = QUANTUM_PREIMAGE, [1]
        cipher = make_saturnin_constraints(nrounds=7,
                                           no_mc_at_last_round=False)
        p.key_prev = [1, 3, 5, 7]
        p.cut_backward = [0, 6, 7]
        cipher.preimage_attack = True

    if attack == "aes-1k":
        p.computation_model, p.covered_rounds = CLASSICAL_GAD, [7]
        cipher = make_aes_constraints(nrounds=7,
                                      flag="aes-1k",
                                      no_mc_at_last_round=True,
                                      no_sr_at_last_round=True)
        # with or without sr at last round, it's the same
        p.minimize_data = True
        p.covered_rounds += [0, 1]
        p.cut_backward, p.cut_forward = [2, 3, 4], [4, 5, 6, 7]

    if attack == "aes-1k-smalldata":
        p.computation_model, p.covered_rounds = CLASSICAL_GAD, [7]
        cipher = make_aes_constraints(nrounds=7,
                                      flag="aes-1k",
                                      no_mc_at_last_round=True,
                                      no_sr_at_last_round=True)
        # with or without sr at last round, it's the same
        p.minimize_data = True
        p.data_limit = 4
        p.covered_rounds += [0, 1]
        p.cut_backward = [2, 3]
        #cut_forward = [4,5,6,7]
        # only to recover the same path (not a symmetric one)
        p.not_fwd_key = ['$k_{4}$']
        p.not_bwd_key = ['$k_{12}$', '$k_{13}$', '$k_{15}$']

    if attack == "aes-2k":
        p.computation_model, p.covered_rounds = CLASSICAL_GAD, [10]
        cipher = make_aes_constraints(nrounds=10,
                                      flag="aes-2k",
                                      no_mc_at_last_round=False)
        # attack requires full codebook
        #minimize_data = True
        p.cut_forward = [0, 8, 9, 10]
        p.cut_backward = [2, 3, 4, 5]
        p.covered_rounds += [0, 1, 2]

    if attack == "future":
        p.computation_model, p.covered_rounds = CLASSICAL_GAD, [10]
        cipher = make_future_constraints(nrounds=10)
        p.key_prev = []
        p.not_fwd_key = ["$k^{%i}_{%i}$" % (0, i) for i in range(64)]
        p.round_bits = True
        # the best complexity we found is 124, but it yields a very complicated path
        p.covered_rounds += [0, 1, 2, 3, 9, 10]
        p.cut_forward, p.cut_backward = [0, 7, 8, 9, 10], [2, 3, 4, 5, 6]
        p.forward_hint = [
            '$c^{1}_{0}$', '$c^{1}_{2}$', '$c^{1}_{3}$', '$c^{2}_{0}$',
            '$c^{2}_{1}$', '$c^{2}_{2}$', '$c^{3}_{0}$', '$c^{3}_{1}$',
            '$c^{3}_{2}$', '$c^{3}_{3}$', '$c^{4}_{1}$', '$c^{4}_{2}$',
            '$c^{4}_{3}$', '$c^{5}_{0}$', '$c^{5}_{1}$'
        ]
        p.backward_hint = [
            '$c^{0}_{0}$', '$c^{0}_{1}$', '$c^{0}_{2}$', '$c^{0}_{3}$',
            '$c^{1}_{1}$', '$c^{7}_{0}$', '$c^{7}_{1}$', '$c^{7}_{2}$',
            '$c^{8}_{0}$', '$c^{8}_{2}$', '$c^{8}_{3}$', '$c^{9}_{0}$',
            '$c^{9}_{1}$', '$c^{9}_{2}$', '$c^{9}_{3}$', '$c^{9}_{4}$',
            '$c^{9}_{5}$', '$c^{9}_{6}$', '$c^{9}_{7}$', '$c^{9}_{8}$',
            '$c^{9}_{9}$', '$c^{9}_{10}$', '$c^{9}_{11}$', '$c^{9}_{12}$',
            '$c^{9}_{13}$', '$c^{9}_{14}$', '$c^{9}_{15}$', '$c^{10}_{0}$'
        ]

    if attack == "saturnin-classical":
        p.computation_model, p.covered_rounds = CLASSICAL_GAD, [7]
        cipher = make_saturnin_constraints(nrounds=7, no_mc_at_last_round=True)
        p.minimize_data = True
        p.memory_limit = 1.5  # find an attack advantageous in memory
        p.cut_backward = [2, 3, 4]
        p.key_prev = [1, 3, 5]
        p.covered_rounds += [0, 1]

    if attack == "saturnin-quantum":
        p.computation_model, p.covered_rounds = QUANTUM_GAD, [7]
        cipher = make_saturnin_constraints(nrounds=7, no_mc_at_last_round=True)
        p.minimize_data = True
        p.cut_backward = [2, 3, 4]
        p.key_prev = [1, 3, 5]
        p.covered_rounds += [0, 1]
        # only to recover the same path (not a symmetric one)
        p.shared_key = [
            "$k_{%i}$" % i for i in [0, 1, 2, 3, 4, 5, 7, 8, 9, 11, 13, 15]
        ]
        p.backward_hint = [
            '$c^{%i}_{%i}$' % (i, j)
            for (i, j) in [(0, 0), (0, 1), (0, 2), (5, 1), (5, 2)]
        ]
        p.forward_hint = [
            '$c^{%i}_{%i}$' % (i, j)
            for (i, j) in [(0, 3), (1, 0), (1, 1), (1, 3), (3, 0), (3,
                                                                    2), (3, 3)]
        ]

    if attack == "gift-64-gad":
        p.computation_model, p.covered_rounds = CLASSICAL_GAD, [15]
        cipher = make_gift64_constraints(nrounds=15)
        p.cut_forward = [10, 11, 12, 13, 14, 15]
        p.cut_backward = [6, 7, 8, 9]
        p.covered_rounds += [0, 1, 2, 3, 4, 5, 6, 12, 13, 14, 15]
        p.not_fwd_key = sum([['$k^{%i}_{%i}$' % (j, i) for i in range(16)]
                             for j in [4, 5]], [])

    if attack == "present-80-lowdata":
        # 9-round attack on Present-80 with minimal data
        p.computation_model, p.covered_rounds = CLASSICAL_GAD, [0, 1, 2, 9]
        p.minimize_data = True
        p.cut_forward, p.cut_backward = [6, 7, 8, 9], [1, 2, 3, 4]
        p.data_limit = 12  # putting 8 or 12, we find different attacks
        p.simplified = True
        cipher = make_present_constraints(nrounds=9, flag="present-80")

    if attack == "present-80-lowdata-quantum":
        # will find something, but the margin is too small w.r.t. the constant
        # factors of the quantum time complexity formula
        p.computation_model, p.covered_rounds = QUANTUM_GAD, [0, 1, 2, 9]
        p.minimize_data = True
        p.cut_forward, p.cut_backward = [6, 7, 8, 9], [1, 2, 3, 4]
        p.data_limit = 12
        p.simplified = True
        cipher = make_present_constraints(nrounds=9, flag="present-80")
        #round_bits = True

    if attack == "pipo-128":
        # structure is same as fly (but in fly we have one round for free,
        # because there is no final key addition)
        p.computation_model, p.covered_rounds = CLASSICAL_GAD, [10]
        cipher = make_pipo_constraints(nrounds=10, flag="pipo-128")
        p.cut_backward, p.cut_forward = [2, 3, 4, 5, 6], [7, 8, 9, 10, 0]
        p.covered_rounds += [0, 1, 2, 9, 10]
        p.not_fwd_key = ["$k^{%i}_{%i}$" % (0, i)
                         for i in range(64)]  # bw or shared
        p.simplified = True

    if attack == "pipo-128-quantum":
        p.computation_model, p.covered_rounds = QUANTUM_GAD, [8]
        cipher = make_pipo_constraints(nrounds=8, flag="pipo-128")
        p.cut_backward, p.cut_forward = [2, 3, 4, 5], [7, 8, 0]
        p.covered_rounds += [0, 1, 7, 8]
        p.not_fwd_key = ["$k^{%i}_{%i}$" % (0, i)
                         for i in range(64)]  # bw or shared
        p.simplified = True

    if attack == "pipo-256":
        p.computation_model, p.covered_rounds = CLASSICAL_GAD, [18]
        cipher = make_pipo_constraints(nrounds=18, flag="pipo-256")
        p.covered_rounds += [0, 1, 2, 15, 16, 17]
        p.cut_forward, p.cut_backward = [0, 1, 2, 15, 16, 17,
                                         18], [4, 5, 6, 7, 8, 9, 10, 11, 12]
        p.not_fwd_key = sum([["$k^{%i}_{%i}$" % (j, i) for i in range(64)]
                             for j in [0, 1, 2]], [])
        p.simplified = True

    if attack == "pipo-256-quantum":
        p.computation_model, p.covered_rounds = QUANTUM_GAD, [16]
        cipher = make_pipo_constraints(nrounds=16, flag="pipo-256")
        p.covered_rounds += [0, 1, 2, 15]
        p.cut_forward, p.cut_backward = [0, 1, 14, 15, 16
                                         ], [3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        p.not_fwd_key = sum([["$k^{%i}_{%i}$" % (j, i) for i in range(64)]
                             for j in [0]], [])
        p.simplified = True

    if attack == "haraka-256":
        # recover the best previous attack on Haraka-256
        # (several alternative solutions exist)
        p.computation_model, p.covered_rounds = CLASSICAL_PREIMAGE, []
        cipher = make_haraka256_constraints(nrounds=9)
        cipher.preimage_attack = True
        p.covered_rounds = [4]
        p.cut_forward = [0, 1, 8, 9]

    if attack == "haraka-512-11":
        # improved 11-round attack on Haraka-512 (better time complexity)
        p.computation_model, p.covered_rounds = CLASSICAL_PREIMAGE, []
        cipher = make_haraka512_constraints(nrounds=11)
        cipher.preimage_attack = True
        p.backward_zero = ([
            '$c^{%i}_{%i}$' % (0, i) for i in range(16) if i >= 4
        ])
        p.backward_zero += ([
            '$c^{%i}_{%i}$' % (1, i) for i in range(16) if i >= 4
        ])
        p.backward_hint = ([
            '$c^{%i}_{%i}$' % (0, i) for i in range(16) if i < 4
        ])
        p.backward_hint += ([
            '$c^{%i}_{%i}$' % (1, i) for i in range(16) if i < 4
        ])
        p.covered_rounds = [2, 3, 4, 5, 6]
        p.cut_forward = [0, 1, 2, 10, 11]

    if attack == "haraka-512-13":
        p.computation_model, p.covered_rounds = CLASSICAL_PREIMAGE, []
        cipher = make_haraka512_constraints(nrounds=13)
        cipher.preimage_attack = True
        p.backward_zero = ([
            '$c^{%i}_{%i}$' % (0, i) for i in range(16) if i >= 4
        ])
        p.backward_zero += ([
            '$c^{%i}_{%i}$' % (1, i) for i in range(16) if i >= 4
        ])
        p.backward_zero += ([
            '$c^{%i}_{%i}$' % (10, i) for i in range(16) if i >= 4
        ])
        p.backward_zero += ([
            '$c^{%i}_{%i}$' % (11, i) for i in range(16) if i >= 4
        ])
        p.backward_hint = ([
            '$c^{%i}_{%i}$' % (0, i) for i in range(16) if i < 4
        ])
        p.backward_hint += ([
            '$c^{%i}_{%i}$' % (1, i) for i in range(16) if i < 4
        ])
        p.backward_hint += ([
            '$c^{%i}_{%i}$' % (10, i) for i in range(16) if i < 4
        ])
        p.backward_hint += ([
            '$c^{%i}_{%i}$' % (11, i) for i in range(16) if i < 4
        ])
        p.covered_rounds = [4]
        p.cut_forward = [0, 1, 2, 8, 9, 10, 11, 12, 13]
        p.cut_backward = [2, 8]

    if attack == "saturnin-simon":
        p.computation_model, p.covered_rounds = QUANTUM_SIMON, [6]
        cipher = make_saturnin_constraints(nrounds=6, no_mc_at_last_round=True)
        p.minimize_data = True

    if attack == "gift-64-simon":
        p.computation_model, p.covered_rounds = QUANTUM_SIMON, [15]
        cipher = make_gift64_constraints(nrounds=15)
        p.minimize_data = True
        p.cut_forward = [0, 1, 2, 3, 12, 13, 14, 15]
        p.covered_rounds += [0, 1, 2, 3, 12, 13, 14, 15]
        p.round_bits = False

    if p.file_name is None:
        # set default file name for result
        p.file_name = attack

    return p, cipher


def _find_mitm_attack(constraints, p):
    return find_mitm_attack(constraints,
                            time_target=p.time_target,
                            computation_model=p.computation_model,
                            optimize_with_mem=p.optimize_with_mem,
                            minimize_data=p.minimize_data,
                            data_limit=p.data_limit,
                            memory_limit=p.memory_limit,
                            cut_forward=p.cut_forward,
                            cut_backward=p.cut_backward,
                            backward_zero=p.backward_zero,
                            forward_zero=p.forward_zero,
                            forward_hint=p.forward_hint,
                            backward_hint=p.backward_hint,
                            shared_key=p.shared_key,
                            not_fwd_key=p.not_fwd_key,
                            not_bwd_key=p.not_bwd_key,
                            covered_rounds=p.covered_rounds,
                            round_bits=p.round_bits,
                            backend="gurobi")


if __name__ == "__main__":
    import sys
    argc = len(sys.argv)
    if argc < 2:
        print(_HELP)
        sys.exit(0)

    if argc == 3:
        override_simplified = (sys.argv[2] == "full")
    else:
        override_simplified = False

    attack = sys.argv[1]

    #=========== attack time complexities that should be found by the model
    # (note that the time here is counted in nibbles)
    attacks = {
        "aes-preimage": 7.5,
        "saturnin-preimage-classical": 12,
        "saturnin-preimage-quantum": 7,
        "aes-1k": 14,
        "aes-1k-smalldata": 15,
        "aes-2k": 31,
        "future": 126,
        "saturnin-classical": 15.5,
        "saturnin-quantum": 7.75,
        "gift-64-gad": 112,
        "present-80-lowdata": 77,
        "present-80-lowdata-quantum": 38.5,
        "pipo-128": 125,
        "pipo-256": 252,
        "haraka-256": 28,
        "haraka-512-11": 28,
        "haraka-512-13": 30,
        "saturnin-simon": 6.5,
        "gift-64-simon": 54,
        "pipo-128-quantum": 60,
        "pipo-256-quantum": 300,
    }

    if attack == "test":
        # Testing all the attacks. This will take some time.
        for a in attacks:
            p, cipher = _parameters(a)
            p.file_name = None  # no file output
            p.time_target = attacks[a]
            p.optimize_with_mem = False
            p.minimize_data = False
            constraints = cipher.convert()
            (success, cell_var_colored, key_bits_colored, key_cells_colored,
             results_lines) = _find_mitm_attack(constraints, p)
            assert success
            print("===================== Test: ", a, " successful!\n\n")
    elif attack not in attacks:
        print(" Not supported: ", attack)
    else:
        p, cipher = _parameters(attack)
        constraints = cipher.convert()
        (success, cell_var_colored, key_bits_colored, key_cells_colored,
         results_lines) = _find_mitm_attack(constraints, p)
        for l in results_lines:
            print(l)
        # format this as a lateX comment, to include in generated file
        comment = "\n".join(["%" + s for s in results_lines]) + "\n\n"

        HARAKA_PICTURE_IMPORTED = False
        try:
            from haraka_picture_util import convert_to_haraka512_pic
            HARAKA_PICTURE_IMPORTED = True
        except ImportError:
            pass
        HARAKA_PICTURE_IMPORTED = False

        if not success:
            print("FAILED")
        else:
            if attack == "haraka-512-13" and HARAKA_PICTURE_IMPORTED:
                tikz_code = convert_to_haraka512_pic(constraints,
                                                     cell_var_colored)
                write_and_compile(tikz_code, "haraka-512-13-aes")

            tikz_code = cipher.convert_to_tikz(
                cell_var_colored,
                key_bits_colored,
                key_cells_colored,
                all_edges=True,
                comment=comment,
                key_prev=p.key_prev,
                future_special=(attack == "future"
                                and not override_simplified),
                simplified=p.simplified if not override_simplified else False)
            if p.file_name is not None:
                write_and_compile(tikz_code, p.file_name)
