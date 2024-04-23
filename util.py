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
This file provides an implementation of a cell-based representation of 
a block cipher, which abstracts the design as a directed graph. This representation
is contained in a "Constraints" object. It includes a key-schedule.
Relations between groups of key bits are formalized by cells, like relations between
state bits. This object is adapted from the code of [SS22].

The "Cipher" object allows to define a block cipher simply by its
sequence of operations, and convert it automatically into a "Constraints" object
(without having to define the cells and edges by hand). It supports S-Boxes,
bit permutations and key schedule operations.

--Reference:
[SS22] Schrottenloher, Stevens, 'Simplified MITM Modeling for Permutations: 
New (Quantum) Attacks.', CRYPTO 2022, code available at:
https://github.com/AndreSchrottenloher/mitm-milp
"""
from modeling import FORWARD, BACKWARD, MERGED, SHARED
import platform
import os

# Additional code to compile the generated tikz code of attacks,
# and to open the corresponding file. Only tested on a Fedora platform.

RESULTS = "results"
TEX_STYLE = "simplifiedpresent.sty"


def write_and_compile(code, file_name, open_after=True):
    tex_file_name = file_name.strip(".") + ".tex"
    pdf_file_name = file_name.strip(".") + ".pdf"
    if not os.path.isdir(RESULTS):
        print("Creating directory", RESULTS)
        os.mkdir(RESULTS)
    full_tex_file_name = os.path.join(RESULTS, tex_file_name)
    full_pdf_file_name = os.path.join(RESULTS, pdf_file_name)
    if os.path.isfile(full_tex_file_name):
        input_res = input("Overwrite file? (y/n)")
        if input_res.lower() != "y":
            print("Ending here")
            return
    with open(full_tex_file_name, 'w') as f:
        f.write(code)
    # check that style file is here
    if not os.path.isfile(TEX_STYLE):
        raise Exception("Style file does not exist")
    # now compile with pdflatex, within this directory
    print("Compiling...")
    exit_code = os.system("cd " + RESULTS + "; pdflatex " + tex_file_name)
    #exit_code = os.system()
    print("Compilation ran with exit code:", exit_code)
    # open file
    if open_after:
        _platform_system = platform.system()
        if _platform_system == "Linux":
            exit_code = os.system("xdg-open " + full_pdf_file_name)
        elif _platform_system == "Windows":
            exit_code = os.system("start " + full_pdf_file_name)
        elif _platform_system == "Darwin":
            exit_code = os.system("open " + full_pdf_file_name)
        else:
            raise Exception("Unrecognized platform")
        print("File opening ran with exit code:", exit_code)


class Constraints:
    """
    Directed graph of cells which abstracts a block cipher (or compression
    function) design. This code is adapted from the class PresentConstraints in [SS22].
    
    Contrary to the previous class, it does not support the merging of cells,
    because of the presence of key additions.
    
    Cells are identified by names, but also by their round and their position
    within this round. They are named x^i_j where i is the
    round number and j the position. They have weights, which correspond to the
    number of bits that we need to know to deduce all other inputs and outputs of
    the cell.
    
    Edges between cells represent linear constraints. Each edge has a weight and
    an (optional) key bit which is XORed on it. 
    A key schedule is also supported. All key bits (including those deduced
    from the initial key by S-Box operations) must be passed into the
    "key_array" argument.
    """

    def __init__(self, nrounds, key_array=[], key_bit_width=None):
        """
        Initializes the object.
        
        nrounds --  number of rounds
        key_array -- list of strings, each string is the name of a bit in the
            key expansion of the cipher.
        """
        self.key_bit_width = key_bit_width
        self.key_array = key_array
        self.key_length = len(self.key_array)  # number of key bits
        self.merged_cells = {}
        self.cell_names_by_round = {}
        self.cell_round_pos_to_name = {}
        self.cell_name_to_data = {}  # rd, pos, width
        self.super_sboxes = []  # cells which are super s-boxes
        self.branching_cells = set()  # cells which are branching

        self.cell_name_to_fwd_edges_width = {}
        self.cell_name_to_bwd_edges_width = {}

        self.fwd_graph = {
        }  # for each cells, connected cells at the next round
        self.bwd_graph = {
        }  # for each cells, connected cells at the prev round

        self.fwd_edges = {}
        # for each cell c, edges between c and cells at the next round (going forward)
        # 4-tuple c1, c2, w, k
        self.bwd_edges = {}
        # for each cell c, edges between c and cells at the previous round (going backward)

        self._edge_numbering_helper = {}
        self._cell_pos_helper = {}
        # each edge has also a name
        self.edge_names_by_round = {}
        self.edge_name_to_data = {}  # c1, c2 (names), width, key_bit

        self.cell_pos_storage = {}  # remember the position of cells
        # (before simplifying and merging)
        self.global_fixed = []
        for r in range(nrounds):
            self.cell_names_by_round[r] = []
            self.edge_names_by_round[r] = []
            self._cell_pos_helper[r] = 0
        self.nrounds = nrounds

        self._key_cell_numbering = 0
        self.key_cells = {}
        self.state_size = None

    def add_key_cell(self, key_bits_input, key_bits_output, name=None):
        if name is None:
            raise ValueError("Expected name to be given")
        if len(key_bits_input) != len(key_bits_output):
            raise ValueError("Invalid key cell definition")
        w = len(key_bits_output)
        if len(set(list(key_bits_input) + list(key_bits_output))) < 2 * w:
            raise ValueError("All key bits must be distinct")

        self.key_cells[name] = (list(key_bits_input), list(key_bits_output), w)
        self._key_cell_numbering += 1

    def add_cell(self, r, w, name=None, super_sbox=False):
        """
        Adds a cell of width w at round r. We remember which cells are "super s-boxes",
        which have the property of matching through the cell.
        """
        pos = self._cell_pos_helper[r]
        if name is None:
            name = "$c^{%i}_{%i}$" % (r, pos)
        if super_sbox:
            self.super_sboxes.append(name)
        self.cell_names_by_round[r].append(name)
        self.cell_name_to_data[name] = (r, pos, w)
        self.cell_round_pos_to_name[(r, pos)] = name
        self.cell_pos_storage[name] = pos
        self._edge_numbering_helper[name] = {}
        self._cell_pos_helper[r] += 1
        self.fwd_edges[name] = []
        self.bwd_edges[name] = []
        self.fwd_graph[name] = {}
        self.bwd_graph[name] = {}

    def get_state_size(self):
        """
        Finds the state size of this design (MINIMAL sum of widths of individual
        cells over all the rounds).
        """
        if self.state_size is None:
            # state size for a round: sum of all cell widths for this round
            # the largest such sum defines the state size of the design
            res = None
            for r in self.cell_names_by_round:
                tmp = sum([
                    self.cell_name_to_data[c][2]
                    for c in self.cell_names_by_round[r]
                ])
                res = min(tmp, res) if res is not None else tmp
            self.state_size = res
        return self.state_size

    def possible_middle_rounds(self):
        """
        Returns a list of rounds which have a size equal to the maximal state size.
        (Not all the rounds have the same size, only the "middle rounds" are complete,
        for ex. if the input-output conditions are enforced only on part of the state).
        """
        s = self.get_state_size()
        res = []
        for r in self.cell_names_by_round:
            if sum([
                    self.cell_name_to_data[c][2]
                    for c in self.cell_names_by_round[r]
            ]) == s:
                res.append(r)
        return res

    def fwd_edges_width(self, c):
        return sum([self.fwd_graph[c][cp] for cp in self.fwd_graph[c]])

    def bwd_edges_width(self, c):
        return sum([self.bwd_graph[c][cp] for cp in self.bwd_graph[c]])

    def get_fwd_edges(self, c):
        """
        Returns names of fwd edges (edges between c and cells at the next round, going forward)
        """
        return self.fwd_edges[c]

    def get_bwd_edges(self, c):
        """
        Returns names of bwd edges
        """
        return self.bwd_edges[c]

    def add_edge(self, c1, c2, w, key_bit=None):
        """
        Adds an edge between two cells and returns its new name. An edge contains a key
        bit or None. All edges with key bits should have the same weight.
        """
        key_bit_name = key_bit
        if key_bit is not None:
            if type(key_bit) == int:
                key_bit_name = self.key_array[key_bit]
        if c1 not in self.cell_name_to_data or c2 not in self.cell_name_to_data:
            raise ValueError("Unexisting cell")
        idx = 0
        if c2 in self._edge_numbering_helper[c1]:
            self._edge_numbering_helper[c1][c2] += 1
            idx = self._edge_numbering_helper[c1][c2]
        else:
            self._edge_numbering_helper[c1][c2] = 0
        name = str(c1) + ":" + str(c2) + ":" + str(idx)

        self.fwd_edges[c1].append((c1, c2, w, key_bit))
        self.bwd_edges[c2].append((c1, c2, w, key_bit))
        cur_round = self.cell_name_to_data[c1][0]
        self.edge_names_by_round[cur_round].append(name)

        # multiple edges are possible
        self.edge_name_to_data[name] = c1, c2, w, key_bit_name
        if c2 not in self.fwd_graph[c1]:
            self.fwd_graph[c1][c2] = 0
        if c1 not in self.bwd_graph[c2]:
            self.bwd_graph[c2][c1] = 0
        self.fwd_graph[c1][c2] += w
        self.bwd_graph[c2][c1] += w

        # update branching cells if necessary
        if self.fwd_edges_width(c1) > self.get_cell_width(c1):
            self.branching_cells.add(c1)

        return name

    def add_edge_2(self, r, i1, i2, w, key_bit=None):
        """
        Adds an edge between two cells, identified by their index.
        """
        c1 = self.get_cell_name(r, i1)
        c2 = self.get_cell_name((r + 1) % self.nrounds, i2)
        return self.add_edge(c1, c2, w, key_bit=key_bit)

    def get_edge_key_bit(self, name):
        return self.edge_name_to_data[name][3]

    def get_cell_name(self, r, pos):
        """
        Returns the name of a cell at a given position.
        """
        return self.cell_round_pos_to_name[(r, pos)]

    def set_global(self, name):
        """
        Sets an edge, given by its name, as a global constraint.
        """
        self.global_fixed.append(name)

    def get_cells_by_round(self, r):
        return self.cell_names_by_round[r]

    def get_cell_width(self, n):
        return self.cell_name_to_data[n][2]

    def get_key_cell_width(self, n):
        return self.key_cells[n][2]

    def get_cell_pos(self, n):
        return self.cell_pos_storage[n]

    def get_edges_by_round(self, r):
        return self.edge_names_by_round[r]

    def get_super_sboxes(self):
        return self.super_sboxes

    def get_branching_cells(self):
        return self.branching_cells

    def edge_data(self, n):
        return self.edge_name_to_data[n]

    def get_data(self):
        """
        Returns the data that we need for the generic solver:
        - a dictionary of cell names : cell data (round, position, weight)
        - a dictionary of rounds : cells for this round
        - a dictionary of edge names : edge data (cell at previous round, cell at next round, weight, key bit)
        - a dictionary of rounds : edges for this round
        - a list of edge names which are globally fixed
        """
        #cells, cells_by_round, linear_constraints, linear_by_round, global_fixed
        return ({
            c: self.cell_name_to_data[c][2]
            for c in self.cell_name_to_data
        }, self.cell_names_by_round, self.edge_name_to_data,
                self.edge_names_by_round, self.global_fixed)

    def __repr__(self):
        return str(self)

    def __str__(self):
        res = "Present-like constraint set: "
        res += str(self.get_data())
        return res


#====================================


def rotate_right(l, v):
    """
    Rotates list l by v positions to the right.
    """
    return l[-v:] + l[:-v]


def rotate_left(l, v):
    """
    Rotates list l by v positions to the left.
    """
    return l[v:] + l[:v]


# Header of generated LateX file
STANDALONE_HEADER = """
\\documentclass{standalone}
\\usepackage{amsmath}
\\usepackage{amsfonts}
\\usepackage{amssymb}
\\usepackage{tikz}
\\usepackage{xcolor}
%\\usepackage{../style/aes}
\\usepackage{../simplifiedpresent}
\\colorlet{darkgreen}{green!50!black} \colorlet{darkblue}{blue!50!black}
\\colorlet{darkred}{red!50!black} \colorlet{darkorange}{orange!70!black}
"""

# specify parameters to draw the figure.
# rdheight is the height of a round
# cellstart is the position (in proportion of round height), starting from 0,
# at which the cell starts. If it's 0.1, then a 0.1 portion of the round height
# is dedicated to the small vertical edge
# outedgestart is the position, starting from 0, at which the cell ends and the
# output edge starts
# swapstart is the position, starting from 0, at which the vertical portion of the
# output edge stops (and the "swap" part of the edge starts)

# Parameters for a small state size (e.g. 16, like AES)
SMALL_PARAMETERS = """
\\begin{figpresent}
\\def\\rdheight{20}
\\def\\cellstart{0.2}
\\def\\outedgestart{0.5}
\\def\\swapstart{0.7}
"""

# Parameters for a large state size and a complicated bit-swapping (e.g, 64)
LARGE_PARAMETERS = """
\\begin{figpresent}[yscale=2]
\\def\\rdheight{20} % height of a round
\\def\\cellstart{0.1} % position at which the cell starts (the rest 
\\def\\outedgestart{0.3} % position at which we start the edge
\\def\\swapstart{0.4} % position at which
"""

# parameters for a 64-bit state, but we won't represent the linear layers,
# and we won't name the individual key bits (just use color boxes)
SIMPLIFIED_PARAMETERS = """
\\begin{figpresent}[xscale=0.4, yscale=0.7]
\\renewcommand{\\figedge}[3][]{}
\\renewcommand{\\figcell}[4][]{\\figemptycell[#1]{#2}{#3}{#4}}
\\renewcommand{\\addkeyabove}[3][]{\\keyaddabove[#1]{#2}{#3}}
\\renewcommand{\\addkeybelow}[3][]{\\keyaddbelow[#1]{#2}{#3}}
\\def\\rdheight{20} % height of a round
\\def\\cellstart{0.2} % position at which the cell starts (the rest 
\\def\\outedgestart{0.7} % position at which we start the edge
\\def\\swapstart{0.8} % position at which we start the swap
"""

FUTURE_SPECIAL_PARAMETERS = """
\\begin{figpresent}[yscale=1., xscale=0.5]
\\def\\rdheight{20}
\\def\\bitwidth{4}
\\def\\cellstart{0.15}
\\def\\outedgestart{0.45}
\\def\\swapstart{0.55}
\\def\\nextcellsep{0.15}
\\def\\edgeshift{6}
"""


class Cipher:
    """
    Defines an SPN cipher by a sequence of operations; can be converted into a 
    Constraints object.
    """

    def __init__(self,
                 state_size,
                 key_register_sizes,
                 key_bit_width=None,
                 preimage_attack=False):
        """
        Key registers: will initialize multiple key registers
        by naming the bits k_i^j. They correspond to the internal
        representation of the key, e.g. a single master key (if there is a
        single register) or several round key registers. They can be updated
        in place.
        """
        self.key_bit_width = key_bit_width
        # if this is a preimage attack or not
        self.preimage_attack = preimage_attack
        self.partial_wrapping = []
        self.single_key = (len(key_register_sizes) == 1)
        self.has_key = False
        if self.single_key:
            self.key_registers = [[
                "$k_{%i}$" % (i) for i in range(key_register_sizes[0])
            ]]
        else:
            self.key_registers = [[
                "$k^{%i}_{%i}$" % (j, i) for i in range(key_register_sizes[j])
            ] for j in range(len(key_register_sizes))]
        self._all_key_bits = sum(self.key_registers, [])

        self._key_numbering_helper = [s for s in key_register_sizes]
        self._state_size = state_size
        self._key_cells = []
        self._cells = {}  # cells, by round. Each cell has a list of input and
        # output nibbles, this will allow to determine their edges. Keys are added
        # on input nibbles
        # for each round: list of (position, input nibbles, output nibbles, is super sbox)
        # A nibble is represented as (i, r, k) where k is a key bit in the round key at round r (or None)
        # r is the current round number (starts at 0, increased when applying Sboxes) and i is the position

        # we do not support partial rounds. So, when everyone has had an s-box
        # applied, we reset all of this to False and increase the current round
        self._is_sbox_applied = [False for i in range(state_size)]
        self._states = []
        self._states.append([(i, None) for i in range(state_size)])
        self._permute_flag = False

    def _new_key_nibbles(self, register, nb):
        """
        Creates new key nibbles.
        """
        res = []
        for i in range(nb):
            if self.single_key:
                res.append("$k_{%i}$" % self._key_numbering_helper[register])
            else:
                res.append("$k^{%i}_{%i}$" %
                           (register, self._key_numbering_helper[register]))
            self._key_numbering_helper[register] += 1
        return res

    def key_permute(self, l, register=0):
        """
        Key schedule operation: permutes the bits in a key register. Default
        is register 0.
        """
        if len(l) != len(self.key_registers[register]):
            raise ValueError("Wrong list size")
        copy = [t for t in self.key_registers[register]]
        for i in range(len(self.key_registers[register])):
            self.key_registers[register][l[i]] = copy[i]

    def key_permute_inv(self, l, register=0):
        if len(l) != len(self.key_registers[register]):
            raise ValueError("Wrong list size")
        copy = [t for t in self.key_registers[register]]
        for i in range(len(self.key_registers[register])):
            self.key_registers[register][i] = copy[l[i]]

    def key_rotate_left(self, p, register=0):
        self.key_registers[register] = rotate_left(
            self.key_registers[register], p)

    def key_rotate_right(self, p, register=0):
        self.key_registers[register] = rotate_right(
            self.key_registers[register], p)

    def key_sbox(self, positions, register=0):
        """
        Key schedule operation: applies an Sbox on several bit positions 
        in one of the key registers.
        The number of output bits is determined by the number of input bits (it has
        to be the same).
        The key register is updated in place.
        """
        nibbles = [self.key_registers[register][i] for i in positions]
        new_nibbles = self._new_key_nibbles(register, len(nibbles))
        new_name = "$y_{%i}$" % (len(self._key_cells))
        # also updates key numbering helper
        self._key_cells.append((new_name, nibbles, new_nibbles))
        self._all_key_bits += new_nibbles
        for i in range(len(nibbles)):
            self.key_registers[register][positions[i]] = new_nibbles[i]

    def add_key_bit(self, key_pos, state_pos, register=0):
        (i, k) = self._states[-1][state_pos]
        if k is not None:
            raise ValueError("Unsupported")
        self.has_key = True
        self._states[-1][state_pos] = (i,
                                       self.key_registers[register][key_pos])

    def permute(self, l):
        if len(l) != self._state_size:
            raise ValueError("Wrong list size")
        for i in range(self._state_size):
            if i not in l:
                raise ValueError("List must define a permutation")
        copy = [t for t in self._states[-1]]
        for i in range(self._state_size):
            self._states[-1][l[i]] = copy[i]
        self._permute_flag = True

    def permute_inv(self, l):
        if len(l) != self._state_size:
            raise ValueError("Wrong list size")
        for i in range(self._state_size):
            if i not in l:
                raise ValueError("List must define a permutation")
        copy = [t for t in self._states[-1]]
        for i in range(self._state_size):
            self._states[-1][i] = copy[l[i]]
        self._permute_flag = True

    def sbox(self, position, l, super_sbox=False, display_name=True):
        self._permute_flag = False
        for j in range(l):
            if self._is_sbox_applied[position + j]:
                raise ValueError("Unsupported: cannot apply partial rounds")
            self._is_sbox_applied[position + j] = True
        r = len(self._states) - 1
        if r not in self._cells:
            self._cells[r] = []
        self._cells[r].append((position, l, super_sbox, display_name))

        # increase the round number, if we have applied an s-box to everyone
        full_round = all(self._is_sbox_applied)
        if full_round:
            self._is_sbox_applied = [False for i in range(self._state_size)]
            # the next round state
            self._states.append([(i, None) for i in range(self._state_size)])

    def set_partial_wrapping(self, l):
        self.partial_wrapping = l

    def convert(self):
        # converts this to Constraint object
        # will first apply the cipher as last round (unless we're in the preimage
        # case, in which we wrap with individual S-Boxes on the nibbles)
        if self.preimage_attack:
            for i in range(self._state_size):
                self.sbox(i, 1, super_sbox=False, display_name=False)
        else:
            self.sbox(0, self._state_size, super_sbox=False, display_name=True)
        # no partial wrapping = full wrapping
        if self.partial_wrapping == []:
            self.partial_wrapping = [i for i in range(self._state_size)]
        self._states = self._states[:-1]  # don't need the last one
        nrounds = len(self._states)

        # now we have all cells and key cells.
        # We will first create an object with the right number of rounds & keys
        cons = Constraints(nrounds=nrounds,
                           key_array=self._all_key_bits,
                           key_bit_width=self.key_bit_width)

        # add the key cells
        for (name, nibbles, new_nibbles) in self._key_cells:
            cons.add_key_cell(nibbles, new_nibbles, name=name)

        self.sorted_cells = {}
        for r in range(nrounds):
            new_cells = self._cells[r]
            new_cells = sorted(self._cells[r], key=lambda x: x[0])
            self.sorted_cells[r] = new_cells

        self.nibble_real_positions = {}
        for r in range(nrounds):
            self.nibble_real_positions[r] = {}
            for j in range(self._state_size):
                (i, k) = self._states[r][j]
                self.nibble_real_positions[r][i] = j

        self.nibble_to_cell_position = {}
        for r in range(nrounds):
            self.nibble_to_cell_position[r] = {}
            for jj in range(len(self.sorted_cells[r])):
                (position, l, super_sbox, _) = self.sorted_cells[r][jj]
                for j in range(l):
                    self.nibble_to_cell_position[r][position + j] = jj

        for r in range(nrounds):
            for (position, l, super_sbox, _) in self.sorted_cells[r]:
                cons.add_cell(r, w=l, super_sbox=super_sbox)

        for r in range(nrounds):
            nibble_position_at_this_round = 0
            next_round = r + 1 if r < nrounds - 1 else 0
            for jj in range(len(self.sorted_cells[r])):
                (position, l, super_sbox, _) = self.sorted_cells[r][jj]

                # cell at round r is applied to nibbles at positions:
                # position, position+1, ... position + l - 1
                # and outputs the nibbles at the same positions
                # relation is between the current cell and the cell where the
                # 'nibble real position' of next round is found
                for j in range(l):
                    _nibble_real_position = self.nibble_real_positions[
                        next_round][position + j]
                    if (r == nrounds - 1 and (nibble_position_at_this_round
                                              not in self.partial_wrapping)):
                        pass
                    else:
                        next_nibble = self._states[next_round][
                            _nibble_real_position]
                        i, k = next_nibble
                        next_cell = self.nibble_to_cell_position[next_round][
                            _nibble_real_position]

                        cons.add_edge(cons.get_cell_name(r, jj),
                                      cons.get_cell_name(
                                          next_round, next_cell),
                                      w=1,
                                      key_bit=k)
                nibble_position_at_this_round += 1

        return cons

    def convert_to_tikz(self,
                        cell_var_colored,
                        key_bits_colored,
                        key_cells_colored,
                        all_edges=False,
                        comment="",
                        key_prev=[],
                        simplified=False,
                        future_special=False):
        """
        Returns a tikz code (as str) to represent this cipher.
        Parameters cell_var_colored, key_bits_colored, key_cells_colored are the
        results of the solved model.
        
        key_prev: associate round key addition that is between layer i-1 and layer i
        to layer i-1 (default is layer i). Will change the display of key bits.
        """
        res = "%=============================\n\n"
        res += STANDALONE_HEADER
        res += comment
        res += """\\begin{document}\n"""
        if simplified:
            res += SIMPLIFIED_PARAMETERS
        elif future_special:
            res += FUTURE_SPECIAL_PARAMETERS
        elif self._state_size < 32:
            res += SMALL_PARAMETERS
        else:
            res += LARGE_PARAMETERS
        nrounds = max([r for r in self._cells]) + 1
        for r in range(nrounds):
            _tmp = 0  # numbering counter for input nibbles
            _ttmp = 0  # other numbering counter
            _tttmp = 0
            _cell_position_ctr = 0
            next_round = r + 1 if r < nrounds - 1 else 0

            for cell_pos in range(len(self.sorted_cells[r])):
                _c = self.sorted_cells[r][cell_pos]
                # check that cell inputs form a list of consecutive positions
                pos, l, super_sbox, display_name = _c[0], _c[1], _c[2], _c[3]
                input_nibbles = [self._states[r][pos + j] for j in range(l)]
                output_nibbles = [
                    self._states[next_round][
                        self.nibble_real_positions[next_round][pos + j]]
                    for j in range(l)
                ]

                #cell_pos = i
                cell_name = "$c^{%i}_{%i}$" % (r, cell_pos)
                #print(cell_name, [n[0] for n in input_nibbles])

                status = ""
                if cell_var_colored["forward"][cell_name] > 0.5:
                    status = "fwd"
                elif cell_var_colored["backward"][cell_name] > 0.5:
                    status = "bwd"
                elif cell_var_colored["merged"][cell_name] > 0.5:
                    status = "mgd"
                res += ("""\\figcell[%s]{%i}{%i}{%s}""" %
                        (status, _cell_position_ctr, len(input_nibbles),
                         cell_name if display_name else ""))
                res += """\n"""
                _cell_position_ctr += len(input_nibbles)

                for j in range(len(output_nibbles)):
                    n = output_nibbles[j]
                    i, key_bit = n[0], n[1]
                    rp = r + 1 if r < nrounds - 1 else 0
                    next_pos = self.nibble_real_positions[rp][i]
                    next_cell_pos = self.nibble_to_cell_position[rp][
                        self.nibble_real_positions[rp][i]]
                    status = ""
                    # input cell is "cell_name"
                    # output cell is
                    input_cell = cell_name
                    output_cell = "$c^{%i}_{%i}$" % (rp, next_cell_pos)
                    # each pair (input_cell, output_cell) will have the same position

                    if cell_var_colored[FORWARD][input_cell]:
                        if cell_var_colored[FORWARD][output_cell]:
                            status = "fwd"
                        elif cell_var_colored[BACKWARD][output_cell]:
                            status = "match"
                        elif cell_var_colored[MERGED][output_cell]:
                            status = "fwd"
                    elif cell_var_colored[BACKWARD][input_cell]:
                        if cell_var_colored[FORWARD][output_cell]:
                            status = "guess"
                        elif cell_var_colored[BACKWARD][output_cell]:
                            status = "bwd"
                        elif cell_var_colored[MERGED][output_cell]:
                            status = "bwd"
                    elif cell_var_colored[MERGED][input_cell]:
                        if cell_var_colored[FORWARD][output_cell]:
                            status = "fwd"
                        elif cell_var_colored[BACKWARD][output_cell]:
                            status = "bwd"
                    if status != "" or all_edges:
                        if ((future_special and _tmp % 4 == 0)
                                or (not future_special)):
                            _inp = (_tmp // 4) * 4 if future_special else _tmp
                            if (r == nrounds - 1
                                    and _inp not in self.partial_wrapping):
                                pass
                            else:
                                res += """\\figedge[%s]{%i}{%i}""" % (
                                    status, _inp, (next_pos // 4) *
                                    4 if future_special else next_pos)
                    _tmp += 1

                res += """\n"""

                # add key bits AFTER edges
                # key bits (input)
                if next_round in key_prev:
                    # associate round key addition between r and r+1 to layer r:
                    # put them below cells of round r
                    for n in output_nibbles:
                        i, key_bit = n[0], n[1]
                        if key_bit is not None:
                            status = ""
                            if key_bits_colored[SHARED][key_bit] > 0.5:
                                status = "shared"
                            elif key_bits_colored[FORWARD][key_bit] > 0.5:
                                status = "fwd"
                            elif key_bits_colored[BACKWARD][key_bit] > 0.5:
                                status = "bwd"
                            _command = "keyaddbelow" if future_special else "addkeybelow"
                            res += ("""\\%s[%s]{%i}{%s}""" %
                                    (_command, status, _ttmp, key_bit))
                        _ttmp += 1
                if r not in key_prev:
                    for n in input_nibbles:
                        i, key_bit = n[0], n[1]
                        if key_bit is not None:
                            status = ""
                            if key_bits_colored[SHARED][key_bit] > 0.5:
                                status = "shared"
                            elif key_bits_colored[FORWARD][key_bit] > 0.5:
                                status = "fwd"
                            elif key_bits_colored[BACKWARD][key_bit] > 0.5:
                                status = "bwd"
                            _command = "keyaddabove" if future_special else "addkeyabove"
                            res += ("""\\%s[%s]{%i}{%s}""" %
                                    (_command, status, _tttmp, key_bit))
                        _tttmp += 1
                    #print(r, cell_pos, input_nibbles)
                    #raise Exception("end")
                res += """\n"""

            res += ("""\\roundlabel{$R_{%i}$}\n""" % r)
            res += """\\newround\n%\n"""

        if not simplified:
            # in "simplified" mode, no key cells (we would read no information on them anyway)

            #=== key cells
            # display the key cells in several layers (which are actually virtual "rounds")
            # nbr of rounds depends on self._state_size and key cell size
            _cell_position_ctr = 0
            if self._key_cells:
                res += """\\newround\n%\n"""
            for (cell_name, key_input_nibbles,
                 key_output_nibbles) in self._key_cells:
                #cons.add_key_cell(nibbles, new_nibbles, name=name)
                status = ""
                if key_cells_colored[FORWARD][cell_name] > 0.5:
                    status = "fwd"
                elif key_cells_colored[BACKWARD][cell_name] > 0.5:
                    status = "bwd"
                elif key_cells_colored[SHARED][cell_name] > 0.5:
                    status = "mgd"
                res += ("""\\figcell[%s]{%i}{%i}{%s}""" %
                        (status, _cell_position_ctr, len(key_input_nibbles),
                         cell_name))
                res += """\n"""

                # display the key bits: up and down
                _tmp = _cell_position_ctr
                for key_bit in key_input_nibbles:
                    status = ""
                    if key_bits_colored[SHARED][key_bit] > 0.5:
                        status = "shared"
                    elif key_bits_colored[FORWARD][key_bit] > 0.5:
                        status = "fwd"
                    elif key_bits_colored[BACKWARD][key_bit] > 0.5:
                        status = "bwd"
                    res += ("""\\addkeyabove[%s]{%i}{%s}""" %
                            (status, _tmp, key_bit))
                    _tmp += 1
                _tmp = _cell_position_ctr
                for key_bit in key_output_nibbles:
                    status = ""
                    if key_bits_colored[SHARED][key_bit] > 0.5:
                        status = "shared"
                    elif key_bits_colored[FORWARD][key_bit] > 0.5:
                        status = "fwd"
                    elif key_bits_colored[BACKWARD][key_bit] > 0.5:
                        status = "bwd"
                    res += ("""\\addkeybelow[%s]{%i}{%s}""" %
                            (status, _tmp, key_bit))
                    _tmp += 1

                # update the display position
                _next_position_ctr = _cell_position_ctr + len(
                    key_input_nibbles)
                # if next position exceeds state size, go to another "round"
                if _next_position_ctr > self._state_size:
                    _cell_position_ctr = 0
                    res += """\\newround\n%\n"""
                else:
                    _cell_position_ctr = _next_position_ctr

        res += """\\end{figpresent}\n\end{document}\n"""
        return res


if __name__ == "__main__":

    pass
