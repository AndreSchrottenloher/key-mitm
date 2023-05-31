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
Generic solver to find MITM attacks on AES-like and Present-like ciphers and
compression functions. The code is built upon the one from [SS22]. Basically,
if one removes all the lines of code related to key bits, key cells, data
complexity, and computing the time complexity in the key-recovery case, then one
falls back on [SS22].

There is a simple interface allowing to use either SCIP (via the python package
pyscipopt) or Gurobi (via the python package gurobipy). Experiments showed that
Gurobi was much more efficient in practice.

The code in this file takes a Constraints object (see util.py),
generates a corresponding MILP model and solves it.

--Reference:
[SS22] Schrottenloher, Stevens, 'Simplified MITM Modeling for Permutations: 
New (Quantum) Attacks.', CRYPTO 2022, code available at:
https://github.com/AndreSchrottenloher/mitm-milp
"""

import math

scip_imported, gurobi_imported = False, False
try:
    from pyscipopt import Model as _SCIPModel
    from pyscipopt import quicksum as _scip_quicksum
    scip_imported = True
except ImportError as e:
    print("Cannot import pyscipopt")

try:
    from gurobipy import Model as _GurobiModel
    from gurobipy import GRB
    from gurobipy import quicksum as _gurobi_quicksum
    gurobi_imported = True
except ImportError as e:
    print("Cannot import gurobipy")

if not gurobi_imported and not scip_imported:
    raise ImportError("Could not import any MILP solver")


class SCIPModel:
    """SCIP Model interface."""

    def __init__(self):
        self.m = _SCIPModel()
        self.has_run = False

    def addVar(self, vtype, lb=None, ub=None):
        if vtype not in ["B", "C", "I"]:
            raise ValueError("Vtype")
        return self.m.addVar(vtype=vtype, lb=lb, ub=ub)

    def addCons(self, c):
        self.m.addCons(c)

    def quicksum(self, l):
        return _scip_quicksum(l)

    def setObjectiveMinimize(self, obj):
        self.m.setObjective(obj, sense="minimize")

    def optimize(self):
        self.m.optimize()
        self.has_run = True

    def getVal(self, v):
        if not self.has_run:
            raise Exception("Must run optimization first")
        else:
            return self.m.getVal(v)


class GurobiModel:
    """Gurobi model interface"""

    def __init__(self):
        self.m = _GurobiModel()
        self.m.setParam(
            GRB.Param.Threads,
            4)  # limit number of threads, to avoid overheating the CPU
        self.has_run = False

    def addVar(self, vtype, lb=None, ub=None):
        if vtype not in ["B", "C", "I"]:
            raise ValueError("Vtype")
        if lb is None and ub is None:
            return self.m.addVar(vtype=vtype)
        elif lb is None:
            return self.m.addVar(vtype=vtype, ub=ub)
        elif ub is None:
            return self.m.addVar(vtype=vtype, lb=lb)
        else:
            return self.m.addVar(vtype=vtype, lb=lb, ub=ub)

    def addCons(self, c):
        self.m.addConstr(c)

    def quicksum(self, l):
        return _gurobi_quicksum(l)

    def setObjectiveMinimize(self, obj):
        self.m.setObjective(obj, GRB.MINIMIZE)

    def optimize(self):
        self.m.optimize()
        self.has_run = True

    def getVal(self, v):
        if not self.has_run:
            raise Exception("Must run optimization first")
        else:
            return v.X


#========================================
# Flags for the coloration of cells and key bits

FORWARD = "forward"  # forward cells and key bits
BACKWARD = "backward"  # backward cells and key bits
MERGED = "merged"  # merged cells
SHARED = "shared"  # shared key bits (both forward and backward)
NEW_MERGED = "new_merged"  # new merged cells (for Super S-Boxes)

#=======================================
# Choice of attack setting for block cipher key-recovery:
# the Guess-and-determine (GAD) and Parallel MITM techniques are exclusive.

CLASSICAL_GAD = "classical"  # classical attacks using GAD
CLASSICAL_PARALLEL = "classical-parallel"  # classical attacks using Parallel MITM
QUANTUM_GAD = "quantum"  # quantum attacks using GAD
QUANTUM_PARALLEL = "quantum-parallel"  # quantum attacks using Parallel MITM
QUANTUM_SIMON = "simon"  # quantum Grover-meet-Simon attacks

#=======================================
# Choice of attack setting for preimage attacks
CLASSICAL_PREIMAGE = "classical-preimage"  # classical preimage attack
QUANTUM_PREIMAGE = "quantum-preimage"  # quantum preimage attack

# A small value that should be smaller than all the parameters considered
# during the MILP solving
EPSILON = 0.01


def round_to_str(v):
    return str(round(v, 2))


def get_val_dict(m, d):
    """
    In a dictionary of variables, replaces each variable by its value in a solved model.
    """
    for s in d:
        d[s] = round(m.getVal(d[s]), 5)


def get_val_dict_int(m, d):
    """
    In a dictionary of variables, replaces each variable by its value in a solved model.
    (Specific to integer variables, which are rounded).
    """
    for s in d:
        d[s] = int(round(m.getVal(d[s]), 5))


def find_mitm_attack(constraints,
                     time_target=None,
                     computation_model=CLASSICAL_GAD,
                     optimize_with_mem=True,
                     minimize_data=False,
                     data_limit=None,
                     memory_limit=None,
                     shared_key=[],
                     not_fwd_key=[],
                     not_bwd_key=[],
                     cut_forward=[],
                     cut_backward=[],
                     backward_hint=[],
                     forward_hint=[],
                     backward_zero=[],
                     forward_zero=[],
                     covered_rounds=[],
                     backend="gurobi",
                     round_bits=False):
    """
    Transforms the constraints into a MILP model and solves it using an 
    off-the-shelf solver (SCIP).
    
    computation_model -- choose between CLASSICAL_GAD, CLASSICAL_PARALLEL, 
        QUANTUM_GAD, QUANTUM_PARALLEL, CLASSICAL_PREIMAGE, QUANTUM_PREIMAGE, QUANTUM_SIMON
    
    Parameters of the optimization:
    time_target -- set a target for the time complexity (usually not a good idea)
    optimize_with_mem -- optimize the memory complexity as a secondary objective
    data_limit -- set a limit on the data complexity
    memory_limit -- set a limit on the memory
    minimize_data -- optimize the data complexity as a secondary objective
    backend -- solver to use: "gurobi" or "scip"
    round_bits -- specifies whether the nibble parameters must be rounded to integers
        (e.g., if the nibbles are actual bits). This will change the type of some
        variables from continuous to boolean, and might affect the runtime of the solver.

    Parameters that constrain the path:
    shared_key -- key nibbles that *must* be shared
    not_fwd_key -- key nibbles that *cannot* be forward
    not_bwd_key -- key nibbles that *cannot* be backward

    cut_forward -- list of rounds in which no cell is "forward"
    cut_backward -- list of rounds in which no cell is "backward"
    backward_hint -- list of cell names which must be set "backward"
    forward_hint -- list of cell names which must be set "forward"
    backward_zero -- list of cell names which cannot be set "backward"
    forward_zero -- list of cell names which cannot be set "forward"
    covered_rounds -- list of rounds which must be covered in the merged list
        (this includes "new merged" cells)
    """

    # access the graph data
    nrounds = constraints.nrounds
    key_bit_width = constraints.key_bit_width  # None by default
    (cells, cells_by_round, linear_constraints, linear_constraints_by_round,
     global_fixed) = constraints.get_data()

    # forward graph structure: for each cell, the cells which are related at the next round
    related_cells_atnextr = constraints.fwd_graph
    # backward graph structure
    related_cells_atprevr = constraints.bwd_graph

    # determine if we are in the Parallel MITM case or in the case of
    # a Grover-meet-Simon attack. This will work differently than the GAD framework
    parallel_case_or_simon = (computation_model in [
        CLASSICAL_PARALLEL, QUANTUM_PARALLEL, QUANTUM_SIMON
    ])
    preimage_case = (computation_model
                     in [CLASSICAL_PREIMAGE, QUANTUM_PREIMAGE])
    quantum_case = (computation_model in [
        QUANTUM_PARALLEL, QUANTUM_SIMON, QUANTUM_PREIMAGE, QUANTUM_GAD
    ])
    if minimize_data and preimage_case:
        raise ValueError("Cannot 'minimize data' in the preimage attack case")

    #============ CHECKS ON THE PARAMETERS ===================================

    # check that the "cut forward" and "cut backward" values are OK
    if cut_forward != []:
        for r in cut_forward:
            if r < 0 or r >= nrounds:
                raise ValueError("Bad cut-forward value")
    if cut_backward != []:
        for r in cut_backward:
            if r < 0 or r >= nrounds:
                raise ValueError("Bad cut-backward value")
    if covered_rounds != []:
        for r in covered_rounds:
            if r < 0 or r >= nrounds:
                raise ValueError("Bad  covered round value")

    # Check that all edges with key nibbles have the same width, equal to 1
    key_bit_width = 1
    for s in constraints.edge_name_to_data:
        c1, c2, width, key_bit = constraints.edge_name_to_data[s]
        if key_bit is not None:
            if width != 1:
                raise ValueError("Edge width (= key bit width) should be 1")

    #============= GENERIC COMPLEXITIES ======================================
    # Compute total state size and key size
    state_size = constraints.get_state_size()
    key_size = (
        len(constraints.key_array) -
        sum([constraints.key_cells[c][2] for c in constraints.key_cells]))
    key_size *= key_bit_width

    # Compute the generic time complexity
    generic_time_complexity = None
    quantum_factor = 0.5 if quantum_case else 1
    # Amount of "wrapping" between input and output (for the priemage case)
    wrapping_cons = (sum([
        linear_constraints[s][2]
        for s in linear_constraints_by_round[nrounds - 1]
    ]))
    # Number of solutions in the path (for the preimage case)
    path_solutions = state_size - wrapping_cons

    if preimage_case:
        # preimage attack: can be partial preimage if only partial wrapping
        generic_time_complexity = quantum_factor * wrapping_cons
    else:
        # key-recovery attack
        generic_time_complexity = quantum_factor * key_size

    # sort the lists of cut rounds forwards and backwards
    cut_fwd = sorted(cut_forward)
    # the first cut fwd round is first in increasing order
    cut_bwd = sorted(cut_backward)  # the first cut bwd round is last
    cut_bwd.reverse()

    #============== DEFINITION OF THE MODEL AND CELL COLORATION =================
    if backend == "scip" and scip_imported:
        m = SCIPModel()
    else:
        m = GurobiModel()

    labels = [FORWARD, BACKWARD, MERGED]

    # Main variables: the colored cells (booleans)
    cell_var_colored = {}
    for l in [FORWARD, BACKWARD, NEW_MERGED, MERGED]:
        cell_var_colored[l] = {}
        for c in cells:
            cell_var_colored[l][c] = m.addVar(vtype="B")

    # Constraints on the colorations of cells
    for c in cells:
        # forward, backward and new_merged are exclusive
        m.addCons(
            cell_var_colored[FORWARD][c] + cell_var_colored[BACKWARD][c] +
            cell_var_colored[NEW_MERGED][c] == cell_var_colored[MERGED][c])
        if c not in constraints.get_super_sboxes():
            # is not a super s-box: cannot be a "new merged" cell
            m.addCons(cell_var_colored[NEW_MERGED][c] == 0)

    # "new merged" cells cannot be created at two consecutive rounds
    # (the actual constraints are rather: "new merged" cells cannot be linked;
    # this is a simplification that reduces the number of constraints)
    hasnewmgd = {}
    for r in range(nrounds):
        hasnewmgd[r] = m.addVar(vtype="B")
        for c in cells_by_round[r]:
            m.addCons(hasnewmgd[r] >= cell_var_colored[NEW_MERGED][c])
    for r in range(nrounds):
        m.addCons(hasnewmgd[r] + hasnewmgd[(r + 1) % nrounds] <= 1)

    #============= SIMPLIFICATIONS OF THE PATH ================================
    # (this represents very specific cases)
    for e in constraints.edge_name_to_data:
        (c1, c2, w, key_bit) = constraints.edge_name_to_data[e]
        if key_bit is None and w == constraints.get_cell_width(c2):
            m.addCons(
                cell_var_colored[FORWARD][c2] == cell_var_colored[FORWARD][c1])
        if key_bit is None and w == constraints.get_cell_width(c1):
            m.addCons(cell_var_colored[BACKWARD][c2] ==
                      cell_var_colored[BACKWARD][c1])

    # This case can happen for a partial "wrapping"
    for c in cells:
        if (not constraints.fwd_edges[c]) or (not constraints.bwd_edges[c]):
            for label in [FORWARD, BACKWARD, NEW_MERGED, MERGED]:
                m.addCons(cell_var_colored[label][c] == 0)

    #========== CONSTRAINTS ON THE PATH (HINTS) ==============================
    for c in backward_hint:
        if c in cell_var_colored[BACKWARD]:
            m.addCons(cell_var_colored[BACKWARD][c] == 1)

    for c in forward_hint:
        if c in cell_var_colored[FORWARD]:
            m.addCons(cell_var_colored[FORWARD][c] == 1)

    for c in backward_zero:
        if c in cell_var_colored[BACKWARD]:
            m.addCons(cell_var_colored[BACKWARD][c] == 0)

    for c in forward_zero:
        if c in cell_var_colored[FORWARD]:
            m.addCons(cell_var_colored[FORWARD][c] == 0)

    #======== CONSTRAINTS ON THE PATH: CUT ROUNDS & COVERED ROUNDS ===========
    cut_fwd_rounds = {}
    cut_bwd_rounds = {}
    cov_rounds = {}
    for r in range(nrounds):
        cut_bwd_rounds[r] = m.addVar(vtype="B")
        cut_fwd_rounds[r] = m.addVar(vtype="B")
        cov_rounds[r] = m.addVar(vtype="B")
    m.addCons(m.quicksum([cut_fwd_rounds[r] for r in range(nrounds)]) >= 1)
    m.addCons(m.quicksum([cut_bwd_rounds[r] for r in range(nrounds)]) >= 1)
    m.addCons(m.quicksum([cov_rounds[r] for r in range(nrounds)]) >= 1)

    # no cell var colored at the cut round(s)
    for r in range(nrounds):
        for c in cells_by_round[r]:
            m.addCons(cell_var_colored[FORWARD][c] <= 1 - cut_fwd_rounds[r])
            m.addCons(cell_var_colored[BACKWARD][c] <= 1 - cut_bwd_rounds[r])
            m.addCons(cell_var_colored[MERGED][c] >= cov_rounds[r])

    # we can set the cut rounds manually
    if cut_fwd != []:
        for r in range(nrounds):
            m.addCons(cut_fwd_rounds[r] == (1 if r in cut_fwd else 0))
    if cut_bwd != []:
        for r in range(nrounds):
            m.addCons(cut_bwd_rounds[r] == (1 if r in cut_bwd else 0))
    if covered_rounds != []:
        for r in range(nrounds):
            m.addCons(cov_rounds[r] == (1 if r in covered_rounds else 0))

    #=========== VARIABLES FOR THE COLORATION OF KEY BITS =====================
    # coloration of key bits. These variables are relaxed to continuous, which
    # works fine in practice.
    key_var_colored = {}
    for l in [BACKWARD, FORWARD, SHARED]:
        key_var_colored[l] = {}
        for k in constraints.key_array:
            # in some rare cases, we can color key bits partially (for example,
            # half of the key byte goes in forward, the other half in backward)
            if not round_bits:
                key_var_colored[l][k] = m.addVar(vtype="C", lb=0, ub=1)
            else:
                key_var_colored[l][k] = m.addVar(vtype="B")

    # not all key bits need to be colored. Both cases (GAD/Parallel) work fine as long as
    # they allow to decrease the number of possibilities for the key bits that
    # intervene in the path, i.e., there is some non-trivial forward/backward matching.
    for k in constraints.key_array:
        # in all cases, the key bits status (shared, backward, forward) is exclusive
        # having e.g. a key bit which is 0.5 backward and 0.5 forward is not
        # impossible, and not a problem (if this is a byte, for example, we can
        # separate the 2^8 possible values in a product of 2^4 * 2^4 pairs)
        m.addCons(key_var_colored[SHARED][k] + key_var_colored[BACKWARD][k] +
                  key_var_colored[FORWARD][k] <= 1)
        # unless in the "parallel" MITM case, or Grover-meet-Simon, all key bits
        # will have to be covered. This simplifies the constraints significantly.
        if not parallel_case_or_simon:
            m.addCons(key_var_colored[SHARED][k] +
                      key_var_colored[BACKWARD][k] +
                      key_var_colored[FORWARD][k] == 1)

    # constraints to simplify with some "hints"
    for k in shared_key:
        m.addCons(key_var_colored[SHARED][k] == 1)
    for k in not_fwd_key:
        # key bit must be bwd or shared
        m.addCons(key_var_colored[FORWARD][k] == 0)
    for k in not_bwd_key:
        # key bit must be fwd or shared
        m.addCons(key_var_colored[BACKWARD][k] == 0)

    #=============== CONSTRAINTS ON THE COLORATION OF KEY BITS
    # The coloration of cells will determine the coloration of key bits, as follows.
    # Cells can have three exclusive types: FORWARD (fwd), BACKWARD (bwd) or NEW_MERGED (newm)
    # - any link fwd-fwd implies that the key bit is fwd or shared
    # - any link bwd-bwd implies that the key bit is bwd or shared
    # - any link between two colored cells implies that the key bit is fwd, bwd or shared

    # - any link fwd-newm implies that the key bit is fwd or shared
    # - any link bwd-newm is impossible
    # - any link newm-fwd is impossible
    # - any link newm-bwd implies nothing on the key bit (can be bwd or fwd or shared)
    # - any link newm-newm is impossible

    for s in linear_constraints:
        c1, c2, w, k = constraints.edge_data(s)
        #m.addCons( cell_var_colored[NEW_MERGED][c1] + cell_var_colored[NEW_MERGED][c2] <= 1 )
        if k is not None:
            if computation_model == QUANTUM_SIMON:
                m.addCons(1 + key_var_colored[SHARED][k] >=
                          cell_var_colored[BACKWARD][c1] +
                          cell_var_colored[BACKWARD][c2])
                # if fwd-fwd or fwd-newm, then shared
                m.addCons(1 + key_var_colored[SHARED][k] >=
                          cell_var_colored[FORWARD][c1] +
                          cell_var_colored[NEW_MERGED][c2] +
                          cell_var_colored[FORWARD][c2])
                # newm-fwd
                m.addCons(cell_var_colored[NEW_MERGED][c1] +
                          cell_var_colored[FORWARD][c2] <= 1)
                # bwd-newm
                m.addCons(cell_var_colored[BACKWARD][c1] +
                          cell_var_colored[NEW_MERGED][c2] <= 1)
            else:
                # if any link, then bwd or fwd or shared
                if computation_model in [CLASSICAL_PARALLEL, QUANTUM_PARALLEL]:
                    m.addCons(1 + key_var_colored[FORWARD][k] +
                              key_var_colored[SHARED][k] +
                              key_var_colored[BACKWARD][k] >=
                              cell_var_colored[MERGED][c1] +
                              cell_var_colored[MERGED][c2])
                # if fwd-fwd or fwd-newm, then fwd or shared
                m.addCons(1 + key_var_colored[FORWARD][k] +
                          key_var_colored[SHARED][k] >=
                          cell_var_colored[FORWARD][c1] +
                          cell_var_colored[NEW_MERGED][c2] +
                          cell_var_colored[FORWARD][c2])
                # if bwd-bwd or bwd-newm, then bwd or shared
                m.addCons(1 + key_var_colored[BACKWARD][k] +
                          key_var_colored[SHARED][k] >=
                          cell_var_colored[BACKWARD][c1] +
                          cell_var_colored[BACKWARD][c2] +
                          cell_var_colored[NEW_MERGED][c2])

    #======== KEY CELL VARIABLES ==============================================
    # A "key cell" is for example an S-Box in the Present key-schedule, which defines
    # 4 output key nibbles from 4 input key nibbles. This creates a relation between these key nibbles.
    # When the key cell is "active", this relation is used to reduce the effective key length.
    # A key cell can have three (exclusive) colors: shared (it is reduced globally),
    # forward (it is reduced only in the fwd path), backward (it is reduced only in the bwd path)
    #key_cell_width = 0
    key_cells_colored = {}
    for l in [FORWARD, BACKWARD, SHARED]:
        key_cells_colored[l] = {}
        for c in constraints.key_cells:
            key_nibbles_input, key_nibbles_output, w = constraints.key_cells[c]
            key_cells_colored[l][c] = m.addVar(vtype="B")

    #===== CONSTRAINTS ON COLORATION OF KEY CELLS ============================
    for c in constraints.key_cells:
        # color is exclusive
        m.addCons(
            key_cells_colored[SHARED][c] + key_cells_colored[FORWARD][c] +
            key_cells_colored[BACKWARD][c] <= 1)
        key_nibbles_input, key_nibbles_output, w = constraints.key_cells[c]
        for k in key_nibbles_input + key_nibbles_output:
            m.addCons(
                key_var_colored[SHARED][k] >= key_cells_colored[SHARED][c])
            # forward key cells: all key nibbles must be fwd or shared
            m.addCons(
                key_var_colored[SHARED][k] +
                key_var_colored[FORWARD][k] >= key_cells_colored[FORWARD][c])
            # bwd: same
            m.addCons(
                key_var_colored[SHARED][k] +
                key_var_colored[BACKWARD][k] >= key_cells_colored[BACKWARD][c])

    #======== TOTAL CONTRIBUTION OF KEY NIBBLES ==================================
    # the total amount of shared / fwd / bwd key nibbles is deduced from the
    # sum of key bit coloration, minus the active key cells
    total_key_nibbles = {}
    for l in [FORWARD, BACKWARD, SHARED]:
        total_key_nibbles[l] = m.addVar(vtype="C", lb=0)
        m.addCons(total_key_nibbles[l] == m.quicksum(
            key_var_colored[l][k]
            for k in constraints.key_array) - m.quicksum([
                key_cells_colored[l][c] * constraints.get_key_cell_width(c)
                for c in constraints.key_cells
            ]))
    total_key_nibbles[MERGED] = m.addVar(vtype="C", lb=0)
    # total amount of key nibbles in the merged list
    m.addCons(total_key_nibbles[MERGED] == total_key_nibbles[BACKWARD] +
              total_key_nibbles[FORWARD])

    # total amount of key nibbles shared between fwd and bwd
    key_nibbles_fixed = m.addVar(vtype="C", lb=0)
    m.addCons(key_nibbles_fixed == total_key_nibbles[SHARED] * key_bit_width)

    #=============== DATA COMPLEXITY ==========================================
    # When we want to minimize the data complexity, we compute it using all edges
    # between the "cipher" cell and other cells.
    # If the cipher cell is fwd, then all such edges of the form bwd-fwd, where the
    # key bit is not fwd, contribute negatively to the data complexity
    # If the cipher cell is bwd, then all such edges of the form bwd-fwd, where the key bit
    # is not bwd, contribute negatively to the data complexity
    data_comp = m.addVar(vtype="C", lb=0, ub=state_size)

    # assume that cipher cell is in position "nrounds-1". This is always the case
    # if the constraints were produced by a Cipher object.
    cipher_cell = cells_by_round[nrounds - 1][0]
    _useful_edges = []
    for s in linear_constraints:
        c1, c2, w, k = constraints.edge_data(s)
        if c1 == cipher_cell or c2 == cipher_cell:
            _useful_edges.append(s)

    _data_comp_vars = []
    for s in _useful_edges:
        c1, c2, w, k = constraints.edge_data(s)
        # we count all edges of the form bwd-fwd among these edges, where the key bit
        # is not of the same color as the cipher cell. We add one temporary variable
        # for each edge.
        _new = m.addVar(vtype="C", lb=0, ub=1)
        _data_comp_vars.append(_new)
        # if there is no key bit
        if k is None:
            # c1 -> c2: c1 is bwd and c2 is fwd
            m.addCons(_new <= cell_var_colored[BACKWARD][c1])
            m.addCons(_new <= cell_var_colored[FORWARD][c2])
        else:
            # same constraint, but also key bit must be of different coloration
            # as cipher cell: if cipher cell is fwd, then it cannot be fwd;
            # if cipher cell is bwd, then it cannot be bwd
            m.addCons(_new <= cell_var_colored[BACKWARD][c1])
            m.addCons(_new <= cell_var_colored[FORWARD][c2])
            m.addCons(_new <= 2 - cell_var_colored[FORWARD][cipher_cell] -
                      key_var_colored[FORWARD][k])
            m.addCons(_new <= 2 - cell_var_colored[BACKWARD][cipher_cell] -
                      key_var_colored[BACKWARD][k])

    data_comp = m.addVar(vtype="C", lb=0)
    m.addCons(data_comp == state_size - m.quicksum(_data_comp_vars))

    if data_limit is not None:
        m.addCons(data_comp <= data_limit)
    #================ END: DATA COMPLEXITY

    #=============== CONSTRAINTS ON LISTS =====================================
    colored_bwd_at_prevr = {
    }  # for a given cell, amount of relation with "bwd" cells at the previous round
    for c in cells:
        colored_bwd_at_prevr[c] = m.addVar(vtype="C", lb=0)
        m.addCons(colored_bwd_at_prevr[c] == m.quicksum([
            related_cells_atprevr[c][cc] * cell_var_colored[BACKWARD][cc]
            for cc in related_cells_atprevr[c]
        ]))
    colored_fwd_at_prevr = {
    }  # for a given cell, amount of relation with "fwd" cells at the previous round
    for c in cells:
        colored_fwd_at_prevr[c] = m.addVar(vtype="C", lb=0)
        m.addCons(colored_fwd_at_prevr[c] == m.quicksum([
            related_cells_atprevr[c][cc] * cell_var_colored[FORWARD][cc]
            for cc in related_cells_atprevr[c]
        ]))
    colored_bwd_at_nextr = {
    }  # for a given cell, amount of relation with "bwd" cells at the next round
    for c in cells:
        colored_bwd_at_nextr[c] = m.addVar(vtype="C", lb=0)
        m.addCons(colored_bwd_at_nextr[c] == m.quicksum([
            related_cells_atnextr[c][cc] * cell_var_colored[BACKWARD][cc]
            for cc in related_cells_atnextr[c]
        ]))
    colored_fwd_at_nextr = {
    }  # for a given cell, amount of relation with "fwd" cells at the previous round
    for c in cells:
        colored_fwd_at_nextr[c] = m.addVar(vtype="C", lb=0)
        m.addCons(colored_fwd_at_nextr[c] == m.quicksum([
            related_cells_atnextr[c][cc] * cell_var_colored[FORWARD][cc]
            for cc in related_cells_atnextr[c]
        ]))

    # variables of "global reduction" = fixed internal state values (possibly through
    # MC in AES-like designs)
    global_reduction_vars = {}
    for c in cells:
        global_reduction_vars[c] = m.addVar(vtype="C", lb=0, ub=cells[c])
    global_reduction = m.addVar(vtype="C", lb=0)

    for c in cells:
        if c in constraints.get_super_sboxes():
            # if the cell is a super s-box, the "global reduction" counts both:
            # - the amount of edges between this cell (if forward) and backward edges,
            # which can be globally fixed
            # - the amount of reduction through MC
            m.addCons(cells[c] * cell_var_colored[MERGED][c] >= cells[c] *
                      cell_var_colored[FORWARD][c])
            # zero if the cell is not "forward" and not "new_merged"
            m.addCons(global_reduction_vars[c] <= cells[c] *
                      cell_var_colored[NEW_MERGED][c] +
                      cells[c] * cell_var_colored[FORWARD][c])
            # globally fixed through MC
            m.addCons(global_reduction_vars[c] <= cells[c] *
                      cell_var_colored[FORWARD][c] +
                      cells[c] * cell_var_colored[BACKWARD][c] -
                      cells[c] * cell_var_colored[MERGED][c] +
                      colored_bwd_at_prevr[c] + colored_bwd_at_nextr[c] +
                      colored_fwd_at_prevr[c] + colored_fwd_at_nextr[c])
            # always up to colored_bwd_at_prevr[c]
            m.addCons(global_reduction_vars[c] <= colored_bwd_at_prevr[c])

            # if the cell is not forward (i.e. "new merged"), then the amount of
            # reduction through MC is
            # up to the amount of links with fwd cells at the next round
            m.addCons(global_reduction_vars[c] <= cells[c] *
                      cell_var_colored[FORWARD][c] + colored_fwd_at_nextr[c])
        else:
            # if the cell is not a super s-box, the "global reduction" is counted only
            # if it is forward, and up to the amount of links with backward cells
            # at the previous round.
            m.addCons(global_reduction_vars[c] <= cells[c] *
                      cell_var_colored[FORWARD][c])
            m.addCons(global_reduction_vars[c] <= colored_bwd_at_prevr[c])

    # global reduction in the path
    m.addCons(global_reduction == m.quicksum(
        [global_reduction_vars[c] for c in cells]))

    if parallel_case_or_simon:
        m.addCons(global_reduction <= state_size)

    #============ LIST SIZES =================================================
    list_sizes = {}
    for label in labels:
        # in the Simon case, backward and forward list sizes are counted as 0,
        # merged list size will be counted as negative (since there are no key nibbles
        # that intervene)
        list_sizes[label] = m.addVar(vtype="C", lb=-1)

    cell_contrib = {}
    for label in [FORWARD, BACKWARD, MERGED]:
        cell_contrib[label] = {}
        for c in cells:
            if label == FORWARD:
                prev_contrib = colored_fwd_at_prevr[c]
            elif label == BACKWARD:
                prev_contrib = colored_bwd_at_nextr[c]
            else:
                prev_contrib = m.quicksum([
                    related_cells_atnextr[c][cc] * cell_var_colored[MERGED][cc]
                    for cc in related_cells_atnextr[c]
                ])

            # Contribution of cell. Takes into account "XOR" and "branching" cells
            # which were also supported in [SS22], although we didn't use them.
            # Each cell contributes to the total list size, but its contribution
            # can be negative in these very specific cases (not in the AES-like /
            # Present-like cases)
            if label == BACKWARD or label == MERGED:
                lower_bound = min(0, cells[c] - constraints.fwd_edges_width(c))
            elif label == FORWARD:
                lower_bound = min(0, cells[c] - constraints.bwd_edges_width(c))

            cell_contrib[label][c] = m.addVar(vtype="C",
                                              lb=lower_bound,
                                              ub=cells[c])
            m.addCons(cell_contrib[label][c] >= cell_var_colored[label][c] *
                      cells[c] - prev_contrib)

            # required only if cell is a "branching" cell (width of the cell is
            # smaller than the total width of output edges)
            if c in constraints.get_branching_cells():
                m.addCons(cell_contrib[label][c] >= -100 *
                          (cell_var_colored[label][c]))

        m.addCons(list_sizes[label] ==
                  m.quicksum([cell_contrib[label][c] for c in cells]) -
                  global_reduction + total_key_nibbles[label] * key_bit_width)

    if parallel_case_or_simon:
        # For the Parallel-MITM technique, or in the Grover-meet-Simon case, we do
        # not need to compute the merged list size. The only constraint is that there
        # is some matching between forward and backward, and that there is no
        # guess-and-determine in both lists (i.e., no contribution of cell that
        # exceeds the global_reduction variables).
        m.addCons(
            m.quicksum([cell_contrib[FORWARD][c]
                        for c in cells]) - global_reduction == 0)
        m.addCons(
            m.quicksum([cell_contrib[BACKWARD][c]
                        for c in cells]) - global_reduction == 0)
        m.addCons(
            m.quicksum([cell_contrib[MERGED][c]
                        for c in cells]) - global_reduction <= -EPSILON)

    max_list_size = m.addVar(vtype="C", lb=0)
    for label in labels:
        # do not count merged list size if we are in the parallel case / Grover-meet-Simon case
        if not (label == MERGED and parallel_case_or_simon):
            m.addCons(max_list_size >= list_sizes[label])

    #======= CONSTRAINTS ON LIST SIZES, TIME COMP., MEMORY COMP. & OBJECTIVE ==
    time_comp = m.addVar(vtype="C", lb=0)
    memory_comp = m.addVar(vtype="C", lb=0)

    # memory comp = min(forward list size, backward list size)
    switch = m.addVar(vtype="B")
    m.addCons(memory_comp >= list_sizes[FORWARD] - 100 * switch)
    m.addCons(memory_comp >= list_sizes[BACKWARD] - 100 * (1 - switch))

    # repetitions are determined by fixed key / state nibbles
    repetitions = m.addVar(vtype="C", lb=0)

    # if we're solving a preimage problem, the amount of repetitions we can
    # have is different
    if preimage_case:
        m.addCons(repetitions >= key_nibbles_fixed + global_reduction -
                  path_solutions - key_size)
    else:
        rep1 = m.addVar(vtype="C", lb=0)
        rep2 = m.addVar(vtype="C", lb=0)
        m.addCons(rep1 >= global_reduction - state_size)  # state repetitions
        m.addCons(rep2 >= key_nibbles_fixed)  # key repetitions
        m.addCons(repetitions >= rep1 + rep2)

    if not quantum_case:
        # classical setting: repetition loop + merging time
        m.addCons(time_comp >= max_list_size + repetitions)
    else:
        # quantum case
        m.addCons(time_comp >= 0.5 * max_list_size + 0.5 * repetitions)
        m.addCons(time_comp >= memory_comp + 0.5 * repetitions)

    if minimize_data:
        # objective: time first, and then for a given time, data complexity
        m.setObjectiveMinimize(1000 * (time_comp) + data_comp)
    elif optimize_with_mem:
        # objective: time first, then memory complexity
        m.setObjectiveMinimize(1000 * (time_comp) + memory_comp)
    else:
        m.setObjectiveMinimize(time_comp)

    if memory_limit is not None:
        m.addCons(memory_comp <= memory_limit)
    if time_target is not None:
        m.addCons(time_comp == time_target)

    #=====================================================================

    m.optimize()

    #================= INTERPRET THE RESULTS =================================
    success = (round(m.getVal(time_comp), 5) < generic_time_complexity)

    results_lines = []
    results_lines.append("""Generic time complexity:  %s """ %
                         (str(generic_time_complexity)))
    results_lines.append("""Max list size:  %s """ %
                         (str(m.getVal(max_list_size))))
    results_lines.append("""Memory complexity:  %s """ %
                         (str(m.getVal(memory_comp))))
    if minimize_data:
        results_lines.append("""Data complexity:  %s """ %
                             (str(m.getVal(data_comp))))
    results_lines.append("""Time complexity:  %s """ %
                         (str(m.getVal(time_comp))))
    results_lines.append("""Fixed state nibbles (global_reduction):  %s """ %
                         (str(m.getVal(global_reduction))))
    results_lines.append("""Repetitions:  %s """ %
                         (str(m.getVal(repetitions))))

    get_val_dict(m, global_reduction_vars)
    results_lines.append("Global reduction of cells: ")
    for c in global_reduction_vars:
        if global_reduction_vars[c] != 0:
            results_lines.append(
                str(c) + " : " + str(global_reduction_vars[c]))

    for label in cell_var_colored:
        get_val_dict_int(m, cell_var_colored[label])
    for label in key_var_colored:
        get_val_dict_int(m, key_var_colored[label])
    for label in key_cells_colored:
        get_val_dict_int(m, key_cells_colored[label])

    # post-process: for each key cell, if >= key_nibbles_input are shared, then
    # put all of the others in shared (does not change the time complexity, and
    # simplifies the writing of the output)
    for c in constraints.key_cells:
        key_nibbles_input, key_nibbles_output, w = constraints.key_cells[c]
        if (sum([
                key_var_colored[SHARED][k]
                for k in key_nibbles_input + key_nibbles_output
        ]) >= len(key_nibbles_input)):
            for k in key_nibbles_input + key_nibbles_output:
                key_var_colored[BACKWARD][k] = 0
                key_var_colored[FORWARD][k] = 0
                key_var_colored[SHARED][k] = 1
            key_cells_colored[SHARED][c] = 1
            key_cells_colored[FORWARD][c] = 0
            key_cells_colored[BACKWARD][c] = 0

    get_val_dict(m, total_key_nibbles)
    get_val_dict(m, list_sizes)
    get_val_dict(m, cell_contrib[MERGED])

    results_lines.append("======= Number of key nibbles:")
    for label in total_key_nibbles:
        results_lines.append(label + " : " + str(total_key_nibbles[label]))

    results_lines.append("======= List sizes:")
    for label in list_sizes:
        results_lines.append(label + " : " + str(list_sizes[label]))

    results_lines.append("======= Key nibbles:")
    for l in [FORWARD, BACKWARD, SHARED]:
        _l = [k for k in constraints.key_array if key_var_colored[l][k] == 1]
        results_lines.append(str(l) + " " + str(_l) + " " + str(len(_l)))

    results_lines.append("============= Key cells:")
    for c in constraints.key_cells:
        results_lines.append(str(c) + " " + str(constraints.key_cells[c]))
    results_lines.append("============= key cells colored:")

    for label in [FORWARD, BACKWARD, SHARED]:
        results_lines.append(
            str(label) + " " + str([
                c for c in key_cells_colored[label]
                if key_cells_colored[label][c] == 1
            ]))

    results_lines.append("============= Covered cells:")
    for label in [FORWARD, BACKWARD, NEW_MERGED]:
        results_lines.append(
            str(label) + " " + str([
                c for c in cell_var_colored[label]
                if cell_var_colored[label][c] == 1
            ]))

    return success, cell_var_colored, key_var_colored, key_cells_colored, results_lines
