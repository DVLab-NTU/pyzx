OPENQASM 2.0;
include "qelib1.inc";
qreg qubits[3];
h qubits[2];
cx qubits[1],qubits[2];
tdg qubits[2];
cx qubits[0],qubits[2];
s qubits[2];
s qubits[2];

