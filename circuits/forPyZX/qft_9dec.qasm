OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(pi/2) q[0];
sx q[0];
rz(pi/2) q[0];
rz(pi/4) q[1];
rz(pi/4) q[0];
cx q[1],q[0];
rz(-pi/4) q[0];
cx q[1],q[0];
rz(pi/8) q[2];
rz(pi/8) q[0];
cx q[2],q[0];
rz(-pi/8) q[0];
cx q[2],q[0];
rz(pi/16) q[3];
rz(pi/16) q[0];
cx q[3],q[0];
rz(-pi/16) q[0];
cx q[3],q[0];
rz(pi/32) q[4];
rz(pi/32) q[0];
cx q[4],q[0];
rz(-pi/32) q[0];
cx q[4],q[0];
rz(pi/64) q[5];
rz(pi/64) q[0];
cx q[5],q[0];
rz(-pi/64) q[0];
cx q[5],q[0];
rz(pi/128) q[6];
rz(pi/128) q[0];
cx q[6],q[0];
rz(-pi/128) q[0];
cx q[6],q[0];
rz(pi/256) q[7];
rz(pi/256) q[0];
cx q[7],q[0];
rz(-pi/256) q[0];
cx q[7],q[0];
rz(pi/512) q[8];
rz(pi/512) q[0];
cx q[8],q[0];
rz(-pi/512) q[0];
cx q[8],q[0];
rz(pi/2) q[1];
sx q[1];
rz(pi/2) q[1];
rz(pi/4) q[2];
rz(pi/4) q[1];
cx q[2],q[1];
rz(-pi/4) q[1];
cx q[2],q[1];
rz(pi/8) q[3];
rz(pi/8) q[1];
cx q[3],q[1];
rz(-pi/8) q[1];
cx q[3],q[1];
rz(pi/16) q[4];
rz(pi/16) q[1];
cx q[4],q[1];
rz(-pi/16) q[1];
cx q[4],q[1];
rz(pi/32) q[5];
rz(pi/32) q[1];
cx q[5],q[1];
rz(-pi/32) q[1];
cx q[5],q[1];
rz(pi/64) q[6];
rz(pi/64) q[1];
cx q[6],q[1];
rz(-pi/64) q[1];
cx q[6],q[1];
rz(pi/128) q[7];
rz(pi/128) q[1];
cx q[7],q[1];
rz(-pi/128) q[1];
cx q[7],q[1];
rz(pi/256) q[8];
rz(pi/256) q[1];
cx q[8],q[1];
rz(-pi/256) q[1];
cx q[8],q[1];
rz(pi/2) q[2];
sx q[2];
rz(pi/2) q[2];
rz(pi/4) q[3];
rz(pi/4) q[2];
cx q[3],q[2];
rz(-pi/4) q[2];
cx q[3],q[2];
rz(pi/8) q[4];
rz(pi/8) q[2];
cx q[4],q[2];
rz(-pi/8) q[2];
cx q[4],q[2];
rz(pi/16) q[5];
rz(pi/16) q[2];
cx q[5],q[2];
rz(-pi/16) q[2];
cx q[5],q[2];
rz(pi/32) q[6];
rz(pi/32) q[2];
cx q[6],q[2];
rz(-pi/32) q[2];
cx q[6],q[2];
rz(pi/64) q[7];
rz(pi/64) q[2];
cx q[7],q[2];
rz(-pi/64) q[2];
cx q[7],q[2];
rz(pi/128) q[8];
rz(pi/128) q[2];
cx q[8],q[2];
rz(-pi/128) q[2];
cx q[8],q[2];
rz(pi/2) q[3];
sx q[3];
rz(pi/2) q[3];
rz(pi/4) q[4];
rz(pi/4) q[3];
cx q[4],q[3];
rz(-pi/4) q[3];
cx q[4],q[3];
rz(pi/8) q[5];
rz(pi/8) q[3];
cx q[5],q[3];
rz(-pi/8) q[3];
cx q[5],q[3];
rz(pi/16) q[6];
rz(pi/16) q[3];
cx q[6],q[3];
rz(-pi/16) q[3];
cx q[6],q[3];
rz(pi/32) q[7];
rz(pi/32) q[3];
cx q[7],q[3];
rz(-pi/32) q[3];
cx q[7],q[3];
rz(pi/64) q[8];
rz(pi/64) q[3];
cx q[8],q[3];
rz(-pi/64) q[3];
cx q[8],q[3];
rz(pi/2) q[4];
sx q[4];
rz(pi/2) q[4];
rz(pi/4) q[5];
rz(pi/4) q[4];
cx q[5],q[4];
rz(-pi/4) q[4];
cx q[5],q[4];
rz(pi/8) q[6];
rz(pi/8) q[4];
cx q[6],q[4];
rz(-pi/8) q[4];
cx q[6],q[4];
rz(pi/16) q[7];
rz(pi/16) q[4];
cx q[7],q[4];
rz(-pi/16) q[4];
cx q[7],q[4];
rz(pi/32) q[8];
rz(pi/32) q[4];
cx q[8],q[4];
rz(-pi/32) q[4];
cx q[8],q[4];
rz(pi/2) q[5];
sx q[5];
rz(pi/2) q[5];
rz(pi/4) q[6];
rz(pi/4) q[5];
cx q[6],q[5];
rz(-pi/4) q[5];
cx q[6],q[5];
rz(pi/8) q[7];
rz(pi/8) q[5];
cx q[7],q[5];
rz(-pi/8) q[5];
cx q[7],q[5];
rz(pi/16) q[8];
rz(pi/16) q[5];
cx q[8],q[5];
rz(-pi/16) q[5];
cx q[8],q[5];
rz(pi/2) q[6];
sx q[6];
rz(pi/2) q[6];
rz(pi/4) q[7];
rz(pi/4) q[6];
cx q[7],q[6];
rz(-pi/4) q[6];
cx q[7],q[6];
rz(pi/8) q[8];
rz(pi/8) q[6];
cx q[8],q[6];
rz(-pi/8) q[6];
cx q[8],q[6];
rz(pi/2) q[7];
sx q[7];
rz(pi/2) q[7];
rz(pi/4) q[8];
rz(pi/4) q[7];
cx q[8],q[7];
rz(-pi/4) q[7];
cx q[8],q[7];
rz(pi/2) q[8];
sx q[8];
rz(pi/2) q[8];
