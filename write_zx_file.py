import pyzx as zx
from pyzx import simplify
from pyzx.utils import EdgeType, VertexType, FloatInt, FractionLike
from pyzx.circuit.gates import TargetMapper
from pyzx.graph import Graph
from pyzx.graph.base import BaseGraph, VT, ET
from fractions import Fraction
import numpy as np

import argparse

def write_zx_file(g: BaseGraph[VT, ET]) -> str:
    bzx = ""
    edge_dict={1:"S", 2:"H"}
    vertex_dict={0:"B", 1:"Z", 2:"X"}
    for i in g.inputs():
        s = ""
        s += "I"+str(i)+" "+str(g.qubit(i))+" "
        for n in g.neighbors(i):
            if((i, n) in g.edges()):
                s += edge_dict[g.edge_type((i,n))]+str(n)+" "
            else:
                s += edge_dict[g.edge_type((n,i))]+str(n)+" "
        bzx += s+"\n"

    for v in g.vertices():
        if v not in g.inputs() and v not in g.outputs():
            s = ""
            s += vertex_dict[g.type(v)]+str(v)+" "+str(g.qubit(v))+" "
            for n in g.neighbors(v):
                if((v, n) in g.edges()):
                    s += edge_dict[g.edge_type((v,n))]+str(n)+" "
                else:
                    s += edge_dict[g.edge_type((n,v))]+str(n)+" "
            if g.phase(v) != 0:
                frac = str(g.phase(v)).split('/')
                if frac[1] != 0:
                    s += frac[0]+"*pi/"+frac[1]
                else:
                    s += frac[0]+"*pi"
            bzx += s+"\n"

    for o in g.outputs():
        s = ""
        s += "O"+str(o)+" "+str(g.qubit(o))+" "
        for n in g.neighbors(o):
            if((o, n) in g.edges()):
                s += edge_dict[g.edge_type((o,n))]+str(n)+" "
            else:
                s += edge_dict[g.edge_type((n,o))]+str(n)+" "
        bzx += s+"\n"
    return bzx
    

def write_zx_graph(f: str, of: str) -> None:
    text_file = open(of, "w")
    circ = zx.Circuit.load(f)
    g = circ.to_graph()
    s = write_zx_file(g)
    text_file.write(s)
    text_file.close()



def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i','--input', help='Input the zx file.\n\n', required = True, metavar='zx_file')
    parser.add_argument('-o','--output', help='Input the pdf file.\n\n', required = True, metavar='pdf_file')
    
    args = parser.parse_args()

    write_zx_graph(args.input, args.output)

if __name__ == '__main__':
    main()
