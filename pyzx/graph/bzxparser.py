import pyzx as zx
from pyzx import simplify
from pyzx.utils import EdgeType, VertexType, FloatInt, FractionLike
from pyzx.circuit.gates import TargetMapper
from pyzx.graph import Graph
from pyzx.graph.base import BaseGraph, VT, ET
from fractions import Fraction
from typing import List, Dict, Any, Optional
import numpy as np

def read_zx_file(f: str):
    my_file = open(f)
    data = my_file.read()
    line_list = [ line.split(' //')[0] for line in data.split('\n') if line[0:2] != '//' and line != '']
    i_line, o_line, v_line = [],[],[]
    for line in line_list:
        if line[0] == 'I':
            i_line.append(line)
        if line[0] == 'O':
            o_line.append(line)
        if line[0] == 'Z' or line[0] == 'X' or line[0] == 'H':
            v_line.append(line)
    
    class IO:
        def __init__(self, id, qubit, column, neighbors):
            self.id = id
            self.qubit = qubit
            self.column = column
            self.neighbors = neighbors
    class V:
        def __init__(self, id, type, phase, qubit, column, neighbors):
            self.id = id
            self.type = type
            self.phase = phase
            self.qubit = qubit
            self.column = column
            self.neighbors = neighbors
            self.marked = False
        
    inputs, outputs, vertices = [],[],[]
    # for i in i_line:
    #     items = i.split(' ')
    #     input = IO(int(items[0][1:]),int(items[1]),int(items[2][1:]),[(items[3][0], int(items[3][1:]))])
    #     inputs.append(input)
    # for o in o_line:
    #     items = o.split(' ')
    #     output = IO(int(items[0][1:]),int(items[1]),int(items[2][1:]), [(items[3][0], int(items[3][1:]))])
    #     outputs.append(output)
    for v in i_line + o_line + v_line:
        items = v.split(' ')
        id = int(items[0][1:])
        type = items[0][0]
        qubit = int(items[1])
        column = int(items[2][1:])
        phase = 0
        neighbors = []
        if len(items) > 3:
            last_item = items[len(items)-1]
            if last_item[0] != 'S' and last_item[0] != 'H':
                if last_item.find('pi') != -1:
                    if last_item[:2] == 'pi':
                        last_item = '1*' + last_item
                    if last_item[-2:] == 'pi':
                        last_item += '/1'
                    num, denum = int(last_item.split('*pi/')[0]), int(last_item.split('*pi/')[1])
                    phase = Fraction(num, denum)
                    items = items[:-1]
                
                else:
                    phase = Fraction(float(last_item)/np.pi).limit_denominator(1000)
                    items = items[:-1]
        for item in items[3:]:
            et, nId = item[0], int(item[1:])
            neighbors.append((et, nId))
        
        vertex = V(id, type, phase, qubit, column, neighbors)
        vertices.append(vertex)
    return inputs, outputs, vertices

def bzx_to_graph(bzx: str, backend:Optional[str]=None) -> BaseGraph[VT, ET]:
    # file read
    inputs, outputs, vertices = read_zx_file(bzx)
    
    # print("Inputs  :")
    # for v in inputs:
    #     print(f"  - id: {v.id}, qubit: {v.qubit}, column: {v.column}, neighbor: {v.neighbors}")
    # print("Outputs :")
    # for v in outputs:
    #     print(f"  - id: {v.id}, qubit: {v.qubit}, column: {v.column}, neighbor: {v.neighbors}")
    # print("Vertices:")
    # for v in vertices:
    #     print(f"  - id: {v.id}, qubit: {v.qubit}, column: {v.column}, neighbor: {v.neighbors}")
    g = Graph(backend)
    # q_mapper: TargetMapper[VT] = TargetMapper()
    inputList, outputList = [], []
    id2label = {}
    
    for vertex in inputs + outputs + vertices:
        new_label = len(id2label)
        id2label[vertex.id] = new_label
        if (vertex.type == 'I' or vertex.type == 'O'): 
            vtype = VertexType.BOUNDARY
        elif (vertex.type == 'H'): 
            vtype = VertexType.H_BOX
        elif (vertex.type == 'Z'):
            vtype = VertexType.Z
        elif (vertex.type == 'X'):
            vtype = VertexType.X
        else: 
            raise TypeError(f"Unexpected vertex type ({vertex.type}).")
        
        v = g.add_vertex(vtype, vertex.qubit, vertex.column, vertex.phase)
        if (vertex.type == 'I'):
            inputList.append(v)
        elif (vertex.type == 'O'):
            outputList.append(v)
            
        vertex.marked = True
    
    # check miss edges
    for v in inputs + outputs + vertices:
        new_label_s = id2label[v.id]
        for n in v.neighbors:
            new_label_t = id2label[n[1]]
            if new_label_t not in g.neighbors(new_label_s):
                etype = EdgeType.SIMPLE if n[0] == 'S' else EdgeType.HADAMARD       
                g.add_edge((new_label_s, new_label_t), etype)
    
    
    g.set_inputs(tuple(inputList))
    g.set_outputs(tuple(outputList))

    return g

def graph_to_bzx(g: BaseGraph[VT,ET]) -> str:
    bzx = ""
    edge_dict={1:"S", 2:"H"}
    vertex_dict={0:"B", 1:"Z", 2:"X", 3:"H"}
    for i in g.inputs():
        s = ""
        s += "I"+str(i)+" "+str(g.qubit(i))+" C"+str(g.row(i))+" "
        for n in g.neighbors(i):
            if((i, n) in g.edges()):
                s += edge_dict[g.edge_type((i,n))]+str(n)+" "
            else:
                s += edge_dict[g.edge_type((n,i))]+str(n)+" "
        bzx += s+"\n"

    for o in g.outputs():
        s = ""
        s += "O"+str(o)+" "+str(g.qubit(o))+" C"+str(g.row(o))+" "
        for n in g.neighbors(o):
            if((o, n) in g.edges()):
                s += edge_dict[g.edge_type((o,n))]+str(n)+" "
            else:
                s += edge_dict[g.edge_type((n,o))]+str(n)+" "
        bzx += s+"\n"

    for v in g.vertices():
        if v not in g.inputs() and v not in g.outputs():
            s = ""
            s += vertex_dict[g.type(v)]+str(v)+" "+str(g.qubit(v))+" C"+str(g.row(v))+" "
            for n in g.neighbors(v):
                if((v, n) in g.edges()):
                    s += edge_dict[g.edge_type((v,n))]+str(n)+" "
                else:
                    s += edge_dict[g.edge_type((n,v))]+str(n)+" "
            if g.phase(v) != 0:
                frac = str(g.phase(v)).split('/')
                if len(frac) == 2 and frac[1] != 0:
                    s += frac[0]+"*pi/"+frac[1]
                else:
                    s += frac[0]+"*pi"
            bzx += s+"\n"
    
    return bzx