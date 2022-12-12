import pyzx as zx
from pyzx import simplify
from pyzx.utils import EdgeType, VertexType, FloatInt, FractionLike
from pyzx.circuit.gates import TargetMapper
from pyzx.graph import Graph
from pyzx.graph.base import BaseGraph, VT, ET
from fractions import Fraction
import numpy as np

import argparse

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
        def __init__(self, id, qubit, next):
            self.id = id
            self.qubit = qubit
            self.next = next
    class V:
        def __init__(self, id, type, phase, qubit, neighbors):
            self.id = id
            self.type = type
            self.phase = phase
            self.qubit = qubit
            self.neighbors = neighbors
            self.marked = False
        
    inputs, outputs, vertices = [],[],[]
    for i in i_line:
        items = i.split(' ')
        input = IO(int(items[0][1:]),int(items[1]),(items[2][0], int(items[2][1:])))
        inputs.append(input)
    for o in o_line:
        items = o.split(' ')
        output = IO(int(items[0][1:]),int(items[1]),(items[2][0], int(items[2][1:])))
        outputs.append(output)
    for v in v_line:
        items = v.split(' ')
        id = int(items[0][1:])
        type = items[0][0]
        qubit = int(items[1])
        phase = 0
        neighbors = []
        if len(items) > 2:
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
        for item in items[2:]:
            et, nId = item[0], int(item[1:])
            neighbors.append((et, nId))
        
        vertex = V(id, type, phase, qubit, neighbors)
        vertices.append(vertex)
    return inputs, outputs, vertices

def read_zx_graph(f: str, of: str) -> BaseGraph[VT, ET]:
    # file read
    inputs, outputs, vertices = read_zx_file(f)
    
    g = Graph()
    q_mapper: TargetMapper[VT] = TargetMapper()
    inputList, outputList = [], []
    id2label = {}
    
    for input in inputs:
        new_label = len(id2label)
        id2label[input.id] = new_label
        v = g.add_vertex(VertexType.BOUNDARY,input.qubit,0)
        inputList.append(v)
        q_mapper.set_prev_vertex(input.qubit, v)
        q_mapper.set_next_row(input.qubit, 1)
        q_mapper.set_qubit(input.qubit, input.qubit)
    
    for ver in vertices:
        new_label = len(id2label)
        id2label[ver.id] = new_label
        if ver.type == 'H':
            v = g.add_vertex(VertexType.H_BOX, q_mapper.to_qubit(ver.qubit), q_mapper.next_row(ver.qubit), ver.phase)
            g.add_edge(g.edge(q_mapper.prev_vertex(ver.qubit), v), EdgeType.SIMPLE)
        elif ver.type == 'Z':
            v = g.add_vertex(VertexType.Z, q_mapper.to_qubit(ver.qubit), q_mapper.next_row(ver.qubit), ver.phase)
            g.add_edge(g.edge(q_mapper.prev_vertex(ver.qubit), v), EdgeType.SIMPLE)
        elif ver.type == 'X':
            v = g.add_vertex(VertexType.X, q_mapper.to_qubit(ver.qubit), q_mapper.next_row(ver.qubit), ver.phase)
            g.add_edge(g.edge(q_mapper.prev_vertex(ver.qubit), v), EdgeType.SIMPLE)
        q_mapper.set_prev_vertex(ver.qubit, v)
        q_mapper.advance_next_row(ver.qubit)
            
        ver.marked = True
    
    for l in q_mapper.labels():
        o = q_mapper.to_qubit(l)
        v = g.add_vertex(VertexType.BOUNDARY,o,q_mapper.max_row())
        outputList.append(v)
        u = q_mapper.prev_vertex(l)
        g.add_edge(g.edge(u,v))
    
    # check miss edges
    for v in vertices:
        new_label_s = id2label[v.id]
        for n in v.neighbors:
            new_label_t = id2label[n[1]]
            if new_label_t not in g.neighbors(new_label_s):
                s_row, t_row = g.row(new_label_s), g.row(new_label_t)
                s_qubit, t_qubit = g.qubit(new_label_s), g.qubit(new_label_t)
                if s_row < t_row:
                    # update s
                    g.set_row(new_label_s, t_row)
                    add_row = t_row - s_row
                    nodes = [node for node in g.qubits() if g.qubits()[node] == s_qubit and g.row(node) > s_row and node != new_label_s]
                    for node in nodes:
                        g.set_row(node, g.row(node)+add_row)
                else:
                    # update t
                    g.set_row(new_label_t, s_row)
                    add_row = s_row - t_row
                    nodes = [node for node in g.qubits() if g.qubits()[node] == t_qubit and g.row(node) > t_row and node != new_label_t]
                    for node in nodes:
                        g.set_row(node, g.row(node)+add_row)
                
                if n[0] == 'S':
                    g.add_edge((new_label_s, new_label_t))
                elif n[0] == 'H':
                    g.add_edge((new_label_s, new_label_t), EdgeType.HADAMARD)
                
    r = q_mapper.max_row()
    for output in outputList:
        g.set_row(output,r+1)
    
    
    g.set_inputs(tuple(inputList))
    g.set_outputs(tuple(outputList))
    fig = zx.draw_matplotlib(g, labels=True)
    zx.draw(g, labels=True)
    fig.savefig(of, bbox_inches ="tight")


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i','--input', help='Input the zx file.\n\n', required = True, metavar='zx_file')
    parser.add_argument('-o','--output', help='Input the pdf file.\n\n', required = True, metavar='pdf_file')
    
    args = parser.parse_args()

    read_zx_graph(args.input, args.output)

if __name__ == '__main__':
    main()
