import pyzx as zx
from pyzx import simplify
from pyzx.utils import EdgeType, VertexType, FloatInt, FractionLike
from pyzx.circuit.gates import TargetMapper
from pyzx.graph import Graph
from pyzx.graph.base import BaseGraph, VT, ET
from fractions import Fraction
from typing import List, Dict, Any, Optional, Union
import numpy as np

class ZXParser(object):
    def __init__(self) -> None:
        self.qubit_count: int = 0
        self.vertices: List[self.V] = []
        
    """Class for parsing ZX source files into graph descriptions."""
    class V:
        def __init__(self, id, type, phase, qubit, column, neighbors):
            self.id: int = id
            self.type: str = type
            self.phase = phase
            self.qubit: int = qubit
            self.column: int = column
            self.neighbors = neighbors
            self.marked = False

        def __str__(self)-> str:
            return "Vertex(id:{}, type:{}, qubit:{}, col:{})".format(
                str(self.id),self.type,str(self.qubit),str(self.column))
        
        def __repr__(self) -> str:
            return str(self)
            

    def parse(self, s: str, strict: bool=True) -> List[V]:
        lines = s.splitlines()
        r = []
        # strip comments
        for l in lines:
            if l.find("//") != -1:
                t = l[0:l.find("//")].strip()
            else: t = l.strip()
            if t: r.append(t)

        for idx, line in enumerate(r):
            info = [x.strip().lower() for x in line.split(' ')]
            if info[0].startswith("i") or info[0].startswith("o"):
                alt = self.parse_IO("i", info) if info[0].startswith("i") else self.parse_IO("o", info)
                if alt == "ID_error": raise TypeError("Line \""+ line +"\" provides an invalid vertex ID")
                if alt == "QC_error": raise TypeError("Line \""+ line +"\" provides an invalid vertex coordinate")
                if alt == "Nbrs_error": raise TypeError("Line \""+ line +"\" provides an invalid neighbor information")
            elif info[0].startswith("z") or info[0].startswith("x") or info[0].startswith("h"):
                alt = self.parse_V("z", info) if info[0].startswith("z") else self.parse_V("x", info) if info[0].startswith("x") else self.parse_V("h", info)
                if alt == "ID_error": raise TypeError("Line \""+ line +"\" provides an invalid vertex ID")
                if alt == "QC_error": raise TypeError("Line \""+ line +"\" provides an invalid vertex coordinate")
                if alt == "Nbrs_error": raise TypeError("Line \""+ line +"\" provides an invalid neighbor information")
                if alt == "Phase_error": raise TypeError("Line \""+ line +"\" provides an invalid phase")
            else:
                raise TypeError("Line \""+ line +"\" does not start with a valid vertex type")
        return self.vertices

    def parse_IO(self, type: str, info: List[str]) -> Union[str,None]:
        id, qc, neighbors = self.parse_ID(info[0]), None, None

        if id == -1: return "ID_error"
        if info[1].startswith("(") and info[1].endswith(")"):
            qc = self.parse_coordinate(info[1])
            if qc == "QC_error": return "QC_error"
            # qubit: qc[0], column: qc[1]
            neighbors = self.parse_neighbors(info[2:])
        else:
            neighbors = self.parse_neighbors(info[1:])
        if neighbors == "Nbrs_error": return neighbors

        vertex = self.V(id, "i", 0, qc[0], qc[1], neighbors) if type == "i" else self.V(id, "o", 0, qc[0], qc[1], neighbors) 
        self.vertices.append(vertex)
    
    def parse_V(self, type: str, info: List[str]) -> Union[str,None]:
        id, qc, neighbors, phase = self.parse_ID(info[0]), None, None, 0
        
        if id == -1: return "IO_error"
        if not info[len(info)-1].startswith("s") and not info[len(info)-1].startswith("h"):
            phase = self.parse_phase(info.pop())
            if phase == "Phase_error": return "Phase_error"
        
        if info[1].startswith("(") and info[1].endswith(")"):
            qc = self.parse_coordinate(info[1])
            if qc == "QC_error": return "QC_error"
            # qubit: qc[0], column: qc[1]
            neighbors = self.parse_neighbors(info[2:])
        else:
            neighbors = self.parse_neighbors(info[1:])
        if neighbors == "Nbrs_error": return "Nbrs_error"

        vertex = self.V(id, "z", phase, qc[0], qc[1], neighbors) if type == "z" else self.V(id, "x", phase, qc[0], qc[1], neighbors) if type == "x" else self.V(id, "h", phase, qc[0], qc[1], neighbors) 
        self.vertices.append(vertex)
    
    def parse_phase(self, data: str) -> Union[Fraction, str]:
        if data == "0": return Fraction(0, 1)
        if data.startswith("pi"): data = "1*" + data
        if data.endswith("pi"): data = data + "/1"
        nums = data.split("*pi/")
        if len(nums) != 2: return "Phase_error"
        if nums[0].startswith("-"):
            if not nums[0][1:].isdigit(): return "Phase_error"
        if nums[1].startswith("-"):
            if not nums[1][1:].isdigit(): return "Phase_error"
        return Fraction(int(nums[0]), int(nums[1]))

    def parse_neighbors(self, data: List[str]) -> Union[str, List[tuple]]:
        neighbors = []
        for item in data:
            if item.startswith("s"):
                if not item[1:].isdigit(): return "Nbrs_error"
                else: neighbors.append(("s", int(item[1:])))
            elif item.startswith("h"):
                if not item[1:].isdigit(): return "Nbrs_error"
                else: neighbors.append(("h", int(item[1:])))
            else: return "Nbrs_error"
        return neighbors

    def parse_ID(self, data: str) -> int:
        if data[1:].isdigit() and not data[1:].startswith("-"): return int(data[1:])
        else: return -1
    
    def parse_coordinate(self, data: str) -> Union[str, tuple]:
        qc = [x.strip() for x in data.strip("(").strip(")").split(",")]
        # q: signed int, c: unsigned int
        if len(qc) != 2: return "QC_error"
        elif qc[0].startswith("-"):
            if qc[0][1:].isdigit() and qc[1].isdigit(): return (int(qc[0]), int(qc[1]))
        elif qc[0].isdigit() and qc[1].isdigit(): return (int(qc[0]), int(qc[1]))
        else: return "QC_error"


def zx(s: str) -> BaseGraph:
    p = ZXParser()
    return p.parse(s, strict=False)

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
        column = int(items[2])
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

def bzx_to_graph(bzx: str, backend:Optional[str]=None) -> BaseGraph:
    p = ZXParser()
    with open(bzx, 'r') as f:
        vertices = p.parse(f.read())
    
    g = Graph(backend)
    q_mapper: TargetMapper[VT] = TargetMapper()
    inputList, outputList = [], []
    id2label = {}
    # 
    for vertex in vertices:
        new_label = len(id2label)
        id2label[vertex.id] = new_label
        if (vertex.type == 'i' or vertex.type == 'o'): vtype = VertexType.BOUNDARY
        elif (vertex.type == 'h'): vtype = VertexType.H_BOX
        elif (vertex.type == 'z'): vtype = VertexType.Z
        elif (vertex.type == 'x'): vtype = VertexType.X
        else: raise TypeError(f"Unexpected vertex type ({vertex.type}).")
        
        v = g.add_vertex(vtype, vertex.qubit, vertex.column, vertex.phase)
        if (vertex.type == 'i'): inputList.append(v)
        elif (vertex.type == 'o'): outputList.append(v)
            
        vertex.marked = True
    
    # check miss edges
    for v in vertices:
        new_label_s = id2label[v.id]
        for n in v.neighbors:
            new_label_t = id2label[n[1]]
            if new_label_t not in g.neighbors(new_label_s):
                etype = EdgeType.SIMPLE if n[0] == 's' else EdgeType.HADAMARD       
                g.add_edge((new_label_s, new_label_t), etype)
    
    g.set_inputs(tuple(inputList))
    g.set_outputs(tuple(outputList))

    # Change H-box into Hadamard-edge
    remove = []
    for v in g.vertices():
        if g.type(v) == VertexType.H_BOX:
            if len(g.neighbors(v)) == 2:
                g.add_edge((list(g.neighbors(v))[0],list(g.neighbors(v))[1]), EdgeType.HADAMARD)
                remove.append(v)
    
    g.remove_vertices(remove)
    return g

def graph_to_bzx(g: BaseGraph[VT,ET]) -> str:
    bzx = ""
    edge_dict={1:"S", 2:"H"}
    vertex_dict={0:"B", 1:"Z", 2:"X", 3:"H"}
    for i in g.inputs():
        s = ""
        s += "I"+str(i)+" ("+str(g.qubit(i))+","+str(g.row(i))+") "
        for n in g.neighbors(i):
            if((i, n) in g.edges()):
                s += edge_dict[g.edge_type((i,n))]+str(n)+" "
            else:
                s += edge_dict[g.edge_type((n,i))]+str(n)+" "
        bzx += s+"\n"

    for o in g.outputs():
        s = ""
        s += "O"+str(o)+" ("+str(g.qubit(o))+","+str(g.row(o))+") "
        for n in g.neighbors(o):
            if((o, n) in g.edges()):
                s += edge_dict[g.edge_type((o,n))]+str(n)+" "
            else:
                s += edge_dict[g.edge_type((n,o))]+str(n)+" "
        bzx += s+"\n"

    for v in g.vertices():
        if v not in g.inputs() and v not in g.outputs():
            s = ""
            s += vertex_dict[g.type(v)]+str(v)+" ("+str(g.qubit(v))+","+str(g.row(v))+") "
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
