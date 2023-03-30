import pyzx as zx
from pyzx import simplify
from pyzx.utils import EdgeType, VertexType, FloatInt, FractionLike
from pyzx.circuit.gates import TargetMapper
from pyzx.graph import Graph
from pyzx.graph.base import BaseGraph, VT, ET
from fractions import Fraction
from typing import List, Dict, Any, Optional, Union
from pyzx.hsimplify import hadamard_simp
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
        id, qc, neighbors, phase = self.parse_ID(info[0]), None, None, 0 if type != 'h' else Fraction(1, 1)
        if id == -1: return "IO_error"
        # if len(info) == 2: phase = 0 if type != 'h' else Fraction(1, 1) # NOTE - removed: redundant assignment
        
        if not info[len(info)-1].startswith("s") and not info[len(info)-1].startswith("h") and len(info) > 2:
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


def zx_to_graph(zx: str, backend:Optional[str]=None) -> BaseGraph:
    p = ZXParser()
    with open(zx, 'r') as f:
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
    # NOTE - the old removal method is not complete and may lead to error when multiple Hs occurs consecutively.
    #        I've replaced it with PyZX's native H-box remover
    removed = hadamard_simp(g, None, True)

    return g

def graph_to_zx(g: BaseGraph[VT,ET]) -> str:
    zx = ""
    edge_dict={1:"S", 2:"H"}
    vertex_dict={0:"B", 1:"Z", 2:"X", 3:"H"}
    
    zx += "// Inputs \n"
    for i in g.inputs():
        s = ""
        s += "I"+str(i)+" ("+str(g.qubit(i))+","+str(g.row(i))+") "
        for n in g.neighbors(i):
            if((i, n) in g.edges()):
                s += edge_dict[g.edge_type((i,n))]+str(n)+" "
            else:
                s += edge_dict[g.edge_type((n,i))]+str(n)+" "
        zx += s+"\n"

    zx += "// Outputs \n"
    for o in g.outputs():
        s = ""
        s += "O"+str(o)+" ("+str(g.qubit(o))+","+str(g.row(o))+") "
        for n in g.neighbors(o):
            if((o, n) in g.edges()):
                s += edge_dict[g.edge_type((o,n))]+str(n)+" "
            else:
                s += edge_dict[g.edge_type((n,o))]+str(n)+" "
        zx += s+"\n"

    zx += "// Non-boundary \n"
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
            zx += s+"\n"
    
    return zx
