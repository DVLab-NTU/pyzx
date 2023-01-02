import pyzx as zx
from pyzx import simplify
from pyzx.utils import EdgeType, VertexType, FloatInt, FractionLike
from pyzx.circuit.gates import TargetMapper
from pyzx.graph import Graph
from pyzx.graph.base import BaseGraph, VT, ET
from pyzx.graph.bzxparser import bzx_to_graph
from fractions import Fraction
import numpy as np

import argparse

def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i','--input', help='Input the zx file.\n\n', required = True, metavar='zx_file')
    parser.add_argument('-o','--output', help='Input the pdf file.\n\n', required = True, metavar='pdf_file')
    
    args = parser.parse_args()

    g = bzx_to_graph(args.input)

    fig = zx.draw_matplotlib(g, labels=True)
    zx.draw(g, labels=True)
    fig.savefig(args.output, bbox_inches ="tight")

if __name__ == '__main__':
    main()
