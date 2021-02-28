import os
import sys

# [ print(x) for x in sys.path ]
from argparse import ArgumentParser
from euphony.bin import benchmarks
from euphony.bin.parsers import parser
from euphony.bin import options

def get_string_solutions(input_file, num_sols=1):
    '''Returns a set of FunctionExpressions representing solutions from the euphony
    synthesizer.

    The outputs of this function can be then evaluated and/or fed into a neural network for ranking

    Arguments:
        - input_file (str): The file containing the logic set and constraints for euphony to synthesize
        - num_sols (int): The number of solutions that the solver should generate

    Returns:
        - list: <- FunctionExpression (see euphony.exprs.FunctionExpression)
    '''
    if not os.path.exists(input_file):
        print("Input file does not exist: {}".format(input_file))
        sys.exit(1)
    benchmark_files = input_file
    # phog_file = "./euphony/bin/phog_str"
    sphog_file = "./euphony/phog_str"
    rcfg_file = ''
    options.noindis = True
    options.inc = False
    options.allex = False
    options.stat = False
    options.noheuristic = False
    options.numsols = num_sols
    if not os.path.exists(sphog_file):
        print("Can't find sphog file! {}".format(sphog_file))
        sys.exit(1)
    file_sexp = parser.sexpFromFile(input_file)
    # result is a list of lists
    sols = benchmarks.make_solver(file_sexp, sphog_file, rcfg_file, options=options)
    return [sol for innersols in sols for sol in innersols]

def main():
    parser = ArgumentParser()
    parser.add_argument("-f", help="filename to generate solutions for", default='./euphony/benchmarks/string/test/phone-5-test.sl')
    parser.add_argument("-n", help="number of solutions to generate", default=1)
    args = parser.parse_args()

    [ print(x) for x in get_string_solutions(args.f, num_sols=int(args.n)) ]

if __name__ == "__main__":
    main()