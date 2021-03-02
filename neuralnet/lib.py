import os
import sys

import benchmarks
import options

from argparse import ArgumentParser

from parsers import parser
from exprs.evaluation import EvaluationContext
from exprs.exprs import Value
from exprs.exprtypes import StringType
from exprs.evaluation import evaluate_expression_raw

def get_string_solutions(input_file, num_sols=1):
    '''Returns a set of FunctionExpressions representing solutions from the
    euphony synthesizer.

    The outputs of this function can be then evaluated and/or fed into a neural
    network for ranking

    Arguments:
        - input_file (str): The file containing the logic set and constraints
          for euphony to synthesize
        - num_sols (int): The number of solutions that the solver should
          generate

    Returns:
        - list: <- FunctionExpression (see euphony.exprs.FunctionExpression)
    '''
    if not os.path.exists(input_file):
        print("Input file does not exist: {}".format(input_file))
        sys.exit(1)
    benchmark_files = input_file
    sphog_file = "./euphony/phog_str"
    rcfg_file = ''
    options.noindis = True
    options.inc = False
    options.allex = False
    options.stat = False
    options.noheuristic = False
    options.numsols = num_sols
    options.rewrite = False
    if not os.path.exists(sphog_file):
        print("Can't find sphog file! {}".format(sphog_file))
        sys.exit(1)
    file_sexp = parser.sexpFromFile(input_file)
    # result is a list of lists
    sols = benchmarks.make_solver(file_sexp, sphog_file, rcfg_file, options=options)
    return [sol for innersols in sols for sol in innersols]

def evaluate(expr, input, synth_file='./euphony/benchmarks/string/test/phone-5-test.sl'):
    '''Evaluate an expression for a given input

    Arguments:
        - expr(FunctionExpression): The expression to evaluate
        - input(str): the input to the given expression.
        - synth_file(str): path to a synthesis file defining the available
          available grammar. This file doesn't need to correspond to the same
          file that the expression was generated from. It just needs to have the
          same grammar definition. Constraints are not used in any way. It's
          required in order to load the proper logic into the expression
          evaluator. This is set to a file containing string logic by default.

    Returns:
        - (str): output from evaluating expression against given input
    '''
    eval_ctx = EvaluationContext()
    val_map = (Value(input, value_type=StringType()),)
    eval_ctx.set_valuation_map(val_map)
    # print('attempting to eval: {}'.format(expr))
    # print('eval ctx: {} -- {}'.format(val_map, type(val_map)))

    # This is pretty flow since it needs to re-read the synthesis file to know
    # the grammar that it needs to parse. Personally, I am not so concerned
    # about performance, but rather just getting it to work in the first place.
    file_sexp = parser.sexpFromFile(synth_file)
    benchmark_tuple = parser.extract_benchmark(file_sexp)
    (
            theories,
            syn_ctx,
            synth_instantiator,
            macro_instantiator,
            uf_instantiator,
            constraints,
            grammar_map,
            forall_vars_map,
            default_grammar_sfs
            ) = benchmark_tuple
    # We should only have one grammar, so just grab the only key to this map.
    # The type of the object is a SynthFun
    synth_fun = list(grammar_map.keys())[0]
    # Pass this into the interpretation along with the expression we're
    # attempting to evaluate, or else we'll get an UnboundLetException
    eval_ctx.set_interpretation(synth_fun, expr)
    return evaluate_expression_raw(expr, eval_ctx)

def main():
    parser = ArgumentParser()
    parser.add_argument("-f", help="filename to generate solutions for", default='./euphony/benchmarks/string/test/phone-5-test.sl')
    parser.add_argument("-n", help="number of solutions to generate", default=1)
    args = parser.parse_args()

    sols = get_string_solutions(args.f, num_sols=int(args.n))
    [ print(x) for x in sols ]
    inputs = [
        "+1 769-858-438",
        "+1 769-858-438",
        "+5 769-858-438",
        "+0 769-858-438",
    ]
    for sol in sols:
        print("Testing program: {}".format(sol))
        for test_input in inputs:
            output = evaluate(sol, test_input)
            print('input: "{}" ---- output: "{}" ({})'.format(test_input, output, type(output)))


if __name__ == "__main__":
    main()