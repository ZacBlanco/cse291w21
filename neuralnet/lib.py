import os
import shutil
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

def generate_distinguishing_input(current_spec):
    '''Generates a new distinguishing input which is not a part of the existing
    specification

    Arguments:
        - current_spec (str): The specification for the problem including the
        existing constraints for the problem.

    Returns:
        - str: a string representing the distinguishing input not part of the
        current specification.

    '''
    raise NotImplementedError("sorry, input generation is not implemented")

def classify_outputs(input, programs, outputs, current_spec):
    '''This classifies program evaluation outputs and returns a dictionary
    mapping each program to the probability its output is correct.

    Arguments:
        - input (str): The input string used for each program evaluation
        - programs (list): The list of programs
        - outputs (list): a list of evaluated outputs for each program

    Returns:
        - (list) an ordered list of outputs by highest probability that the output was correct.
    '''
    raise NotImplementedError('output classification is not yet implemented')

def add_constraint_to_spec(current_spec, input, output):
    '''Adds a new input+output constraint to a given spec file

    Arguments:
        - current_spec (str): The path to the current spec file
        - input (str): The input part of the new constraint
        - output (str): The output part of the new constraint

    Returns:
        - None

    '''
    raise NotImplementedError("adding constraint to file not")

def rank_programs(programs, m):
    '''Ranks a given set of programs with probability that it is a "correct" program

    Arguments:
        - programs (list): The list of function expressions
        - m (int): The number of programs to return in ranked order.

    Returns:
        - list: A list of the top (m) most likely programs
    '''
    return programs[:m]

    raise NotImplementedError("program ranking not yet implemented")


def test():
    parser = ArgumentParser()
    parser.add_argument("-f", help="filename to generate solutions for", default='./euphony/benchmarks/string/test/phone-5-test.sl')
    parser.add_argument("-n", help="number of solutions to generate", default=1)
    args = parser.parse_args()

    sols = get_string_solutions(args.f, num_sols=int(args.n))
    # [ print(x) for x in sols ]
    inputs = [
        "+1 769-858-438",
        "+1 769-858-438",
        "+5 769-858-438",
        "+0 769-858-438",
        "+123 123-123-1230"
    ]
    import string_builder
    inputs_r = [string_builder.RString(inp) for inp in inputs]

    print_grid = lambda x: [print(item) for item in x]
    def distances(input_list, distfunc):
        dists = []
        for x in input_list:
            print(x)
            curr_dists = []
            for y in input_list:
                curr_dists.append(distfunc(x, y))
            dists.append(curr_dists)
        print_grid(dists)

    import textdistance
    distances(inputs_r, lambda x, y: x.distance(y))

    distances(inputs, lambda x, y: textdistance.hamming(x, y))

    for sol in sols:
        print("Testing program: {}".format(sol))
        for test_input in inputs:
            output = evaluate(sol, test_input)
            print('input: "{}" ---- output: "{}" ({})'.format(test_input, output, type(output)))

def main():
    parser = ArgumentParser()
    parser.add_argument("-f", help="filename to generate solutions for", default='./euphony/benchmarks/string/test/phone-5-test.sl')
    parser.add_argument("-n", help="number of solutions to generate", default=1)
    parser.add_argument("-i", help="number of iterations to perform", default=5)
    parser.add_argument("-m", help="the number M top programs to pick from program ranking. Must be <= -n", default=1)

    args = parser.parse_args()

    tmp_input_file = '.spec.tmp'
    try:
        os.remove(tmp_input_file)
    except FileNotFoundError as e:
        # OK to not exist yet
        pass
    shutil.copyfile(args.f, tmp_input_file)

    # initial input specification file
    input_spec = args.f
    # number of candidate programs to generate each iteration
    num_progs = args.n
    top_progs = args.m
    assert(top_progs <= num_progs)

    while True:
        # 1. generate up to N programs
        candidate_programs = get_string_solutions(input_spec, num_sols=num_progs)
        # 1a. rank the N programs and take the top M programs
        ranked_programs = rank_programs(candidate_programs, top_progs)
        # 2. we now need to generate a distinguishing candidate input, not part of the current input spec
        distinguishing_input = generate_distinguishing_input(input_spec)
        # 3. With the distinguishing input now available, execute all of the candidate programs on the input
        candidate_outputs = [evaluate(expression, distinguishing_input, synth_file=input_spec) for expression in ranked_programs]
        # 4. Execute the neural oracle to get the probability of the most likely correct output
        ranked_outputs = classify_outputs(distinguishing_input, candidate_programs, candidate_outputs, input_spec)
        # 5. Add the new input/output example to the specification
        input_spec = add_constraint_to_spec(input_spec, distinguishing_input, ranked_outputs[0])
        # Loop and continue to refine by generating more examples...




if __name__ == "__main__":
    main()