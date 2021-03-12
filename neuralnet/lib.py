import os
import shutil
import sys
import random

import numpy as np

import benchmarks
import options
import string_builder

from argparse import ArgumentParser

from parsers import parser
from exprs.evaluation import EvaluationContext
from exprs.exprs import Value
from exprs.exprtypes import StringType
from exprs.evaluation import evaluate_expression_raw
import classifier
import rank

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
    #print(constraints)
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
    file_sexp = parser.sexpFromFile(current_spec)
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
    inputs = []
    for constraint in constraints:
        inputs.append(constraint.children[0].children[0].value_object.value_object)
    #print(inputs)
    import string_builder
    inputs_r = [string_builder.RString(inp) for inp in inputs]
    #print(inputs_r)
    print_grid = lambda x: [print(item) for item in x]
    def distances(input_list, distfunc):
        dists = []
        for x in input_list:
            # print(x)
            # print(x.groupstr)
            curr_dists = []
            for y in input_list:
                curr_dists.append(distfunc(x, y))
            dists.append(curr_dists)
        # print_grid(dists)
        return dists


    # find the metric which varies the least between all the examples
    char_dist  = distances(inputs_r, lambda x, y: x.orig_distance(y))
    class_dist = distances(inputs_r, lambda x, y: x.class_distance(y))
    group_dist = distances(inputs_r, lambda x, y: x.group_distance(y))
    vals = {
        'char': char_dist,
        'class': class_dist,
        'group': group_dist
    }
    stats = {}
    for key in vals:
        val = vals[key]
        tril = np.tril(np.array(val))
        mu = tril.mean()
        std = tril.std()
        stats[key] = string_builder.TextStats(mu, std)
        # print("{}: mean: {}; std: {}".format(key, mu, std))
    stat_set = string_builder.EditStatSet(stats['char'], stats['class'], stats['group'])
    mutated = random.sample(inputs_r, 1)[0].generate_mutation(stat_set)
    return mutated

def classify_outputs(input, outputs):
    '''This classifies program evaluation outputs and returns a dictionary
    mapping each program to the probability its output is correct.

    Arguments:
        - input (str): The input string used for each program evaluation
        - outputs (list): a list of evaluated outputs for each program

    Returns:
        - (list) an ordered list of outputs by highest probability that the output was correct.
    '''

    clist=[]
    clist=classifier.test(input,outputs)
    o=outputs[clist.index(max(clist))]
    d=dict(zip(outputs,clist))
    #print(d)
    a1 = sorted(d.items(),key = lambda x:x[1],reverse = True)
    #print(a1)
    sample_key_list = []

    for item in a1:
        sample_key_list.append(item[0])
    #print(sample_key_list)
    return sample_key_list
    #raise NotImplementedError('output classification is not yet implemented')

def rank_programs(current_spec, programs, m):
    '''Ranks a given set of programs with probability that it is a "correct" program

    Arguments:
        - programs (list): The list of function expressions
        - m (int): The number of programs to return in ranked order.

    Returns:
        - list: A list of the top (m) most likely programs
    '''
    specs=get_constraints(current_spec)
    #print(specs)
    #print(programs)
    inputs=[]
    for i in specs:
        inputs.append(i)
    ps=[]
    for i in programs:
        ps.append(str(i))
    rlist=rank.test(inputs,ps)
    # print(rlist)

    d=dict(zip(programs,rlist))
    #print(d)
    a1 = sorted(d.items(),key = lambda x:x[1],reverse = True)
    #print(a1)
    sample_key_list = []

    for item in a1:
        sample_key_list.append(item[0])

    return sample_key_list[:m]
    #return programs[:m]

    raise NotImplementedError("program ranking not yet implemented")

def add_constraint_to_spec(current_spec, inputs, outputs):
    '''Adds a new input+output constraint to a given spec file

    Arguments:
        - current_spec (str): The path to the current spec file
        - inputs (str): The input part of the new constraint
        - outputs (str): The output part of the new constraint

    Returns:
        - None

    '''

    current_spec_lines = []
    with open(current_spec, 'r') as f:
        current_spec_lines = f.readlines()

        constraint_begin = None
        for i in range(len(current_spec_lines)):
            if '(constraint (= (f "' in current_spec_lines[i]:
                constraint_begin = i
                break
        if constraint_begin == None:
            raise Exception("Couldn't find proper location to put new constraint in spec")
        current_spec_lines.insert(i + 1, '(constraint (= (f "{}") "{}"))\n'.format(inputs, outputs))

    with open(current_spec, 'w') as f:
        f.writelines(current_spec_lines)
def get_constraints(current_spec):
    '''Generates a new distinguishing input which is not a part of the existing
    specification

    Arguments:
        - current_spec (str): The specification for the problem including the
        existing constraints for the problem.

    Returns:
        - list: contains all the specifications

    '''
    file_sexp = parser.sexpFromFile(current_spec)
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
    inputs = []
    vals= []
    for constraint in constraints:
        inputs.append(constraint.children[0].children[0].value_object.value_object)
        vals.append(constraint.children[1].value_object.value_object)
    result=[]
    for i in range(0,len(inputs)):
        result.append(inputs[i]+'; '+vals[i])
    #print(result)
    return result

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
            # print(x.groupstr)
            curr_dists = []
            for y in input_list:
                curr_dists.append(distfunc(x, y))
            dists.append(curr_dists)
        print_grid(dists)
        return dists


    # find the metric which varies the least between all the examples
    char_dist  = distances(inputs_r, lambda x, y: x.orig_distance(y))
    class_dist = distances(inputs_r, lambda x, y: x.class_distance(y))
    group_dist = distances(inputs_r, lambda x, y: x.group_distance(y))
    vals = {
        'char': char_dist,
        'class': class_dist,
        'group': group_dist
    }
    stats = {}
    for key in vals:
        val = vals[key]
        tril = np.tril(np.array(val))
        mu = tril.mean()
        std = tril.std()
        stats[key] = string_builder.TextStats(mu, std)
        print("{}: mean: {}; std: {}".format(key, mu, std))
    stat_set = string_builder.EditStatSet(stats['char'], stats['class'], stats['group'])
    print("MUTATED: {}".format(random.sample(inputs_r, 1)[0].generate_mutation(stat_set)))


    # once finding the least-varying metric, choose the type of modification
    # to make to an existing example.
    # Types of modifications
    # 1. single character addition/deletion/edit (within existing group)
    # 2. character class addition/edit/insert
    # 3. group addition/deletion/edit

    for sol in sols:
        print("Testing program: {}".format(sol))
        for test_input in inputs:
            output = evaluate(sol, test_input)
            print('input: "{}" ---- output: "{}" ({})'.format(test_input, output, type(output)))

def main():
    # test()
    parser = ArgumentParser()
    parser.add_argument("-f", help="filename to generate solutions for", default='./euphony/benchmarks/string/test/phone-5-test.sl')
    parser.add_argument("-n", help="number of solutions to generate", default=3, type=int)
    parser.add_argument("-i", help="number of iterations to perform", default=5)
    parser.add_argument("-m", help="the number M top programs to pick from program ranking. Must be <= -n", default=3, type=int)

    args = parser.parse_args()

    tmp_input_file = '.spec.tmp'
    try:
        os.remove(tmp_input_file)
    except FileNotFoundError as e:
        # OK to not exist yet
        pass
    shutil.copyfile(args.f, tmp_input_file)

    # initial input specification file
    input_spec = tmp_input_file
    # number of candidate programs to generate each iteration
    num_progs = args.n
    top_progs = args.m
    assert(top_progs <= num_progs)

    while True:
        # 1. generate up to N programs
        candidate_programs = get_string_solutions(input_spec, num_sols=num_progs)
        # 1a. rank the N programs and take the top M programs
        ranked_programs = rank_programs(input_spec,candidate_programs, top_progs)

        # 2. we now need to generate a distinguishing candidate input, not part of the current input spec
        distinguishing_input = generate_distinguishing_input(input_spec)
        print("new distinguishing input: {}".format(distinguishing_input))
        # 3. With the distinguishing input now available, execute all of the candidate programs on the input
        candidate_outputs = [evaluate(expression, distinguishing_input, synth_file=input_spec) for expression in ranked_programs]
        # print(candidate_outputs)
        # 4. Execute the neural oracle to get the probability of the most likely correct output
        ranked_outputs = classify_outputs(distinguishing_input, candidate_outputs)
        print(ranked_outputs)
        # 5. Add the new input/output example to the specification
        add_constraint_to_spec(input_spec, distinguishing_input, ranked_outputs[0])
        # Loop and continue to refine by generating more examples...




if __name__ == "__main__":
    main()
