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
from semantics import semantics_core
from semantics import semantics_types
from core import synthesis_context
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
    print(result)
    return result

def get_programs(current_spec):
    '
    file_sexp = parser.sexpFromFile(current_spec)
    

    core_instantiator = semantics_core.CoreInstantiator()
    theory_instantiators = [parser.get_theory_instantiator(theory) for theory in parser._known_theories]

    macro_instantiator = semantics_core.MacroInstantiator()
    uf_instantiator = semantics_core.UninterpretedFunctionInstantiator()
    synth_instantiator = semantics_core.SynthFunctionInstantiator()

    syn_ctx = synthesis_context.SynthesisContext(
            core_instantiator,
            *theory_instantiators,
            macro_instantiator,
            uf_instantiator,
            synth_instantiator)
    syn_ctx.set_macro_instantiator(macro_instantiator)

    defs, _ = parser.filter_sexp_for('define-fun', file_sexp)
    if defs is None: 
        defs = []
    #print(defs)
    result=""
    for [name, args_data, ret_type_data, interpretation] in defs:
        ((arg_vars, arg_types, arg_var_map), return_type) = parser._process_function_defintion(args_data, ret_type_data)
        expr = parser.sexp_to_expr(interpretation, syn_ctx, arg_var_map)
        result=expr
    
    return result

if __name__ == "__main__":
    
    file_dir="euphony/benchmarks/string/train"
    fs=[]
    for root, dirs, files in os.walk(file_dir, topdown=False):
        fs=files
    #print(fs)
    f=open("/Users/chunyuxia/Desktop/program ranking/n_spec.txt","w")
    o=open("/Users/chunyuxia/Desktop/program ranking/n_programs.txt","w")
    for i in fs:
        rs=[]
        ff=""
        if i.startswith('phone') or i.startswith('lastname') or i.startswith('firstname') or i.startswith('dr') or i.startswith('bikes') or i.startswith('change-neg'):
            ff=i
            sp="euphony/benchmarks/string/train"
        
            rs=get_constraints(sp+i)
            
            for line in rs:
                f.write(line+'\n')
            
            f.write('\n')
            p=get_programs(sp+i)
            print(p)
            o.write(str(p)+'\n')

        else:
            continue
        
    f.close()
    """
    list01 = []
    for i in open("/Users/chunyuxia/Desktop/program ranking/examples.txt"):
        print(i)
        if i in list01:
            continue
        list01.append(i)
    with open('/Users/chunyuxia/Desktop/program ranking/newdata.txt', 'w') as handle:
        handle.writelines(list01)
    """
    #print(get_programs('/Users/chunyuxia/Desktop/291main/cse291w21/euphony/benchmarks/string/train/phone-8-long-repeat.sl'))

###test for ranking


"""
a = open('n_spec.txt','r')

f1 = a.read().split('\n\n')

specs=[]
for i in f1:
    l=i.split('\n')
    specs.append(l)


b = open('n_programs.txt','r')

f2 = b.read().splitlines()

programs=[]
for i in f2:
    l=i.split('\n')
    programs.append(l)

specs.pop()
avg=0
l=[]
for i in range(0,len(specs)):
    a=test(specs[i],programs[i])[0]
    print(a)
    l.append(a)
    avg=avg+a
#print(max(l))
avg=avg/len(specs)
print(avg)
print((avg-min(l))/(max(l)-min(l)))
"""
#test for classifier
"""
a=[]
b=[]
for i in open("/Users/chunyuxia/Desktop/program ranking/examples.txt"):
    ll = i.split("; ")
    a.append(ll[0])
    b.append(ll[1])
avg=0
l=[]
for i in range(0,len(a)):
    l.append(test(a[i],b[i])[0])
    avg=avg+test(a[i],b[i])[0]
print(max(l))
avg=avg/len(a)
print(avg)
print((avg-min(l))/(max(l)-min(l)))
"""