# -*- coding: utf-8 -*-
from urnai.runner.parserbuilder import ParserBuilder
import os

def main():
    os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

    parser = ParserBuilder.DefaultParser()
    args = parser.parse_args()
    try:
        args.func(args)
    except AttributeError as ae:
        if "'Namespace' object has no attribute 'func'" in str(ae):
            parser.print_help()
        else:
            raise
