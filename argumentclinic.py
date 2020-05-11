#MIT License
#
#Copyright (c) 2019 Douwe Osinga
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
import argparse
import inspect


def not_empty(value):
    return value != inspect.Signature.empty

def make_parser(func,defaultparse=eval):
    description = inspect.getdoc(func)
    sig = inspect.signature(func)

    parser = argparse.ArgumentParser(description=description,
                            formatter_class=argparse.RawDescriptionHelpFormatter)

    for par_name, par_info in sig.parameters.items():
        kwargs = {}
        arg_name = par_name
        if not_empty(par_info.annotation):
            kwargs['type'] = par_info.annotation
        else:
            kwargs['type'] = defaultparse
        if not_empty(par_info.default):
            kwargs['default'] = par_info.default
            arg_name = '--' + arg_name
            #kwargs['nargs'] = '?'
        elif par_info.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs['required'] = True
        #if par_info.kind == inspect.Parameter.KEYWORD_ONLY:
        #    arg_name = '--' + arg_name
        parser.add_argument(arg_name, **kwargs)
    return parser

def entrypoint(func,defaultparse=eval):
    parser = make_parser(func,defaultparse=defaultparse)
    args = parser.parse_args()
    return func(**vars(args))
