# File reporter.py

## Class Reporter

This class should be used to print instead of print()
It should be imported as follows:
from urnai.utils.reporter import Reporter as rp
or
from utils.reporter import Reporter as rp
And then the function report should be called to print:
rp.report("My message")
If the message is a debug one, a level different from 0
should be used:
rp.report("Debug message", 2)

## Method report()

- Arguments: message, verbosity_lvl, end

## Method save()

- Arguments: persist_path

## Method load()

- Arguments: persist_path
