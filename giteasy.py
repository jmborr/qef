#!/usr/bin/env python
# PYTHON_ARGCOMPLETE_OK

from __future__ import print_function

import os
import re
import subprocess
import argparse
import argcomplete
import sys
from pdb import set_trace as tr

def retrieveBranch():
    proc = subprocess.Popen(['git','branch'],stdout=subprocess.PIPE)
    for line in proc.stdout:
        if '*' in line:
            return line.split()[1]

def suggestAction(errorMessage):
    """Suggest corrective action based on the output after running the Git commands"""
    branch = retrieveBranch()
    hint={r"error: The branch.*is not fully merged[.\n]*If you are sure you want to delete it":"Checkout again to the branch you want do delete, and this time include flag --force"}
    buffer = "TAKE ACTION: "
    for errorPattern in hint.keys():
        if re.search(errorPattern, errorMessage):
            print( buffer + "You are in the "+branch+" branch. "+hint[errorPattern])
            break

parser = argparse.ArgumentParser(description='Git commands for Mantid')
parser.add_argument('--create', type=str, default='', help="Create a branch. The name should be <git-issue>_descriptive_text")
parser.add_argument('--pushToOrigin', action='store_true', help="push branch to origin")
parser.add_argument('--force', action='store_true', help="in combination with pushToOrigin to overwrite the remote branch.")
parser.add_argument('--update', action='store_true', help="update branch with a master rebase.")
parser.add_argument('--delete', action='store_true', help="delete current local branch")
parser.add_argument('--updateMaster', action='store_true', help="update local master branch.")
parser.add_argument('--dryrun', action='store_true', help="print commands to be run, but don't run them.")
argcomplete.autocomplete(parser)
args=parser.parse_args()

script = """#!/bin/bash
# COMMANDS to run
"""

if args.create:
    branch = args.create
    script += """git checkout master
git fetch -p
git branch --no-track {0} origin/master
git checkout {0}
""".format(branch)

if args.pushToOrigin:
    branch = retrieveBranch()
    if args.force:
        branch = "+" + branch
    if not branch or 'master' in branch:
        raise IOError("branch not retrieved")
    script += """git push origin {0}""".format(branch)
    
if args.update:
    script += """git fetch -p
git rebase -v origin/master"""

if args.delete:
    branch = retrieveBranch()
    script += """git checkout master
git branch -D {0}
""".format(branch)

if args.updateMaster:
    branch = retrieveBranch()
    if branch == 'master':
        script += """git fetch -p
git pull --rebase
"""

print(script)  #inform of the commands to be run
output=None   #collect output from the commands
if not args.dryrun:
    try:
        output = subprocess.check_output(script, shell=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as errorObject:
        output = errorObject.output
    print("OUTPUT from the commands\n"+output)
    suggestAction(output)
