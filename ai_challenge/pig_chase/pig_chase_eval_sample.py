# Copyright (c) 2017 Microsoft Corporation.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
#  rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===================================================================================================================

from common import ENV_AGENT_NAMES
from evaluation import PigChaseEvaluator
from environment import PigChaseTopDownStateBuilder, PigChaseSymbolicStateBuilder
from malmopy.agent import RandomAgent
from agent import FocusedAgent, QLearnHighAgent, QLearnLowAgent
from common import ENV_AGENT_NAMES, ENV_TARGET_NAMES
import numpy as np

if __name__ == '__main__':
    # Warn for Agent name !!!

	clients = [('127.0.0.1', 10000), ('127.0.0.1', 10001)]
	agent_100k = QLearnHighAgent(ENV_AGENT_NAMES[0], ENV_AGENT_NAMES[1], ENV_TARGET_NAMES[0], 0.3, 0.9, 0.05)
	agent_500k = QLearnHighAgent(ENV_AGENT_NAMES[1], ENV_AGENT_NAMES[0], ENV_TARGET_NAMES[0], 0.3, 0.9, 0.05)

	agent_100k.Q = np.load('Q_files/Q110k.npy').item()
	print "Size 100k: ", len(agent_100k.Q.keys())
	agent_500k.Q = np.load('Q_files/Q270k.npy').item()
	print "Size 500k: ", len(agent_500k.Q.keys())

	eval = PigChaseEvaluator(clients, agent_100k, agent_500k, PigChaseSymbolicStateBuilder())
	eval.run()

	eval.save('My Exp 1', 'results/pig_chase_results.json')