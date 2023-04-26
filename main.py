#!/usr/bin/env python
import sys

from getopt import getopt
from Agent import SmartAgent

opts, args = getopt(sys.argv[1:], "t:e:hr", ["train=","evaluate=", "help", "resume"])

resume, train, evaluate = False, False, False
train_t, eval_t = 0, 0
brain_path = ""

for o, a in opts:
    if o in ('-t', '--train'):
        train = True
        train_t = int(a)
    elif o in ('-e', '--evaluate'):
        evaluate = True
        eval_t = int(a)
    elif o in ('-h', '--help'):
        print('Help')
    elif o in ('-r', '--resume'):
        resume = True
    else:
        print("Error! Unknown Option")

if len(args) == 1:
    brain_path = args[0]
else:
    print(f"Expecting 1 position argument, gor {len(args)}")

agent = SmartAgent(game_name='ALE/Pong-v5', resume=resume, brain_path=brain_path)

if(train):
    agent.train(train_t)
    agent.save(brain_path)
agent.save(agent.brain_path)

if(evaluate):
    agent.evaluate(eval_t)

