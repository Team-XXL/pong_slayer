#!/usr/bin/env python
import sys

from getopt import getopt

opts, args = getopt(sys.argv[1:], "t:e:h", ["train=","evaluate=", "help"])

train, evaluate = False, False
train_t, eval_t = 0, 0

for o, a in opts:
    print(o)
    if o in ('-t', '--train'):
        train = True
        train_t = a
    elif o in ('-e', '--evaluate'):
        evaluate = True
        eval_t = a
    elif o in ('-h', 'help'):
        print('Help')
    else:
        print("Error")

# agent = SmartAgent()
# if(train):
#     agent.train(train_t)
#     agent.save()
# if(evaluate):
#     agent.evaluate(eval_t)

print(train, evaluate)