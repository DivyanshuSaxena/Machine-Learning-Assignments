import sys
import os

if sys.argv[3] == 0:
    os.system("python digit-classification.py {0} {1} {2}".format(sys.argv[1], sys.argv[2], sys.argv[4]))
else:
    os.system("python multidigit-classification.py {0} {1} {2}".format(sys.argv[1], sys.argv[2], sys.argv[4]))