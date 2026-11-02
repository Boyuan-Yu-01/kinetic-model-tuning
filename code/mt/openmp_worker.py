"""Private persistent worker protocol. Never import this module as an API."""
import os
import pickle
import sys
import traceback

# Reserve protocol stdout, including against native libraries writing fd 1.
protocol = os.fdopen(os.dup(1), 'w', buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr
for line in sys.stdin:
    path = line.rstrip('\n')
    try:
        with open(path, 'rb') as stream:
            function, args, kwargs = pickle.load(stream)
        payload = pickle.dumps((True, function(*args, **kwargs)), protocol=pickle.HIGHEST_PROTOCOL)
    except BaseException:
        payload = pickle.dumps((False, traceback.format_exc()))
    finally:
        # Do not retain the previous task's mechanism or arrays while idle.
        function = args = kwargs = None
    try:
        with open(path + '.result', 'wb') as stream:
            stream.write(payload)
        del payload
        protocol.write('done\n')
    except BaseException:
        traceback.print_exc()
        break
