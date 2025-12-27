import sys
try:
    import numpy as np
    print('NUMPY FILE:', getattr(np, '__file__', 'builtin'))
    print('NUMPY VERSION:', np.__version__)
except Exception as e:
    print('NUMPY IMPORT ERROR:')
    import traceback
    traceback.print_exc()
    print('CWD:', sys.path[0])
    print('SYS.PATH:')
    for p in sys.path:
        print(' -', p)
