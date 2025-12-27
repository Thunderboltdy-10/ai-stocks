import sys, os
print('EXE:', sys.executable)
print('CWD:', os.getcwd())
print('---SYS.PATH---')
for p in sys.path:
    print(p)
