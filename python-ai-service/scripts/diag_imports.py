import sys, importlib, pkgutil
print('\nPYTHON VERSION:', sys.version)
print('\nSITELIST (first 8 entries):')
for p in sys.path[:8]:
    print(' -', p)
print('\nfind_spec(google)=', importlib.util.find_spec('google'))
print('find_spec(google.protobuf)=', importlib.util.find_spec('google.protobuf'))
print('\nSome installed packages matching google/protobuf/yfinance:')
for m in pkgutil.iter_modules():
    if m.name.startswith('google') or m.name.startswith('protobuf') or m.name.startswith('yfinance'):
        print(' *', m.name, 'pkg?', m.ispkg)

# Try importing them and show errors
for name in ('google', 'google.protobuf', 'yfinance'):
    try:
        mod = importlib.import_module(name)
        print(f"\nImported {name}:", getattr(mod, '__file__', getattr(mod, '__path__', str(mod))))
    except Exception as e:
        print(f"\nFailed to import {name}: {e!r}")
