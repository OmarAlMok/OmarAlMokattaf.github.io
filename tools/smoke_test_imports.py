import sys, importlib
modules = [('PyQt6','PyQt6'), ('numpy','numpy'), ('PIL','PIL')]
print('Python executable:', sys.executable)
ok = True
for label, mod in modules:
    try:
        m = importlib.import_module(mod)
        v = getattr(m, '__version__', getattr(m, 'VERSION', 'unknown'))
        print(f'{label}: OK ({v})')
    except Exception as e:
        print(f'{label}: FAIL: {e}')
        ok = False
if not ok:
    sys.exit(2)
print('SMOKE_TEST: all imports OK')
