"""Quick sanity check before working on Member 2's section: run `python check_env.py`."""
import importlib, sys
ok = True
for mod in ["sklearn", "imblearn", "xgboost", "pandas", "numpy", "matplotlib", "seaborn"]:
    try:
        m = importlib.import_module(mod); print(f"OK   {mod} {getattr(m, '__version__', '')}")
    except ImportError:
        ok = False; print(f"MISSING {mod}  ->  pip install -r requirements.txt")
if not ok:
    sys.exit(1)

from shared_setup import load_and_split, build_preprocessor, CV, SEED
X_train, X_test, y_train, y_test = load_and_split("heart.csv")
assert X_train.shape == (734, 11) and X_test.shape == (184, 11), "split differs from Member 1's"
build_preprocessor("improved").fit(X_train)
print(f"OK   split train={X_train.shape} test={X_test.shape}, seed={SEED}, folds={CV.get_n_splits()}")
print("Environment ready.")
