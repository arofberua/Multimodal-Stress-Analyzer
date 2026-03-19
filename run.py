"""
Launcher script for AI-Microexpression-Analyzer.
Fixes relative imports by injecting the package into sys.path.
"""
import sys
import os
import importlib
import types

# The package folder (this file's directory)
pkg_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(pkg_dir)

# Create a synthetic package so relative imports work
pkg_name = "microexpr"
package = types.ModuleType(pkg_name)
package.__path__ = [pkg_dir]
package.__package__ = pkg_name
sys.modules[pkg_name] = package

sys.path.insert(0, pkg_dir)
sys.path.insert(0, parent_dir)

# Patch relative imports by loading each module manually
def load_submodule(name, filepath):
    import importlib.util
    spec = importlib.util.spec_from_file_location(f"{pkg_name}.{name}", filepath)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = pkg_name
    sys.modules[f"{pkg_name}.{name}"] = mod
    spec.loader.exec_module(mod)
    return mod

base = pkg_dir
load_submodule("data_logger", os.path.join(base, "data_logger.py"))
load_submodule("dashboard", os.path.join(base, "dashboard.py"))
load_submodule("face_mesh_module", os.path.join(base, "face_mesh_module.py"))
load_submodule("feature_engineering", os.path.join(base, "feature_engineering.py"))
load_submodule("stress_model", os.path.join(base, "stress_model.py"))

main_mod = load_submodule("main", os.path.join(base, "main.py"))
main_mod.main()
