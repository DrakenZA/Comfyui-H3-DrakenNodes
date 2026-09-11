"""Run the test suite against a ComfyUI checkout on CPU.

    python tests/run_tests.py --comfy /path/to/ComfyUI

Stubs comfy_aimdo when it is not installed (only needed at runtime by core, not by these tests).
"""
import argparse
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(HERE)


class _StubLoader(importlib.abc.Loader):
    """Permissive stub for optional compiled deps: any attribute is a MagicMock, any submodule exists."""

    def create_module(self, spec):
        from unittest import mock
        m = types.ModuleType(spec.name)
        m.__path__ = []
        m.__getattr__ = lambda attr, _n=spec.name: mock.MagicMock(name=f"{_n}.{attr}")
        return m

    def exec_module(self, module):
        pass


class _StubFinder(importlib.abc.MetaPathFinder):
    def __init__(self, roots):
        self.roots = tuple(roots)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.roots:
            return importlib.machinery.ModuleSpec(fullname, _StubLoader(), is_package=True)
        return None


def _install_stubs(roots):
    missing = []
    for name in roots:
        try:
            importlib.import_module(name)
        except ImportError:
            missing.append(name)
    if missing:
        sys.meta_path.append(_StubFinder(missing))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", required=True)
    args = ap.parse_args()
    sys.argv = [sys.argv[0], "--cpu"]
    sys.path.insert(0, args.comfy)
    _install_stubs(("comfy_aimdo", "comfy_kitchen"))
    # import the pack as a package under a stable alias (folder name has dashes)
    spec = importlib.util.spec_from_file_location(
        "h3_drakennodes_pkg", os.path.join(PKG_DIR, "__init__.py"),
        submodule_search_locations=[PKG_DIR])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["h3_drakennodes_pkg"] = pkg
    spec.loader.exec_module(pkg)
    sys.path.insert(0, HERE)
    import test_grid
    import test_handler_cpu
    import test_nodes_cpu
    import test_extras_cpu
    test_grid.test_basic_grid()
    test_grid.test_plan()
    test_handler_cpu.test_identity_and_phase()
    test_handler_cpu.test_keyframes_remap()
    test_handler_cpu.test_refs_and_split_conds()
    test_handler_cpu.test_freenoise_shapes()
    test_nodes_cpu.test_long_latent_and_trim()
    test_nodes_cpu.test_context_windows_node()
    test_extras_cpu.test_halo_and_probe()
    test_extras_cpu.test_guide_starting_before_window()
    test_extras_cpu.test_branch_isolation_and_split_conds()
    test_extras_cpu.test_margins()
    test_extras_cpu.test_audio_lock_and_plan()
    print("ALL TESTS OK")


if __name__ == "__main__":
    main()
