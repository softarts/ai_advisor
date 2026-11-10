"""Thin entry point: python -m strands_advisor → delegates to strands.__main__"""
import runpy
runpy.run_module("strands.__main__", run_name="__main__", alter_sys=True)
