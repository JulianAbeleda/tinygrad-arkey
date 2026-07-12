"""Compatibility re-export for legacy extra.qk imports."""
from tinygrad.codegen.opt.compiler_policies import (RegisterPipePlan, ResourcePlan, StoragePolicy, WaitPolicy)

__all__ = ("StoragePolicy", "WaitPolicy", "ResourcePlan", "RegisterPipePlan")
