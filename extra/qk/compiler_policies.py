"""Compatibility re-export for legacy extra.qk imports."""
from tinygrad.codegen.opt.compiler_policies import (RegisterPipePlan, ResourcePlan, StoragePolicy, WaitPolicy, WaitDependency, amdllvm_wait_dependency)

__all__ = ("StoragePolicy", "WaitPolicy", "ResourcePlan", "RegisterPipePlan", "WaitDependency", "amdllvm_wait_dependency")
