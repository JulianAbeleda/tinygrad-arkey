"""Compatibility re-export for legacy extra.qk imports."""
from tinygrad.codegen.opt.compiler_policies import (PipelinePolicy, RegisterPipePlan, ResourcePlan, StoragePolicy, WaitPolicy,
  WaitDependency, amdllvm_wait_dependency, pipeline_policy_for_route)

__all__ = ("StoragePolicy", "WaitPolicy", "ResourcePlan", "PipelinePolicy", "RegisterPipePlan", "WaitDependency",
           "amdllvm_wait_dependency", "pipeline_policy_for_route")
