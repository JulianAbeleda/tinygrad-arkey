"""Compatibility re-export for legacy extra.qk imports."""
from tinygrad.codegen.opt.compiler_policies import (PipelinePolicy, RegisterPipePlan, ResourcePlan, StoragePolicy, WaitPolicy,
  WaitCount, WaitDependency, WaitDependencyCoverage, amdllvm_wait_dependency, pipeline_policy_for_route,
  prove_wait_dependency_coverage)

__all__ = ("StoragePolicy", "WaitPolicy", "ResourcePlan", "PipelinePolicy", "RegisterPipePlan", "WaitCount", "WaitDependency",
           "WaitDependencyCoverage", "amdllvm_wait_dependency", "pipeline_policy_for_route", "prove_wait_dependency_coverage")
