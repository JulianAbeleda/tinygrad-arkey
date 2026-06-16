# Worked Example: Development to Audit Loop

This example shows the full loop:

1. operator assigns work
2. development agent completes it
3. development agent hands off to audit
4. audit agent reviews it
5. operator decides what happens next

The example is intentionally generic. Replace the placeholders with real project files and commands.

## Step 1: Operator Sends a Development Task

```text
Role:
Development Agent

Model:
use active assignment table

Assignment source:
explicit user assignment

Duration:
this task

Task:
Add a short troubleshooting section to the setup guide.

Context:
New users are missing the common setup failure cases. The guide should explain the most likely causes without becoming a full support manual.

Scope:
- structure/User_Guide/getting-started.md

Out of scope:
- Do not change deployment docs.
- Do not change architecture docs.
- Do not add project-specific secrets, paths, or credentials.

Success criteria:
- troubleshooting section exists
- wording is beginner-readable
- setup guide remains project-agnostic
- no unrelated files are changed

Verification:
Manual doc review.

Handoff artifact:
List changed files, summarize the doc change, state verification, and name any risks.
```

## Step 2: Development Agent Responds

```text
Role: Development Agent
Model: <model>
Assignment source: explicit user assignment
Duration: this task
Purpose: update the setup guide within the stated scope.
Scope: structure/User_Guide/getting-started.md
Boundaries: I will not modify deployment or architecture docs.
Done means: the requested documentation is updated, scoped, reviewed, and ready for audit.
```

The development agent makes the change, then produces a handoff.

## Step 3: Development to Audit Handoff

```text
From:
<model>, Development Agent

To:
<model>, Audit Agent

Task completed:
Added a troubleshooting section to the setup guide.

Artifacts:
- changed files: structure/User_Guide/getting-started.md
- commit id or diff reference: uncommitted diff

Verification:
Manual doc review. Confirmed the section is beginner-readable and does not include project-specific secrets, paths, or credentials.

Known risks:
No command-based verification was run because this was a documentation-only change.

Requested next action:
Audit the implementation against the task, the stated scope, and the project documentation standard. Report findings ordered by severity. Do not implement fixes unless reassigned.
```

## Step 4: Operator Sends the Handoff to Audit

```text
Role:
Audit Agent

Model:
use active assignment table

Assignment source:
handoff

Duration:
this task

Task:
Review the development handoff and changed file.

Context:
The development task added beginner troubleshooting guidance to the setup guide.

Scope:
- development handoff pasted above
- structure/User_Guide/getting-started.md

Out of scope:
- Do not rewrite the guide.
- Do not review unrelated files.

Audit standard:
Use the task success criteria and the project documentation standard.

Success criteria:
- findings are ordered by severity
- evidence is cited when available
- verification gaps are named
- recommendation is accept, revise, or block

Verification:
Inspect the changed file and compare it to the original task.

Handoff artifact:
Findings, open questions, verification reviewed, and recommendation.
```

## Step 5: Audit Agent Responds

```text
Role: Audit Agent
Model: <model>
Assignment source: handoff
Duration: this task
Purpose: review the setup guide change against the task contract.
Scope: development handoff and structure/User_Guide/getting-started.md
Boundaries: I will not implement fixes or review unrelated files.
Done means: findings are ordered by severity, evidence is cited, verification gaps are named, and a recommendation is given.
```

Example audit result:

```text
Findings:
None.

Open questions:
None.

Verification reviewed:
- development handoff
- structure/User_Guide/getting-started.md
- task scope and success criteria

Recommendation:
Accept.
```

## Step 6: Operator Decision

If the audit says `accept`, the operator can close the task or ask a development agent to commit or push.

If the audit says `revise`, copy the audit findings into a new development task.

If the audit says `block`, stop the work until the blocking issue is resolved.
