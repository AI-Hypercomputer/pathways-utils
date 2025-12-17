---
name: Bug report
about: Create a report to help us improve
title: ''
labels: ["bug", "triage"]
assignees: ''

---

**Type of the bug**
Identify whether it is a pathway-utils bug, or a general Pathways bug or an ML bug.

**Initial troubleshooting steps**
- Check the user workload log and confirm the error is not due to the workload itself.
- Check whether this is a bug with Pathways or the same error also happens to running with multi-controller JAX.

**Pathways image SHAs or tags**
The SHAs or tags of the Pathways server image and the Pathways proxy server image used.

**Describe the bug**
A clear and concise description of what the bug is.

**Expected behavior**
A clear and concise description of what you expected to happen.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Run command '....'
3. See error

**Logs**
The output of `kubectl describe jobset <jobset_name>`. Logs from the workload, pathways-proxy, pathways-rm, and all of the pathways-workers would be helpful. Please share the logs in a tarball or in a GCS bucket.

**Additional context**
Add any other context about the problem here.
