# Codex Project Instructions

- Avoid the GitHub connector for this repository unless the user explicitly asks for it; use local Git over SSH instead.
- The remote is expected to be `git@github.com:Rocky18919975029/Inference-Time-Distribution-Steering.git`.
- Run `git push` as its own command and wait at least 30 seconds before treating it as stalled. Normal GitHub SSH operations on this machine take about 2-3 seconds.
- Do not chain `git status`, `git add`, `git commit`, and `git push` into a single command when reporting progress to the user. Keep commit and push as separate visible steps.
- Before pushing, run a focused verification command for the changed files when feasible.
