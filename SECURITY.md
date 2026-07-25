# Security Policy

## Supported versions

SDPX is experimental and has not reached a stable 1.0 release. Security and
correctness fixes are applied to the latest development branch; older commits
and untagged snapshots are not maintained as supported release lines.

## Reporting a vulnerability

Do not open a public issue for a vulnerability that could expose credentials,
private benchmark data, cluster access, or arbitrary code execution.

Use GitHub's private vulnerability reporting feature for this repository when
available. If it is not enabled, contact the maintainer through the address in
`Project.toml` and clearly mark the message as a private security report.

Include:

- affected commit or version;
- operating system and Julia version;
- a minimal reproduction that contains no private data;
- expected and observed behavior;
- potential impact;
- any known workaround.

Please allow reasonable time for acknowledgement, investigation, and a
coordinated fix before public disclosure.

## Numerical correctness

Incorrect optimization results are important, but most are correctness bugs
rather than security vulnerabilities. Report them through the normal issue
tracker unless they can cross a trust boundary, expose confidential inputs, or
execute unintended code. Include the arithmetic type, precision, thread count,
options, status, termination reason, and final certificate diagnostics.

## Credentials and cluster data

Never commit access tokens, SSH private keys, scheduler credentials, private
hostnames, proprietary solver licenses, or confidential problem instances.
Revoke any credential immediately if it is exposed; deleting it from the
latest commit is not enough because Git history and cached artifacts may retain
it.
