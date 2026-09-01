# Reporting a vulnerability

**This repository is a fork, and the process below is NVIDIA's.** Route reports
by what the issue is in:

- **This fork's own code** — the tracing package under `src/alpamayo1_5/trace/`
  and the scripts that record runs: open a private security advisory on this
  repository (Security → Report a vulnerability). NVIDIA PSIRT does not maintain
  this code and cannot act on it.
- **Upstream Alpamayo 1.5, the released model, or any other NVIDIA product**:
  use NVIDIA's process, below. Anything that reproduces on
  [NVlabs/alpamayo1.5](https://github.com/NVlabs/alpamayo1.5) without this
  fork's scripts belongs there.

---

## NVIDIA: Report a Security Vulnerability

To report a potential security vulnerability in any NVIDIA product, please use either:

* This web form: [Security Vulnerability Submission Form](https://www.nvidia.com/en-us/support/submit-security-vulnerability/), or
* Send email to: [NVIDIA PSIRT](mailto:psirt@nvidia.com)

If reporting a potential vulnerability via email, please encrypt it using NVIDIA's public PGP key ([see PGP Key page](https://www.nvidia.com/en-us/security/pgp-key/)) and include the following information:

1. Product/Driver name and version/branch that contains the vulnerability
2. Type of vulnerability (code execution, denial of service, buffer overflow, etc.)
3. Instructions to reproduce the vulnerability
4. Proof-of-concept or exploit code
5. Potential impact of the vulnerability, including how an attacker could exploit the vulnerability

See https://www.nvidia.com/en-us/security/ for past NVIDIA Security Bulletins and Notices.
