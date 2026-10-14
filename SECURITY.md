# Security policy

## Reporting a vulnerability

If private vulnerability reporting is enabled for this repository, report
through GitHub's
[Security Advisories](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
for this repository, or by opening a minimal issue that asks for a private
contact channel **without disclosing the details**.

Please do not open a public issue containing an exploit, a credential, or
real scroll data.

## Scope and expectations

This is research code with no network service, no authentication and no
user accounts. The realistic security surface is:

- **Data handling**: the code fetches public Vesuvius datasets at runtime.
  The contribution policy excludes new scan data, labels, coordinates, AFV
  files and weights; a redistribution exception requires explicit review of
  the applicable terms (see [`DATA.md`](DATA.md)).
- **Credentials**: no credentials belong in this repository. If you spot a
  committed secret, report it privately so it can be rotated and removed.

There is no bug bounty and no response-time guarantee; reports are handled on
a best-effort basis.
