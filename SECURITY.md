# Security policy

Do not open a public issue containing a credential or exploitable detail. Report a suspected
vulnerability privately through GitHub's security advisory feature.

The application is designed to be read-only. It does not require brokerage credentials, execute
trades, or persist user portfolios. API credentials must exist only in the deployment platform's
secret manager. Rotate a credential immediately if it appears in Git history, logs, screenshots,
or a client-facing widget.

Before release, run the test, lint, static-analysis, dependency-audit, and container checks in the
README. Public deployments should also enforce HTTPS, request limits, cost limits, and monitoring.

Current-event requests use validated tickers, fixed RSS hosts, bounded results, hardened XML
parsing, timeouts, retries, and validated HTTP or HTTPS links. Headline text is rendered through
Streamlit rather than inserted as raw HTML. This public version has no AI-provider integration and
does not require any application secrets.
