# Security Policy

## Supported Versions

We are committed to maintaining the security of Training Lens. Security updates are provided for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in Training Lens, please help us by reporting it responsibly.

### How to Report

1. **Do NOT create a public GitHub issue** for security vulnerabilities
2. **Email us directly** at security@training-lens.org (if available) or create a private security advisory on GitHub
3. **Provide detailed information** about the vulnerability
4. **Include steps to reproduce** the issue if possible

### What to Include

When reporting a security issue, please include:

- **Description** of the vulnerability
- **Steps to reproduce** the issue
- **Potential impact** of the vulnerability
- **Affected versions** of Training Lens
- **Your contact information** for follow-up questions

### Response Process

1. **Acknowledgment**: We will acknowledge receipt of your report within 48 hours
2. **Investigation**: We will investigate the issue and assess its severity
3. **Fix Development**: If confirmed, we will develop a fix
4. **Disclosure**: We will coordinate with you on public disclosure timing
5. **Release**: We will release a security update and publish an advisory

### Security Best Practices

When using Training Lens:

#### API Keys and Secrets
- **Never commit** API keys, tokens, or secrets to version control
- **Use environment variables** for sensitive configuration
- **Rotate keys regularly** for external services (W&B, HuggingFace)

#### Model and Data Security
- **Validate inputs** when loading external datasets
- **Be cautious** with untrusted model checkpoints
- **Use secure storage** for sensitive training data

#### Network Security
- **Use HTTPS** for all external API calls
- **Verify SSL certificates** for external services
- **Be aware** of data transmission to external services

#### Environment Security
- **Keep dependencies updated** to latest secure versions
- **Use virtual environments** to isolate dependencies
- **Monitor for known vulnerabilities** in dependencies

### Known Security Considerations

#### External Services
Training Lens integrates with external services:
- **Weights & Biases**: Training data and metrics are sent to W&B servers
- **HuggingFace Hub**: Models and checkpoints are uploaded to HF servers
- **PyTorch Hub**: May download models from external sources

#### Data Handling
- Training data and model weights are processed locally
- Checkpoints may contain sensitive model information
- Exported data should be handled according to your security policies

#### Dependencies
Training Lens depends on several external packages:
- Keep dependencies updated for security patches
- Monitor security advisories for key dependencies (PyTorch, transformers, etc.)

### Vulnerability Disclosure Timeline

- **Day 0**: Vulnerability reported
- **Day 1-2**: Acknowledgment sent
- **Day 3-14**: Investigation and verification
- **Day 15-30**: Fix development and testing
- **Day 31**: Coordinated disclosure and release

### Recognition

We appreciate security researchers who help keep Training Lens secure. With your permission, we will:

- **Acknowledge your contribution** in our security advisories
- **Credit you** in our release notes
- **List you** in our contributors

### Contact

For security-related questions or concerns:

- **GitHub Security Advisories**: [Create a private security advisory](https://github.com/training-lens/training-lens/security/advisories)
- **General Security Questions**: Create a discussion in our GitHub repository

### Additional Resources

- [OWASP Machine Learning Security Top 10](https://owasp.org/www-project-machine-learning-security-top-10/)
- [Python Security Guidelines](https://python.org/dev/security/)
- [PyTorch Security](https://pytorch.org/docs/stable/notes/serialization.html#security)

Thank you for helping keep Training Lens and our community safe!