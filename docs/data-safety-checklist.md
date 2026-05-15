# Data Safety Checklist

Use this checklist before training, evaluating, committing, or publishing.

- [ ] Dataset examples are synthetic.
- [ ] No real client, resident, employee, agency, case, ticket, or incident data is included.
- [ ] No secrets, passwords, API keys, bearer tokens, private keys, or connection strings are included.
- [ ] No internal URLs, hostnames, logs, or IP addresses are included.
- [ ] No regulated information is included.
- [ ] `python -m ticket_classifier.data_validate data/train.jsonl` passes.
- [ ] `python -m ticket_classifier.data_validate data/validation.jsonl` passes.
- [ ] `python -m ticket_classifier.data_validate data/test.jsonl` passes.
- [ ] The generated model card accurately describes synthetic training data and limitations.
- [ ] Public publishing has been reviewed by a responsible human owner.
