# Model Card: Public Sector Ticket Classifier

## Intended Use

This model routes synthetic government/public-sector helpdesk tickets to operational queues such as login support, procurement, compliance, infrastructure, HR grievance intake, and change management. It is intended for triage assistance with human oversight.

## Out-of-Scope Use

Do not use this model to make legal, benefits, employment, disciplinary, HR outcome, eligibility, or citizen-service entitlement decisions without human review. It should not be used for surveillance, identity verification, or prioritising emergency response.

## Training Data Provenance

Training data is synthetic only and generated from controlled templates plus varied wording. Version tag: `synthetic-public-sector-ticket-classifier-v1.2.0`. The dataset intentionally excludes real PII, credentials, internal URLs, IP addresses, and real agency names.

## Evaluation Results

| Metric | Value | Notes |
|---|---:|---|
| Labels | 12 | Balanced synthetic test split |
| Test examples | 300 | 25 per label |
| Macro F1 | Pending trained run | Populated by CI after training |
| Minimum per-label F1 | Pending trained run | Promotion requires at least 0.65 |
| Calibration ECE | Pending trained run | Promotion target is at most 0.10 |

## Fairness

The dataset includes non-native English phrasing to improve access equity, but no demographic attributes are present and demographic fairness testing has not been performed. Before production use, evaluate false routing rates across geography, language background proxies where lawful and appropriate, disability-related phrasing, and role seniority.

## Refresh Cadence

Retrain at least every 6 months, or sooner if macro F1 drops more than 5% from the accepted baseline, label distribution drift exceeds the JS threshold, or new ticket queues are introduced.

## Owner

Owner: nprasann  
Contact: repository issues or the designated platform operations team.

## Safety Notes

The inference service logs audit metadata only. It never logs raw ticket text.
