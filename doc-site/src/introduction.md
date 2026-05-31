# UC Davis VMTH Cancer Registry — Developer Docs

This site contains developer handoff documentation for the **UC Davis VMTH Canine Cancer Registry**, built by ECS 193A Team 14.

## Contents

| Document | What it covers |
|---|---|
| [Project Handoff](handoff.md) | Tech stack, repo structure, what's implemented, running locally, database schema, data pipeline, known issues, remaining work, credentials |
| [User Guide](user-guide.md) | Guide for using the front-end. |
| [Future Plans](future-plans.md) | Multi-tenancy, data format standardization, auth, NLP generalization, infrastructure scaling, geographic expansion, clinic onboarding, data privacy, EHR integrations |

## Quick facts

- **Repository:** <https://github.com/ECS-193A-Team-14/UC-Davis-VMTH-Cancer-Registry>
- **Handoff date:** April 15, 2026
- **Last updated:** May 18, 2026
- **Current dataset:** 395 patients, ~2,348 case diagnoses (PetBERT predictions)
- **Frontend:** React 19 + Vite hosted on Vercel
- **Backend:** FastAPI on GCP Cloud Run
- **Database:** PostgreSQL 16 + PostGIS on Supabase
- **NLP model:** PetBERT (110M params, veterinary EHR pretrain) via GCP Batch
