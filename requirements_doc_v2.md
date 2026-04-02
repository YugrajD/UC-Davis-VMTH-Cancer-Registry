# Requirements Document v2

**Team:** Team 14  
**Project Title:** UC Davis VMTH Veterinary Cancer Registry  
**Authors:** Yugraj Dhillon, David Estrella, Chun Ho Li, Justin Pak

## Introduction

UC Davis VMTH has decades of veterinary pathology and related clinical records, but much of that information exists as free-text report data that is difficult to query, aggregate, or compare across cases. Without a structured registry, researchers cannot efficiently study cancer prevalence, tumor distributions, demographic risk factors, or geographic patterns in companion animals.

This project addresses that problem by building a veterinary cancer registry platform that ingests pathology report data, converts unstructured text into standardized cancer labels, stores the resulting structured records in a searchable database, and presents trends and geographic patterns through a researcher-facing dashboard. The goal is to make retrospective oncology data usable for population-level veterinary cancer research in California.

The technical contribution of the system is the integration of text annotation, model training, evaluation, and production inference into one pipeline-backed platform. The system maps veterinary pathology report text to the `Vet-ICD-O-canine-1` coding standard, uses the veterinary-domain transformer `PetBERT` as the production language backbone, and supports geospatial analysis through `PostGIS` so that structured oncology data can be explored over time and across regions.

## System Architecture Overview

The system architecture is a multi-component pipeline that connects data ingestion, structured storage, NLP processing, model evaluation, and researcher-facing analytics. It is designed so that raw report data can be preserved while derived cancer labels and geographic summaries are generated in a separate, modular processing flow.

### Architecture Diagram

![System architecture diagram](docs/diagrams/requirements-v2-architecture.svg)

### Components

**Frontend:** The user interface is implemented with React and TypeScript. It is responsible for dataset upload workflows, interactive filtering, table and trend views, and map-based exploration of cancer records and summaries.

**Backend:** The application layer is implemented with FastAPI in Python. It is responsible for ingesting uploaded data, validating input structure, orchestrating processing, and serving registry and analytics data back to the frontend through REST interfaces.

**Database:** PostgreSQL stores structured registry records, raw text references, metadata, and derived labels. The PostGIS extension supports geographic storage and spatial queries needed for regional aggregation and map-based analysis.

**ML/NLP subsystem:** The machine learning implementation lives under the `ml/` directory and is divided into four main workflows:
- an annotation pipeline that generates label supervision from diagnosis text,
- a production inference pipeline that scores report text against the Vet-ICD taxonomy,
- a training pipeline that updates classifiers and adapted backbones,
- and an evaluation pipeline that measures prediction quality against verified labels.

![Production ML pipeline diagram](docs/productionML.png)

**Supporting model services:** The project also includes a local Ollama-backed LLM annotation workflow. This workflow supports label generation and refinement during annotation, but it is not the main production classifier used for registry inference.

### Data Flow

1. Raw pathology or report CSV data is uploaded or prepared for ingestion.
2. Structured metadata fields and raw free-text report content are preserved so the original record remains auditable.
3. Diagnosis text can be passed through the annotation workflow to generate training labels aligned to `Vet-ICD-O-canine-1`.
4. Full pathology report text is processed by the `PetBERT`-based production pipeline to generate report embeddings and classification scores.
5. Predictions are mapped into standardized `term`, `group`, and `ICD code` outputs using the veterinary cancer taxonomy.
6. Structured outputs and associated metadata are stored in PostgreSQL.
7. Geographic attributes are queried through PostGIS to support regional aggregation and map-based visualization.
8. The frontend consumes backend REST endpoints to display case tables, filters, trend summaries, and geographic views.

### APIs / Interfaces

The architecture relies on a small number of core interfaces rather than a large number of tightly coupled services. Data enters the system through a CSV ingestion interface for pathology and registry records. The frontend communicates with the FastAPI backend through REST APIs. The backend coordinates with the ML processing layer through a decoupled boundary so that long-running annotation, inference, or retraining tasks do not block the user interface. The same backend also exposes analytics and query interfaces that return structured registry data for dashboard views and research-oriented exports.

### Dependencies

The architecture depends on several major frameworks and tools. React and TypeScript are used for the frontend because they support interactive dashboards and safer team-based UI development. FastAPI and Python are used for the backend because the same language ecosystem is needed for NLP, model training, and data processing. PostgreSQL and PostGIS are used because the application needs both structured relational storage and geospatial query support. The HuggingFace ecosystem is used for transformer-based model loading and training workflows, while Ollama supports the local annotation-assistance pipeline. GitHub Projects is used for issue tracking and project coordination because it integrates directly with the code repository and workflow history.

This architecture is modular because ingestion, storage, model execution, and analytics are separated into distinct responsibilities. That separation makes retraining easier, supports future expansion to additional data sources, and provides better privacy control than a production design that depends on external hosted inference services.

## Requirements

### Priority Rationale

High-priority stories define the minimum viable registry needed for client value: ingesting data, producing standardized cancer labels, supporting core filtering and geographic analysis, handling uncertainty safely, and protecting restricted data. Medium-priority stories improve researcher usability and analytical depth, but the system can still deliver baseline value without all of them in the first release. Lower-priority stories focus on future scalability, long-term model improvement, and broader downstream reuse once the core registry workflow is stable.

### Prioritized User Stories

#### High Priority

1. As a veterinary researcher, I want to upload pathology report datasets so that the registry can be updated with new cases.
2. As a veterinary researcher, I want free-text pathology reports to be converted into structured cancer labels so that unstructured records can be analyzed at scale.
3. As a veterinary researcher, I want each case mapped to `Vet-ICD-O-canine-1` terms, groups, and codes so that veterinary cases can be compared consistently.
4. As a veterinary researcher, I want to filter records by species, breed, sex, year, and cancer category so that I can study demographic and clinical risk patterns.
5. As a veterinary researcher, I want to view geographic cancer patterns by county or ZIP-linked region so that I can identify potential environmental trends.
6. As a veterinary researcher, I want ambiguous or low-confidence cases to be flagged for review so that uncertain model outputs are not accepted without oversight.
7. As an authorized user, I want uploaded data to remain restricted and de-identified so that patient privacy is protected.

#### Medium Priority

8. As a veterinary researcher, I want to export structured registry results so that I can perform downstream statistical analysis outside the application.
9. As a veterinary researcher, I want to view temporal trends in cancer prevalence so that I can study how patterns change over time.
10. As a veterinary researcher, I want to inspect case-level predictions and source report context so that I can assess whether a classification is reasonable.
11. As a veterinary researcher, I want to search and query records without needing to know ICD codes or SQL so that the registry remains accessible to non-technical users.
12. As a veterinary researcher, I want the system to preserve raw source text alongside structured outputs so that the registry remains auditable.
13. As a veterinary researcher, I want the system to support non-cancer or uncategorized cases so that the registry does not force every record into an incorrect cancer label.
14. As a veterinary researcher, I want to compare patterns across tumor groups so that I can investigate how different cancer categories behave across regions and populations.

#### Lower Priority

15. As a project stakeholder, I want the architecture to support onboarding additional clinics in the future so that the registry can expand beyond one institution.
16. As a project stakeholder, I want model quality to improve as more labeled data becomes available so that the registry becomes more reliable over time.
17. As a researcher, I want the registry to support future downstream research workflows so that its structured outputs remain useful beyond the dashboard itself.

### Acceptance Tests

1. **Given** a correctly formatted pathology report dataset, **when** an authorized user uploads it, **then** the system accepts the file and stores the records for processing.
2. **Given** an incorrectly formatted or incomplete upload file, **when** the user submits it, **then** the system rejects the upload and provides clear validation feedback.
3. **Given** a pathology report containing recognizable cancer language, **when** the inference pipeline processes the case, **then** the system returns a structured cancer prediction with a term, group, and ICD code.
4. **Given** a pathology report with weak, conflicting, or ambiguous cancer evidence, **when** the system processes it, **then** the result is flagged for review or handled as low confidence rather than silently treated as certain.
5. **Given** a non-cancer case or a case with no supported cancer evidence, **when** the system performs classification, **then** it does not force the record into a cancer label.
6. **Given** a populated registry dataset, **when** a user applies filters for species, breed, sex, year, or tumor category, **then** the displayed tables and visual summaries update to reflect only the matching records.
7. **Given** records with geographic attributes, **when** a user opens the geographic view, **then** the system displays aggregated cancer information by region.
8. **Given** restricted registry data, **when** an unauthorized user attempts to access uploaded records, **then** the system denies access.
9. **Given** registry data used for model processing and analytics, **when** the data is stored and exposed through the system, **then** direct personally identifying information is excluded from the accessible outputs.
10. **Given** a successfully processed cancer case, **when** the result is viewed or exported, **then** the output includes the standardized cancer term, group, and ICD code.

### Functional Requirements

1. The system shall ingest pathology report datasets from CSV-based input.
2. The system shall preserve raw report text alongside structured extracted fields.
3. The system shall classify veterinary pathology report text into standardized cancer labels using the ML pipeline.
4. The system shall map predictions to `Vet-ICD-O-canine-1` term, group, and code outputs.
5. The system shall support filtering and querying of structured cancer records.
6. The system shall support geographic aggregation of registry records for map-based analysis.
7. The system shall flag uncertain or ambiguous cases for manual review or lower-confidence handling.
8. The system shall provide visual summaries of temporal and geographic trends.
9. The system shall support export of structured registry outputs for research use.
10. The system shall restrict data access to authorized users.

### Non-Functional Requirements

1. The system shall support asynchronous or decoupled model processing so that a valid dataset upload returns an acceptance or validation response within 10 seconds even if downstream NLP processing continues in the background.
2. The system shall maintain de-identified storage and controlled access for sensitive veterinary records, and direct personally identifying information shall not appear in researcher-facing dashboards or exports.
3. The system shall support retraining or updating models without requiring a redesign of the frontend, backend, and database architecture.
4. The system shall remain usable for researchers without SQL or specialized coding knowledge, and common filter or search interactions in the dashboard should return updated results within 3 seconds for the expected project dataset size.
5. The system shall support future expansion to additional clinics, datasets, and regions, with the architecture designed to add new data sources through the ingestion pipeline rather than through a full system rewrite.
6. The system shall prioritize modularity so that annotation, training, inference, and dashboard components can evolve independently.
7. The system shall use technology choices that are maintainable by future student teams or client-side developers and should rely on widely supported frameworks with active documentation.
8. The system shall preserve traceability between structured outputs and their underlying source records so that each exported or displayed case can be linked back to its originating record identifier and source text reference.
9. The system shall support batch inference workloads of at least 1,000 pathology reports per processing run without requiring manual intervention between individual records.

## Technologies Employed

### Programming Languages

The project uses Python and TypeScript as its primary programming languages. Python is used for NLP, model training, annotation, evaluation, and backend integration because the machine learning workflows depend on libraries such as HuggingFace, Pandas, and PyTorch-compatible tooling. TypeScript is used for the frontend because static typing improves maintainability and reduces common UI integration errors in a team-based web application.

### Frontend

React is used for the frontend because the application requires an interactive dashboard with filtering, tabular views, trend visualizations, and geographic exploration. React also has a mature ecosystem for data visualization and works well with REST-based backends.

### Backend

FastAPI is used for the backend because it aligns well with Python-based ML and data-processing workflows. It supports straightforward REST API development, clear request and response modeling, and easy integration with the same runtime environment used by the project’s NLP and training code.

### Database

PostgreSQL is used as the main database because the registry requires reliable relational storage for structured case data, extracted labels, metadata, and derived analytics fields. PostGIS extends PostgreSQL with spatial storage and query support, which is necessary for regional aggregation and map-based cancer analysis.

### API Structure

The system is organized around REST interfaces because REST is a simple and appropriate fit for a React frontend communicating with a FastAPI backend. It supports predictable request patterns for uploads, querying, analytics, and structured output retrieval without adding unnecessary complexity.

### NLP / Machine Learning

The project deploys `PetBERT` as the core production language model. PetBERT is used because it was trained on veterinary electronic health records and is therefore better aligned with veterinary pathology terminology than general-language BERT models. This project needs a model that can better represent animal-specific clinical language, abbreviations, and report phrasing that appear in veterinary oncology records.

The implementation uses `PetBERT` embeddings as the backbone for classification and structured cancer label prediction. Report text and taxonomy labels are embedded into the same vector space, and those representations are then used by the production pipeline to score likely cancer labels. This makes PetBERT a stronger fit than standard `BERT Base`, which is trained on general English and lacks veterinary domain adaptation. Human-clinical models such as `ClinicalBERT` or biomedical literature models such as `BioBERT` may be useful references, but they are still not trained specifically on veterinary terms.

Externally hosted LLMs are not the primary production path because they create privacy concerns, recurring usage costs, and less deterministic structured-output behavior. The project does include a local LLM-based annotation workflow, but that workflow exists to support annotation and label generation rather than to serve as the main deployed classifier. The current implementation direction is a combination of annotation support, `PetBERT`-based production inference, and contrastive adaptation plus classifier-based scoring as the strongest internal approach documented to date.

### Geospatial Analysis

PostGIS is used for spatial storage and query execution, while GeoPandas is a useful companion tool for geospatial preprocessing and analysis tasks in Python. Together they support spatial joins, aggregation by geographic region, and map-ready data preparation for the dashboard.

### Project Management / Tooling

GitHub Projects is used for project management because it integrates directly with the repository, issues, and code review workflow. This keeps implementation tracking aligned with the actual development process and reduces coordination overhead.

## Real-World Constraint Analysis

### Cost

Project cost is shaped by both software infrastructure and machine learning operations. Development costs include model training, fine-tuning, evaluation, and experimentation with annotated veterinary oncology data. Deployment costs are lower and more predictable when inference is self-hosted rather than charged per token through an external API. In addition, labeling and validating oncology data is itself a significant cost because domain expertise is needed to generate reliable supervision.

### Space

The platform can be hosted either locally or in cloud infrastructure depending on available resources. Space requirements include storage for raw pathology records, structured registry tables, cached embeddings, trained model artifacts, evaluation outputs, and exported analytics datasets. These needs are manageable, but they must still be considered when the registry grows.

### Security

The system must protect uploaded veterinary records through controlled access and authenticated workflows. Uploaded data should not be exposed publicly, and the registry should restrict access to authorized users only. This matters because even though the platform is not a public clinical system, it still manages health-adjacent research data that should be handled carefully.

### Privacy

Privacy is a major design constraint. Data used for training, inference, and analytics should be de-identified before use, and the system should avoid exposing direct personally identifying information. Avoiding third-party hosted production inference also reduces the privacy risk associated with sending sensitive report text outside the project’s controlled environment. Veterinary records still deserve strong confidentiality handling, especially when they originate from institutional clinical systems.

### Scalability

The system must scale in two different ways. First, the platform itself must support more records, more derived outputs, and eventually more participating clinics or regions. Second, the model pipeline must scale to different writing styles and data distributions. Because the current dataset is institution-specific, performance may decrease when the system is applied to records from different clinics. That makes retraining and revalidation important parts of future scalability.

### Maintainability

Long-term maintainability depends on keeping the architecture modular and well documented. The annotation pipeline, training workflow, production inference pipeline, and dashboard-facing application logic should be separable enough that future contributors can improve one area without destabilizing the entire system. Maintainability also depends on using technologies that are widely understood and well supported in academic and applied software environments.

### Primary Challenge

The biggest ML challenge in this project is the limited availability of labeled veterinary oncology data. Labels are expensive to create, domain-specific, and unevenly distributed across cancer groups. Some tumor categories are underrepresented, which limits generalization and makes certain model architectures harder to train effectively. This challenge directly motivates the project’s annotation workflow, iterative retraining strategy, and emphasis on preserving auditable source text and evaluation pipelines.

## Social / Legal Aspects

### Social / Ethical

The registry has clear research value for veterinary oncology because it makes large amounts of previously unstructured pathology data easier to study. By supporting temporal and geographic analysis, the system may also enable exploratory comparative insights relevant to environmental exposure patterns and broader public health questions. At the same time, the system has ethical limitations: a model trained primarily on data from one institution may be biased toward that institution’s reporting style, and rare cancer categories may be predicted less reliably than common ones. For that reason, the registry is intended to support research and structured analysis, not to act as an autonomous diagnostic system.

### Legal / Governance

From a governance perspective, the client retains ownership of the underlying data used for model training, analysis, and registry development. Records should be handled in de-identified form, and source data should not be made publicly accessible. Structured outputs should be used responsibly, with human oversight when interpretation matters, especially if model outputs are used to inform research conclusions. This approach supports privacy, data stewardship, and accountable use of model-assisted results.

## Glossary of Terms

- **Acceptance Test**: A product-level test written from the user perspective that defines how a requirement is verified.
- **Annotation Pipeline**: A workflow that converts diagnosis text into training labels used to supervise the machine learning system.
- **API (Application Programming Interface)**: A defined way for software components to exchange data and functionality.
- **Contrastive Fine-Tuning**: A training approach that adjusts an embedding model so related text pairs are closer together in vector space.
- **De-identification**: The removal of direct personally identifying information from a dataset before it is used or shared.
- **FastAPI**: A Python web framework used to build backend APIs.
- **Geospatial Analysis**: Analysis that uses geographic location or region-based data to identify spatial patterns.
- **NLP (Natural Language Processing)**: A field of computing focused on understanding and extracting information from human language text.
- **PetBERT**: A transformer model trained on veterinary electronic health records and used in this project as the main production language backbone.
- **PostGIS**: A PostgreSQL extension that adds geographic object storage and spatial query support.
- **PostgreSQL**: An open-source relational database system used for structured application data.
- **Presence Classifier**: A classifier that scores whether a given report and cancer label pair should be treated as a positive match.
- **React**: A JavaScript and TypeScript library used to build interactive web interfaces.
- **REST**: A common web API style based on standard HTTP requests and resource-oriented communication.
- **Vet-ICD-O-canine-1**: A veterinary cancer coding standard derived from ICD-O for canine neoplasms and used as the project’s target taxonomy.

## References

- Take C.H.A.R.G.E. https://takechargeregistry.com/
- Veterinary Medical Database (VMDB). https://vmdl.missouri.edu/
- SAVSNET. https://www.liverpool.ac.uk/savsnet/
- Swiss Canine and Feline Cancer Registry. https://www.zora.uzh.ch/entities/publication/cdca5d34-1aad-4a6f-a73a-53cf232e53a8
- Australian Companion Animals Registry of Cancers (ACARCinom). https://veterinary-science.uq.edu.au/australian-companion-animal-registry-cancers
- UCSF Catchment Area Dashboard. https://cancer.ucsf.edu/catchment-area-dashboard
- Farrell S, Appleton C, Noble P-JM, Al Moubayed N. "PetBERT: automated ICD-11 syndromic disease coding for outbreak detection in first opinion veterinary electronic health records." *Scientific Reports* 13, 18015 (2023). DOI: 10.1038/s41598-023-44047-8
- Pinello K, Baldassarre V, Steiger K, et al. "Vet-ICD-O-Canine-1, a System for Coding Canine Neoplasms Based on the Human ICD-O-3.2." *Cancers (Basel).* 2022;14(6):1529. DOI: 10.3390/cancers14061529
- World Health Organization. ICD-11. https://icd.who.int/en/
- SEER Program. ICD-O-3 Coding Materials. https://seer.cancer.gov/icd-o-3/
- PostgreSQL Documentation. https://www.postgresql.org/docs/
- PostGIS Documentation. https://postgis.net/documentation/
- FastAPI Documentation. https://fastapi.tiangolo.com/
- React Documentation. https://react.dev/
- Hugging Face Documentation. https://huggingface.co/docs
