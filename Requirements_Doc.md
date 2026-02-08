# **Product Requirements Document**

Team 14: UC Davis VMTH Cancer Registry  
Authors: [Yugraj Dhillon](mailto:ydhillon@ucdavis.edu), [David Estrella](mailto:dtestrella@ucdavis.edu), [Chun Ho Li](mailto:lchli@ucdavis.edu), [Justin Pak](mailto:jetpak@ucdavis.edu)

## Table of Contents {#table-of-contents}

[Table of Contents	1](#table-of-contents)

[Introduction	2](#introduction)

[Competitive Analysis	3](#competitive-analysis)

[Take C.H.A.R.G.E	3](#take-c.h.a.r.g.e)

[VMDB	3](#vmdb)

[SAVSNET	3](#savsnet)

[Swiss Canine and Feline Cancer Registry	3](#swiss-canine-and-feline-cancer-registry)

[ACARCinom	4](#acarcinom)

[Technology Survey	5](#technology-survey)

[Front-end	5](#front-end)

[Back-end	5](#back-end)

[Database	6](#database)

[API Structuring	7](#api-structuring)

[NLP/Data Extraction	8](#nlp/data-extraction)

[Geospatial Analysis	8](#geospatial-analysis)

[Project Management Software	9](#project-management-software)

[System Architecture Overview	10-11](#system-architecture-overview)

[Requirements	12](#requirements)

[User Stories	12](#user-stories)

[Prototyping Code	13](#prototyping-code)

[Technologies Employed	14](#technologies-employed)

[Real-World Constraint Analysis	15](#real-world-constraint-analysis)

[Cost	15](#cost)

[Space	15](#space)

[Security	15](#security)

[Privacy	15](#privacy)

[Scalability	15](#scalability)

[Maintainability	15](#maintainability)

[Social/Legal Aspect of the Product	16](#social/legal-aspect-of-the-product)

[Glossary of Terms	17](#glossary-of-terms)

[References	18](#references)

## Introduction  {#introduction}

There exists refined and expansive databases for human cancer cases, but the same cannot be said for animals such as dogs and cats. Efforts have been made to create one, but nothing definitive has yet to be created. UC Davis, one of the number one Veterinarian programs in the world has 30+ years of free text clinical data that is unusable currently due to it being unorganized. 

What we aim to accomplish is to organize the clinical data, and to create a cancer registry with a dashboard for researchers to use to identify geographical data to help us understand cancer risks in certain regions. It is not only helpful to know that cancer rates are high or low in a region for dogs and cats, but it can also reveal cancer risks that may lay for humans as well (e.g., we could explore if environmental factors are affecting both humans and animals in an area). This registry will be one of a kind in California, and we hope that other states in this nation will adopt this as well. 

## Competitive Analysis {#competitive-analysis}

We analyzed five similar existing cancer registries:

### Take C.H.A.R.G.E {#take-c.h.a.r.g.e}

| *What they do well* | *Areas for improvement* |
| :---- | :---- |
| Allows the public to upload their dog’s medical records anonymously Informs users of average costs and treatment options for specific cancers in their region Uses the Vet-ICD-O-canine-1 coding system to allow comparison between dogs and humans | Allowing the public to upload records includes a risk of erroneous or incomplete medical records Funded by pharmaceutical company (Jaguar Health) which brings unease as to what is done with the data |

### VMDB {#vmdb}

| *What they do well* | *Areas for improvement* |
| :---- | :---- |
| Receives data from only Veterinary Teaching Hospitals, ensuring accurate diagnoses Operating since 1964 so it has a history of data that can be used to track cancer trends | Uses the SNOMED coding system which requires specific formatting, not scalable to other Veterinarian facilities Requires formal requests and fees to access the database Data is skewed to pets whose owners could afford university specialists |

### SAVSNET {#savsnet}

| *What they do well* | *Areas for improvement* |
| :---- | :---- |
| Receives data in real-time through the electronic health records of participating private vet clinics and labs Uses NLP (PetBERT) and Text Mining to map data to codes Reduces data bias as less specialized clinics can contribute data | Uses the VeNom coding system (UK standard) and SNOMED-CT which are not as scalable Uses data only from the UK |

### Swiss Canine and Feline Cancer Registry {#swiss-canine-and-feline-cancer-registry}

| *What they do well* | *Areas for improvement* |
| :---- | :---- |
| Tracks all cases in a specific geographic population, allowing calculation of true incidence rates Uses the Vet-ICD-O-canine-1 coding system to allow comparison between dogs and humans | Restricted to Switzerland, lack of scalability Maintenance is expensive and labor-intensive due to level of detail |

### ACARCinom {#acarcinom}

| *What they do well* | *Areas for improvement* |
| :---- | :---- |
| Gathers data in a convenient way (through biopsy) which reduces additional work required Uses the Vet-ICD-O-canine-1 coding system to allow comparison between dogs and humans | Requires tumor biopsy for data, leading to undercounting of cancer if a pet is not biopsied |

## Technology Survey {#technology-survey}

### Front-end {#front-end}

Conclusion: We evaluated both React and Angular and we came to the conclusion that React would be the most logical choice. It’s the most supported frontend framework out there. It has strong support for data visualization libraries, it’s easy to integrate with REST APIs, and it’s widely adopted in research environments over angular. 

| React |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Commonly used for websites with interactive elements Relatively easy to pick up | Might be too heavy and can easily end up creating a slow website |

| Angular |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Strong TypeScript support | Steep learning curve Probably overkill for our use case |

### Back-end {#back-end}

Conclusion: NLP pipelines and data processing are primarily Python-based. Coupled with our plan to use BERT, Python is an obvious inclusion for our project (at least on the business logic side). We also plan on using FastAPI for backend development since we felt the FastAPI’s strong points aligned with our team’s goals. Lastly, for the website itself, our plan is to code the website using TypeScript due to its benefits over JavaScript (type safety, tooling support, error catching, etc) making it well suited for a team project.

| FastAPI |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Native support for NLP (BERT, SpaCy, HuggingFace) Easy REST API creation Supports powerful libraries like Pandas and NumPy Widely used in research environments | Slightly slower than node for high-throughput workloads (relevant given the amount of data we’re handling) |

| Node.js/Bun |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Good frontend tooling support Good native ecosystem Since we have no tech debt we don’t have to worry about trying to adopt new tech | New tech so there might be issues Limited ML/NLP support as far as we know Self-contained ecosystem doesn’t matter as much when we still need Python anyways for BERT |

### Database {#database}

Conclusion: We’ll go with PostgreSQL initially since it suits our use case the best, but may switch to a MySQL service depending on how things pan out at the start.

| MySQL |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Simple to setup Good performance for read-heavy workloads Native full-text search capabilities | Weaker geospatial support since can’t use PostGIS Less flexible with complex data types |

| PostgreSQL |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Geospatial support via PostGIS extension Native full-text search capabilities Strong data integrity with ACID compliance | Slightly more complex setup than MySQL |

| MongoDB |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Flexible schema, good for inconsistent medical record formats Easy to iterate on data models | Not as good for doing relational queries (e.g., linking patient records, diagnoses, and outcomes to each other) Geospatial features less mature compared to PostGIS extension |

| Elastisearch |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Excellent for doing processes involving full-text search (e.g., pathology reports) Fast aggregations and filtering Good for exploratory data analysis | Complex to maintain |

### API Structuring {#api-structuring}

Conclusion: The clear choice is RESTful APIs to connect the frontend with the backend, it just provides the most simplicity with good performance. It’ll be easy to debug and it’ll provide the analytics endpoints necessary to create the dashboard.

| REST |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Simple Native support in FastAPI Works well with React Minimal learning curve | Can overfetch in some scenarios (mostly a non-issue given our use case). |

| GraphQL |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Flexible queries | More complex, higher learning curves |

| gRPC |  |
| ----- | ----- |
| **Pros** | **Cons** |
| High performance Real time streaming | Overkill for our use case |

### NLP/Data Extraction {#nlp/data-extraction}

Conclusion: The client wants BERT, so the best option for doing NLP is using BERT. Also using an LLM like ChatGPT for example presents many privacy issues that we do not want to go down. 

| BERT |  |
| ----- | ----- |
| **Pros** | **Cons** |
| It’s what the client wants us to use. The client is familiar with BERT as well so they’ll have an easier time using the final product when we deliver it to them. | The entire team has to take some time get familiar with BERT (not really a downside so much as a required aspect). |

| LLM APIs |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Strong extractions Minimal | Cost Privacy Concerns |

| SpaCy NER |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Lightweight Easy to customize | Lower accuracy |

### Geospatial Analysis {#geospatial-analysis}

Conclusion: We decided that using both for this project ended up being the best choice. GeoPandas for geospatial data processing, and PostGIS for spatial storage and queries. This will allow us to do spatial joins between cancer cases and census tracts. Frontend visualization will be done through react mapping libraries.

| PostGIS (PostgreSQL Extension) |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Stores geometry directly in the database Enables spatial queries Allows aggregations by census tract or region | Lengthy setup |

| GeoPandas |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Python-native Supports spatial joins Integrates easily with Pandas workflow | Slower when working with big datasets |

### Project Management Software {#project-management-software}

Conclusion: We will be using GitHub Projects for now due to its tight integrations, but if it fails to fulfill our needs then we can quickly and easily pivot to a Trello-based solution

| Trello |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Easy to setup More familiar | Some features we want might be paywalled Requires setup for automation (e.g., PM needs to write GitHub Actions) |

| GitHub Projects |  |
| ----- | ----- |
| **Pros** | **Cons** |
| Can be linked with issues on GitHub Comes built in to GitHub Tight integration with GitHub issues | Less familiar (more onboarding/setup) Less features Might not have the features we want |

## System Architecture Overview  {#system-architecture-overview}

![][image1]

We have a three layered approach here that will be able to support cancer data ingestion, natural language processing, geospatial awareness, and interactive visualization. CSV files will form the system’s raw data portion and are stored locally to preserve original records. 

The frontend is implemented using React and this will provide the interface for uploading datasets, exploring cancer case records, and visualizing statistics and geographic distributions. The frontend will communicate with the backend through RESTful API endpoints. 

The backend is built with FastAPI in Python and serves at the central layer of this application. It will handle CSV ingestion, input validation, and storage of structured fields into a PostgreSQL database with PostGIS for geospatial queries. Unstructured text is preserved and forwarded to a separate NLP processing worker. 

The NLP worker performs asynchronous extraction using BERT (possible we change models) to identify the cancer type and then those extracted results are written back into PostgreSQL. This will add onto the original records without overwriting any of the raw data.

PostgreSQL will store normalized clinical data and it will allow us to use PostGIS so that we can conduct spatial aggregation. Through that we can do geographical analysis of cancer by county and ZIP code.

We’ll separate the NLP worker from the API layer so that we prevent long running interface tasks from affecting the application responsiveness. This way, we can allow the processing pipeline to scale independently from the user facing services.

In conclusion, the architecture prioritizes the important aspects of a health oriented project, data integrity, privacy, and modularity. This way we can have structured clinical analytics while maintaining an auditable raw data layer. 

## Requirements {#requirements}

### User Stories {#user-stories}

1. As a user, I want to view diagnoses coded with the Vet-ICD-O-canine-1 coding system to compare with human cancer registry data to determine environmental and other factors involved.  
2. As a user, I want to upload static data files that will update the registry with the latest patient records without needing a complex API.  
3. As a user, I want to be able to upload free-text notes that will be converted into structured data for analysis.  
4. As a user, I want to keep all of the data private and secure to avoid a patient’s data from being traceable, preserving anonymity.  
5. As a user, I want to view California counties through a heatmap to identify geographically relevant environments.  
6. As a user, I want to view a trend line over the years to determine if the prevalence of specific cancers is increasing or decreasing.  
7. As a user, I want to filter visualization data by breed, sex, and tumor type to investigate specific risk factors.  
8. As a user, I want the system to flag ambiguous diagnoses for manual review to avoid discarding data.  
9. As a user, I want the system to handle different types of data input to avoid discarding data.  
10. As a user, I want to be able to query with little to no knowledge of the coding system or SQL.  
11. As a user, I want to be able to view visualized data to efficiently understand key statistics with little effort.

## Prototyping Code {#prototyping-code}

Github URL: [https://github.com/ECS-193A-Team-14/UC-Davis-VMTH-Cancer-Registry](https://github.com/ECS-193A-Team-14/UC-Davis-VMTH-Cancer-Registry) 

## Technologies Employed {#technologies-employed}

**Programming Languages:** Python (NLP), TypeScript (Frontend)  
**Backend:** FastAPI  
**Frontend:** React  
**Database:** PostgreSQL  
**API Structuring:** REST  
**NLP/Data Extraction:** BERT (+ Potential LLM Solution)  
**Geospatial Analysis:** PostGIS/GeoPandas  
**Project Management Software:** GitHub Projects (Migrate to Trello if run into roadblock)

## Real-World Constraint Analysis {#real-world-constraint-analysis}

### Cost {#cost}

The majority of project costs will be incurred during the development phase, primarily due to the computational resources required to train the machine learning model on client-provided data. During deployment, costs will mainly consist of maintaining the server infrastructure, which is expected to be relatively low. As the platform expands to include additional clinics or datasets, periodic retraining of the model will be necessary, resulting in additional computational expenses.

### Space {#space}

If physical space is a concern, the server infrastructure can be hosted using cloud computing services, eliminating the need for on-site hardware. Alternatively, the system could be maintained on a desktop-sized machine capable of hosting both the database and application server.

### Security {#security}

Any portion of the platform that allows clients to upload sensitive patient data will be protected through authentication mechanisms such as password-restricted access. Uploaded data will not be publicly accessible, and access will be limited strictly to authorized users.

### Privacy {#privacy}

All data used for training the model, as well as any future datasets, will be fully de-identified prior to use. As a result, no personally identifiable information (PII) will be stored or processed, minimizing potential privacy risks and ensuring compliance with data protection expectations.

### Scalability {#scalability}

While we have access to extensive records from a single clinic for training, we anticipate a potential decrease in model accuracy when applying the system to clinics from different locations. This is likely due to variations in diagnostic writing styles among veterinarians. To mitigate this issue, the model would need to be retrained or fine-tuned using data from new clinics, allowing it to adapt as the registry expands.

### Maintainability {#maintainability}

To ensure long-term maintainability, the development team will adhere to modern software engineering principles such as **SOLID** and **KISS**. Comprehensive documentation will be provided to allow future developers or clients to update, debug, and extend the platform as needed.

## Social/Legal Aspect of the Product  {#social/legal-aspect-of-the-product}

The cancer registry for cats and dogs provides meaningful social value by offering researchers in the veterinary field access to an interactive dashboard with a geospatial interface. This platform would represent the first comprehensive cancer registry for cats and dogs in California. By centralizing and visualizing veterinary cancer data, the registry would support researchers in identifying spatial and temporal patterns in cancer cases among companion animals. Additionally, the registry may enable exploratory analysis of potential correlations between cancer trends in cats and dogs and those observed in human populations, contributing to broader public and comparative health research.

From a legal perspective, the client will retain full ownership and rights to all data used for model training and analysis. All records included in the registry will be de-identified prior to use, and no personally identifiable patient information will be made publicly accessible under any circumstances. This ensures compliance with data privacy expectations while allowing the data to be used responsibly for research purposes.

## Glossary of Terms {#glossary-of-terms}

- **API (Application Programming Interface) \-** A set of rules that allows different software components to communicate with each other.  
- **Angular \-** A development platform, built on TypeScript. As a platform, Angular includes: A component-based framework for building scalable web applications. A collection of well-integrated libraries that cover a wide variety of features, including routing, forms management, client-server communication, and more.  
- **Asynchronous Processing \-** A technique where long running tasks are executed separately so the main application remains responsive.  
- **BERT (Bidirectional Encoder Representations from Transformers) \-** A machine learning model used for understanding and extracting information from natural language text.  
- **Bun \-** An all-in-one toolkit for JavaScript and TypeScript apps.  
- **Elastisearch \-** A distributed search and analytics engine, scalable data store and vector database optimized for speed and relevance on production-scale workloads.  
- **FastAPI \-** A python-based web framework used to create backend APIs, well suited for data-driven applications.  
- **GeoPandas \-** An open-source Python library that extends pandas data structures for working with geospatial vector data.  
- **GitHub Projects \-** An adaptable collection of items that you can view as a table, a kanban board, or a roadmap and that stays up-to-date with GitHub data.  
- **GraphQL \-** An open-source query language and server-side runtime that specifies how clients should interact with application programming interfaces (APIs).  
- **MongoDB \-** An open source, nonrelational database management system (DBMS) that uses flexible documents instead of tables and rows to process and store various forms of data.  
- **NLP (Natural Language Processing) \-** A field of artificial intelligence focused on extracting meaning from human language text.  
- **NLP Worker \-** A separate processing component responsible for running NLP models asynchronously without blocking user requests.  
- **Node.js \-** An open-source, cross-platform JavaScript runtime environment.  
- **PostGIS \-** A PostgreSQL extension that adds support for geographic objects and spatial queries.  
- **PostgreSQL \-** An open-source relational database used to store structured application data.  
- **React \-** A JavaScript framework used to build interactive web user interfaces.  
- **REST (Representational State Transfer) \-** A common architectural style for web APIs that uses standard HTTP methods to exchange data between a client and server.  
- **Trello \-** A web-based, kanban-style, list-making application for team project management.

## References {#references}

[Take C.H.A.R.G.E](https://takechargeregistry.com/)  
[VMDB](https://vmdl.missouri.edu/)  
[SAVSNET](https://www.liverpool.ac.uk/savsnet/)  
[Swiss Canine and Feline Cancer Registry](https://www.zora.uzh.ch/entities/publication/cdca5d34-1aad-4a6f-a73a-53cf232e53a8)  
[ACARCinom](https://veterinary-science.uq.edu.au/australian-companion-animal-registry-cancers)  
[UCSF Catchment Area Dashboard](https://cancer.ucsf.edu/catchment-area-dashboard)

[image1]: <data:image/png;base64,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>