# Pipeline Flowcharts

These diagrams show the strongest documented path only:
- LLM annotation
- Contrastive backbone adaptation
- PresenceClassifier-based production scoring
- Standard evaluation against verified labels

This keeps the focus on architecture rather than implementation details like caching,
fallback modes, or experimental branches.

---

## 1. Annotation Pipeline

Best documented supervision path: `LLM annotation`.

```mermaid
flowchart TD
    subgraph IN1["Input"]
        A["diagnoses.csv<br/>Diagnosis text"]
    end

    subgraph P1["Process"]
        B["Detect cancer-signal rows"]
        C["Exact label match"]
        D["Fuzzy label match"]
        E["Ollama LLM label resolution"]
    end

    subgraph OUT1["Output"]
        F["llm_annotation.csv<br/>Verified term, group, ICD code"]
        G["Training supervision"]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
```

---

## 2. Training Pipeline

Best documented training path: `Contrastive PetBERT adaptation -> PresenceClassifier training`.

```mermaid
flowchart TD
    subgraph IN2["Input"]
        A["report.csv"]
        B["llm_annotation.csv / verified labels"]
    end

    subgraph P2["Process"]
        C["Create case-level train/test split"]
        D["Contrastive fine-tune"]
        E["Train PresenceClassifier"]
        F["Testing data branch<br/>Hold out test split for evaluation"]
    end

    subgraph OUT2["Output"]
        G["contrastive-adapted PetBERT model"]
        H["presence_classifier_best.pt"]
        I["Held-out testing data for evaluation"]
    end

    A --> C
    B --> C
    C -->|"training data"| D
    D --> G
    D -->|"training data"| E
    E --> H
    C -->|"testing data"| I
```

---

## 3. Production Pipeline

Best documented inference path: `Contrastive-adapted PetBERT + PresenceClassifier`.

```mermaid
flowchart TD
    subgraph IN3["Input"]
        A["report.csv<br/>Report text columns"]
        C["labels.csv<br/>Taxonomy labels"]
    end

    subgraph P3["Process"]
        B["Embed each report column with adapted PetBERT"]
        D["Embed each label with adapted PetBERT"]
        E["Concatenate report-column embeddings"]
        F["Prepare label embeddings"]
        G["PresenceClassifier scores each case-label pair"]
        H["Rank labels by score"]
        I["Select top predictions"]
        J["Project label index to term, group, ICD code"]
    end

    subgraph OUT3["Output"]
        K["predictions.csv"]
    end

    A --> B
    C --> D
    B --> E
    D --> F
    E --> G
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
```

---

## 4. Evaluation Pipeline

Standard evaluation path used to measure model quality.

```mermaid
---
config:
  theme: redux
---
flowchart TB
 subgraph IN4["Input"]
        A["predictions.csv"]
        B["llm_annotation.csv / verified labels"]
  end
 subgraph P4["Process"]
        C["Compare predicted labels to verified labels"]
        D{"Case verdict"}
        E["Good<br>Exact term match"]
        F["Slightly Off<br>Correct group, wrong term"]
        G["Completely Off<br>Wrong group"]
        H["false_positive"]
        I["false_negative"]
  end
 subgraph OUT4["Output"]
        J["evaluation.csv"]
        L["evaluation_history.csv"]
  end
    A --> C
    B --> C
    C --> D
    D --> E & F & G & H & I
    E --> J
    F --> J
    G --> J
    H --> J
    I --> J
    J --> L
```
