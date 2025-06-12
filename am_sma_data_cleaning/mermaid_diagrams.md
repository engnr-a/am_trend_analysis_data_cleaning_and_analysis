```mermaid
classDiagram
    %% Flow relationships
    DataCollection --> Exploratory Data Analysis (Pre-cleaning)
    Exploratory Data Analysis (Pre-cleaning) --> CleaningPreprocessing
    CleaningPreprocessing --> Exploratory Data Analysis (Post-cleaning)
    Exploratory Data Analysis (Post-cleaning) --> Analysis Using LLM
    Analysis Using LLM --> Evaluation and Comparison
    Evaluation and Comparison --> Review

    %% Stage definitions
    class DataCollection {
      - Scrape tweets via Selenium  
      - Keyword‐driven filtering  
      - Store raw CSV batches  
    }

    class Exploratory Data Analysis (Pre-cleaning) {
      - Volume over time  
      - Keyword frequencies  
      - Language & spam checks  
    }

    class CleaningPreprocessing {
      - Bot & spam filtering  
      - Near‐duplicate removal  
      - Text normalization  
    }

    class Exploratory Data Analysis (Post-cleaning) {
      - Updated volume & engagement  
      - Thematic coherence checks  
    }

    class Analysis Using LLM {
      + Text Classification  
        • Business  
        • Technological  
        • Use‐case  
      + Sentiment Analysis  
        • Positive  
        • Neutral  
        • Negative  
    }

    class Evaluation and Comparison {
      - Compare to Gartner Hype Cycle  
      - Correlate with Wohlers Report  
    }

    class Review {
      - Interpret results  
      - Draft methodology chapter  
    }

```



```mermaid
classDiagram
  %% Flow relationships
  DataCollection --> ExploratoryDataAnalysisPreCleaning
  ExploratoryDataAnalysisPreCleaning --> CleaningPreprocessing
  CleaningPreprocessing --> ExploratoryDataAnalysisPostCleaning
  ExploratoryDataAnalysisPostCleaning --> AnalysisUsingLLM
  AnalysisUsingLLM --> EvaluationAndComparison
  EvaluationAndComparison --> Review

  %% Stage definitions
  class DataCollection {
    - Scrape tweets via Selenium
    - Keyword ‐ driven filtering
    - Store raw CSV batches
  }

  class ExploratoryDataAnalysisPreCleaning {
    - Volume over time
    - Keyword frequencies
    - Language & spam checks
  }

  class CleaningPreprocessing {
    - Bot & spam filtering
    - Near ‐ duplicate removal
    - Text normalization
  }

  class ExploratoryDataAnalysisPostCleaning {
    - Updated volume & engagement
    - Thematic coherence checks
  }

  class AnalysisUsingLLM {
    + Text Classification
      • Business
      • Technological
      • Use ‐ case
    + Sentiment Analysis
      • Positive
      • Neutral
      • Negative
  }

  class EvaluationAndComparison {
    - Compare to Gartner Hype Cycle
    - Correlate with Wohlers Report
  }

  class Review {
    - Interpret results
    - Draft methodology chapter
  }

```


# Post LLM Review and Cleaning
```mermaid
flowchart LR
    A[Original Tweets Dataset] --> B[LLM Inference: Initial Classification]
    B --> C[Stage One: Standardization & Cleaning]
    C --> D{Identify Residual Inaccuracies}
    D -->|Yes| E[Re-pass Inaccurate Records to LLM]
    E --> F[LLM Re-Inference: Corrected Classification]
    F --> G[Merge Corrected Results with Stage One Output]
    G --> H[Stage Two: Final Validation & Cleaning]
    H --> H1[Remove Hallucinated Tweet IDs]
    H --> H2[Remove Row-Number Mismatches]
    H --> H3[Convert Remaining Invalid Categories to N/A]
    H1 --> I[Final Cleaned Dataset]
    H2 --> I
    H3 --> I
    D -->|No| H

```