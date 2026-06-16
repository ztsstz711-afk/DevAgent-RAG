# Retrieval Mode Evaluation

This comparison uses the built-in sample documents and does not represent production quality.

| Requested mode | Effective mode | Backend | Hit@1 | Hit@3 | Non-empty | Valid evidence | Unsupported refusal | Fallback |
|---|---|---|---:|---:|---:|---:|---:|---|
| tfidf | tfidf | sklearn_tfidf | 92.3% | 92.3% | 93.3% | 92.3% | 100.0% | none |
| embedding | tfidf | sklearn_tfidf_fallback | 92.3% | 92.3% | 93.3% | 92.3% | 100.0% | test embedding unavailable |
| hybrid | tfidf | sklearn_tfidf_fallback | 92.3% | 92.3% | 93.3% | 92.3% | 100.0% | test embedding unavailable |
