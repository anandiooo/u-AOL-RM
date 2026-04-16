# Research Protocol Template

## Title

Temporal Causal Mental Health Chatbot Using NLP, Personal Knowledge Graph, and Explainable AI

## 1. Objective

- Build and evaluate a proof-of-concept chatbot that captures longitudinal mental health patterns.

## 2. Research Questions

1. How well can NLP extract emotion, symptom, and trigger from longitudinal conversations?
2. Can a Temporal Personal Causal Graph represent cause-effect progression over time?
3. Does XAI graph visualization improve user understanding vs reactive chatbot baseline?
4. How does rule-based risk prediction compare with a GNN-based approach?

## 3. Dataset Plan

- Public mental health conversational datasets.
- Optional adaptation to Indonesian context.
- Annotation dimensions:
  - emotion
  - trigger
  - symptom
  - mechanism
  - timestamp / turn order

## 4. Experimental Design

- Baseline A: reactive chatbot (no causal memory)
- Baseline B: sentiment-only memory
- Proposed: NLP + TPCG + XAI + early risk module

## 5. Metrics (SMART-aligned)

- Emotion detection F1 >= 0.75
- Symptom extraction precision >= 0.70
- Cause-effect validity >= 0.70
- Early warning accuracy >= 0.65
- User insight understanding >= 0.75

## 6. Ethics and Scope

- Non-diagnostic assistant.
- Provide crisis-resource disclaimer.
- Preserve privacy and anonymize conversation records.

## 7. Two-Month Timeline

Week 1-2: dataset curation and labeling schema
Week 3-4: NLP extraction pipeline and baseline evaluation
Week 5-6: TPCG + risk predictor + XAI module
Week 7: user study and questionnaire
Week 8: analysis and final reporting
