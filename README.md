# Multimodal & Privacy-Preserving ML for Human Behavior Analysis

## Overview
This repository explores **representation-level privacy** in multimodal human-behavior analysis.  
We study how to **retain utility** (emotion / expression recognition) while **suppressing sensitive attributes** (gender) in intermediate embeddings for **speech** and **face** modalities.

## Task
- **Problem:** Intermediate embeddings can leak private attributes (e.g., gender).  
- **Goal:** Train encoders that are **useful for the main task** but **uninformative about sensitive attributes**.  
- **Approach:** Use a **Gradient Reversal Layer (GRL)** branch to adversarially remove gender information while preserving emotion/expression recognition.

## Datasets
- **Speech:** IEMOCAP, RAVDESS, CREMA-D (4 emotions, gender labels).  
- **Face:** RAF-DB (4 expressions, gender pseudo-labels).  
- Each modality split into:  
  - `X_main` → train utility model  
  - `X_attacker` → train external attackers  
  - `X_test` → held-out evaluation  

## Evaluation
- **Utility:** Accuracy, Macro-F1 (emotion/expression).  
- **Privacy:** Attacker Unweighted Average Recall (UAR) on gender.  
- **Attackers:** Logistic Regression & shallow MLP probes.  
- **Scenarios:** Baseline, Ignorant, Informed (model), Informed (model+data).

## Key Findings
- GRL reduces gender recoverability toward chance while keeping acceptable utility.  
- Earlier layers leak more; deeper representations are easier to anonymize.  
- Face embeddings contain stronger demographic priors than speech features.  
- Warm-up schedules for λ may stabilize training.

## Structure (to be updated)
