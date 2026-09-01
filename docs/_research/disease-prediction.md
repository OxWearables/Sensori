---
title: Diagnosing and predicting disease
subtitle: Disease-focused representations add information beyond clinical covariates and conventional wearable-derived behavioural measures.
summary: Organ-level supervised fine-tuning strengthens movement-related disease signals for both prevalent and incident conditions.
kicker: Disease
number: "04"
order: 4
accent: lime
signal: [12, 22, 48, 36, 78, 60, 82, 43, 69, 88, 54, 27]
hero_label: Improved outcomes
hero_stat: "52 + 26"
hero_caption: prevalent classifications and incident risk predictions
tags: [Disease, Supervised fine-tuning, Prognosis]
back_url: /research/
back_label: All research
---

## Movement as a disease phenotype

Many diseases alter daily life before their effects are fully captured in a clinic: gait changes, sleep becomes fragmented, routines narrow or activity is redistributed across the day. We tested whether Sensori could consolidate these diffuse patterns into useful disease representations.

## Fine-tuning shared structure

Disease outcomes were defined using CALIBER phenotypes. To strengthen disease-related signal without fitting an entirely separate model for every condition, related diseases were grouped by organ system during supervised fine-tuning. The convolutional feature extractor was frozen while the projection, transformer and organ-level readout were updated. Fixed participant-level embeddings were then evaluated with logistic regression or Cox models.

## Beyond established risk factors

The comparison was incremental. Sensori was added to models containing common clinical covariates and compared with both basic acceleration statistics and conventional wearable-derived behavioural measures.

Sensori improved prevalent disease classification for 52 of 102 eligible conditions and six-year incident disease risk prediction for 26 of 87 eligible conditions. The largest gains were observed for neurological and psychiatric disorders.

## What the result does—and does not—show

These are population-level evaluations, not evidence that Sensori is ready for clinical deployment. Prodromal disease and healthy-volunteer bias remain important limitations. The findings instead establish a more focused point: continuous wrist movement contains disease-relevant information that is not exhausted by common clinical variables or predefined behavioural summaries.
