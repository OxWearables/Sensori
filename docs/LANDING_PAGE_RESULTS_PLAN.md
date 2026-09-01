# Sensori landing page: results-section plan

## Purpose

Extend the homepage after the existing hero and study-scale band so that it
makes a concise, visual argument for the paper. Preserve the current editorial
style and keep the page lightweight: it should guide readers into the paper,
not reproduce it.

No site implementation is included in this plan.

## Existing structure to preserve

```text
Hero
Study-scale metrics
Main results: four coloured cards
Footer
```

Keep the hero, the metrics band and the four-colour visual language. The work
is a focused extension of `index.html`, not a homepage redesign.

## Proposed homepage narrative

```text
What is Sensori?
  -> Why it stands out in the literature
  -> What was evaluated at scale
  -> What the results look like in practice
  -> Paper and resources
```

### 1. Main-results opening: literature landscape

Place this at the start of the Main Results area, before the four cards. It
should establish the paper's contribution before readers encounter individual
outcomes.

Use two calm, side-by-side 2D plots:

- **Diagnosis / prevalent disease:** x-axis = observation or recording time
  scale; y-axis = number of disease conditions with improved classification.
- **Prognosis / incident disease:** x-axis = observation or recording time
  scale; y-axis = number of conditions with improved future-risk prediction.

Show prior work as subdued reference points and highlight Sensori clearly. The
plots need a complete, citable comparison table before implementation. Do not
infer a prior study's time horizon, evaluation definition, condition count, or
comparability from its title or abstract.

Suggested supporting headline: **A day of movement reaches further.**

### 2. Evidence index: retain the four coloured cards

Keep the existing four cards, but reduce their vertical weight slightly and
add a small purpose-built graphic to each. They should serve as a compact index
of the study rather than four large text panels.

| Card | Content | Small visual |
| --- | --- | --- |
| Comparisons | Baselines evaluated: handcrafted features, general-purpose time-series foundation models, and domain-specific movement models. Name models only after source-level verification. | A simple three-tier comparison key or ranked benchmark marker. |
| 18 behavioural traits | Make the trait groups explicit: physical activity (6), steps (5), sleep (7). | Three grouped bands or a 6/5/7 composition. |
| Four population cohorts | UK Biobank, China Kadoorie Biobank, ELSA and NHANES; United Kingdom, China and United States. | Minimal geography/cohort connection diagram. |
| Disease outcomes | Retain `52/102` prevalent and `26/87` incident outcomes. Use precise labels: classification and six-year risk prediction. | Paired diagnostic/prognostic markers or bars. |

Avoid trying to make all four cards equally explanatory in prose. The graphics
should provide the density that the current cards lack.

### 3. Closer look: three selectively chosen result graphics

Add a new section after the evidence index. It should have three graphical
illustrations and no attempt to exhaustively cover the paper.

1. **Activity recognition**
   - A compact comparison across PAMAP2, RealWorld, WISDM and CAPTURE-24.
   - Use the manuscript's participant-wise cross-validation results and label
     the performance measure clearly (Cohen's kappa; macro F1 only if used and
     sourced from the corresponding supplementary result).
   - Communicate that Sensori was highest on PAMAP2 and RealWorld, and second
     on WISDM and CAPTURE-24.

2. **Across cohorts**
   - A matrix is preferable to a conventional line/bar chart.
   - Show UKB, CKB, ELSA and NHANES against age, sex and BMI; optionally add
     the harmonised health axes as a separate small panel rather than cramming
     them into the same chart.
   - Clearly distinguish metric types: AUROC for sex and Pearson's r for age
     and BMI. Do not use a shared quantitative colour scale across these
     different metrics without an explicit normalisation rationale.

3. **One disease example: prevalent disease classification**
   - Use a single high-signal neurological or psychiatric condition, selected
     after deciding which result best serves the narrative. Do not default to
     Parkinson's disease.
   - Candidates supported by strong manuscript results include multiple
     sclerosis, essential tremor, schizophrenia-spectrum disorders and bipolar
     affective disorder/mania. Select one only after checking the exact
     condition label, estimate and confidence interval in the final source.
   - Show clinical baseline versus clinical plus Sensori.
   - Frame this accurately as an improvement in prevalent-disease
     classification in the UK Biobank test set, not a diagnostic product or a
     clinical claim.
   - Include the paired-bootstrap 95% CI where an effect estimate is shown.

Suggested section heading: **A closer look at the signal.**

## Research facts currently supported by `sn-article.tex`

- Sensori evaluates minute-level transfer on PAMAP2, RealWorld, WISDM and
  CAPTURE-24.
- It models 18 device-measured traits: six physical-activity, five step and
  seven sleep traits.
- Cross-cohort evaluation includes UKB, CKB, ELSA and NHANES; external
  transfer is without refitting where outcomes are available.
- Adding Sensori to common clinical covariates significantly improved
  prevalent classification for 52 of 102 eligible conditions and six-year
  incident risk prediction for 26 of 87 eligible conditions.
- Strong prevalent-classification improvements are reported for neurological
  and psychiatric conditions. Candidate close-ups include multiple sclerosis,
  essential tremor, schizophrenia-spectrum disorders and bipolar affective
  disorder/mania; select the final example only from manuscript-verified
  estimates and paired-bootstrap 95% CIs.

## Data and copy guardrails

- Resolve the scale-number discrepancy before reusing homepage metrics. The
  homepage currently displays 130,271 participants and 729,824 person-days;
  the supplied manuscript's post-quality-control result is 122,640
  participants and 683,617 person-days.
- Separate discovery, classification, prognosis and clinical translation in
  both labels and copy.
- Do not claim external disease validation: disease analyses are UKB-only in
  the current manuscript.
- Do not turn association/prediction findings into individual diagnosis,
  screening, or care claims.
- Every displayed estimate needs an appropriate metric label and uncertainty
  statement where applicable.
- Literature landscape comparisons must be documented in a source-data file
  with paper citation, cohort, input time scale, outcome type, condition count,
  and notes on comparability.

## Implementation direction

- Use native HTML/CSS and inline SVG or carefully accessible canvas graphics;
  do not paste miniature manuscript figures into the homepage.
- Reuse the existing palette, rounded panels, typography and restrained motion.
- Keep each graphic understandable with a concise caption and accessible text
  alternative.
- Keep the new code local to `index.html`, `_sass/_components.scss`, and (only
  if animation is needed) `assets/js/main.js`.
- Preserve reduced-motion support.

## Recommended implementation order

1. Create and verify a small source-data table for the literature landscape.
2. Resolve the participant/person-day source of truth.
3. Build the section scaffolding and responsive layout without final data.
4. Refine the four cards with their compact graphics.
5. Implement the three closer-look graphics from manuscript-verified values.
6. Check desktop and narrow mobile layouts; inspect caption legibility,
   overflow and reading order.
7. Run JavaScript syntax checking and a Jekyll build using Homebrew Ruby, then
   check generated internal links.

## Relevant files

- Homepage markup: `index.html`
- Homepage styles: `_sass/_components.scss`
- Tokens: `_sass/_tokens.scss`
- Existing motion visual and reduced-motion behaviour: `assets/js/main.js`
- Current research source: `/Users/hangy/Dphil/writing/sensori/sn-article.tex`
