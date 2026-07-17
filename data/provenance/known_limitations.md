# Known Limitations

- Public ChEMBL data is heterogeneous.
- Censoring is assay/source-dependent.
- Publication/source year is not true experimental project chronology.
- IC50/Ki/AC50/Potency are not merged.
- Different assay types are not merged.
- Labels are p-scale intervals, not guaranteed true latent values.
- The main modelling subset is dominated by CYP and hERG endpoints; transfer to
  permeability, solubility, clearance, transport, and other ADMET properties
  remains to be established.

## Censoring mechanism and identifiability

The interval-censored likelihood identifies the predictive mean and scale under
a non-informative-censoring assumption: whether and where a measurement is
censored is assumed independent of its latent value conditional on the model
features. This assumption may be violated in ChEMBL because assay choice and
dynamic range can depend on expected potency. Under informative censoring, the
maximum-likelihood estimates can be biased toward the observed side.

The released interval-aware results must therefore be interpreted under the
non-informative-censoring assumption. Synthetic hidden-truth experiments in the
release use prespecified random censoring at 20%, 40%, and 60%. The split
generator implements a separate informative-censoring sensitivity arm, but that
arm is disabled in the version 1.0.0 release and is not part of the reported
results. Selection-model or inverse-probability corrections remain future work.
