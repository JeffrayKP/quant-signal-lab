# Model card

## Intended use

Cross-sectional equity research, portfolio-construction demonstrations, and risk education. The
system is not intended for automated execution, personalized advice, or guaranteed forecasting.

## Models and targets

The classifier estimates whether an equity's adjusted price return will be positive over the
selected horizon. The regressor estimates the magnitude of that return. Both use historical price,
momentum, volume, benchmark-relative, and risk features.

## Validation and evidence gate

Validation folds are based on unique calendar dates across the entire security panel. Each fold
uses only earlier dates for training and applies a horizon-length embargo. Probability calibration
uses out-of-fold predictions. Brier score and return MAE are compared with naive historical
baselines. The evidence gate requires both forecast components to beat their baselines before the
dashboard uses the label `Validated edge`. Point forecasts remain separate from the reliability
indicator so weak validation evidence does not erase cross-sectional differences.

## Known limitations

- The default security list produces survivorship and selection bias.
- Yahoo Finance is not an institutional point-in-time database.
- Probability calibration can deteriorate after regime changes.
- Technical features omit fundamental, options, macroeconomic, and intraday information.
- A good historical result does not establish future profitability.
- Multiple experimentation and model-selection effects are not fully corrected.
- The current-selection portfolio comparison is descriptive and selection-biased, not an
  out-of-sample performance claim.
- Current-event relevance is a deterministic context label and does not change the forecast.
- International symbols depend on Yahoo Finance coverage and require the provider's exchange
  suffix. Newly listed stocks can be researched, but limited feature coverage prevents them from
  qualifying as high-conviction signals.

## Monitoring

Before presenting a deployment as current, verify data age, symbol coverage, model performance
against both baselines, calibration by probability bucket, turnover, costs, and regime-specific
performance. When the evidence gate is mixed or negative, present the ranking as exploratory and
do not describe it as a validated investment edge.
