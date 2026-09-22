# Dataset

Place the UCI Air Quality CSV file here as:

```text
data/AirQualityUCI.csv
```

The dataset is intentionally not redistributed in this repository. The training
script handles the original UCI column names, replaces `-200` with missing
values, linearly interpolates input features, and drops samples whose target is
missing according to the released experimental protocol.
