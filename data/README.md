# Data

This project uses the **Household Income and Expenditure Survey (HIES) 2019** microdata
(100% sample, CSV format) published by the **Department of Census and Statistics (DCS), Sri Lanka**.

The raw files are **not committed** to this repository: they are official survey microdata and
redistribution is governed by DCS. To reproduce the results:

1. Obtain the HIES 2019 CSV files from DCS (http://www.statistics.gov.lk - Household Income and
   Expenditure Survey 2019 data / microdata request).
2. Copy the CSV files into `data/raw/` (file names exactly as released, e.g.
   `HH_expenditure_hh_Income.csv`, `SEC_1_DEMOGRAPHIC.csv`, `SEC_8_HOUSING.csv`, ...).
3. Run `python scripts/eda.py`, `python scripts/tune.py`, `python scripts/train.py`,
   `python scripts/explain.py` (see the main README).

Files used: `HH_expenditure_hh_Income`, `SEC_1_DEMOGRAPHIC`, `SEC_5_5_1_OTHER_INCOME` (Samurdhi, audit only),
`SEC_6A_DURABLE_GOODS`, `SEC_7_BASIC_FACILITIES`, `SEC_8_HOUSING`, `SEC_9_LAND_ANIMAL`,
`SECTION_9_IS_A_LAND`, `SECTION_9_2_OWNED_LIVESTOCKS`.

Key ID fields: `psu` (primary sampling unit), `snumber` (housing unit), `hhno` (household).
Weight: `finalweight` (household design weight).

**Privacy.** The microdata contain no names, addresses or contact details. The repository ships
only the trained model and aggregate result tables; one held-out household's feature vector is
included as a demo input (`app/sample_household.json`), containing no identifiers.
