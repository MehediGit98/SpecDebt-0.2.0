# Reference data provenance

The CSV files are retained from the user-supplied source archive. Exact SHA-256 and supplied-table checksums are recorded in results/run_metadata.json.

- CIE 2019, CIE spectral luminous efficiency for photopic vision, DOI https://doi.org/10.25039/CIE.DS.dktna2s3 ; original CIE 018:2019 Table 1. Official metadata: https://files.cie.co.at/Publications-datasets/CIE_sle_photopic.csv_metadata.json
- CIE 2018, CIE alpha-opic action spectra, DOI https://doi.org/10.25039/CIE.DS.vqqhzp5a ; original CIE S 026:2018 Table 2. Official metadata: https://files.cie.co.at/Publications-datasets/CIE_a-opic_action_spectra.csv_metadata.json
- e490_00a_amo.csv is the supplied ASTM E490 extraterrestrial spectrum. convert_e490_to_nm.py produces e490_00a_amo_nm.csv by explicit wavelength-unit conversion. The supplied raw-to-converted transformation is reproducible; the raw solar file was not independently reauthenticated against an external download.

The CIE table checksums were checked against official metadata. Data-owner terms apply; this package does not relicense third-party datasets. Spectral interpolation uses a 380–780 nm grid at 5 nm spacing. Embedded daylight-basis functions are used; the official_daylight_components provenance flag is false.
