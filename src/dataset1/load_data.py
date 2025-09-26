import pandas as pd
import gzip

def load_expr_and_labels(
    data_path="C:/Users/ccz/Documents/pulpitis_ml/data/GSE77459_series_matrix.txt.gz"
):
    expr_raw = pd.read_csv(
        data_path,
        sep="\t",
        comment="!",
        compression="gzip",  # <--- this line handles the .gz
        header=0,
        dtype=str
    )

    # Rename ID_REF to gene_probe
    expr_raw = expr_raw.rename(columns={"ID_REF": "gene_probe"})

    # Transpose for ML
    X = expr_raw.set_index("gene_probe").T

    meta = {}
    with gzip.open(data_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("!Sample_source_name_ch1"):
                meta["tissue"] = line.strip().split("\t")[1:]
            elif "pain intensity" in line and line.startswith("!Sample_characteristics_ch1"):
                meta["pain_intensity"] = line.strip().split("\t")[1:]
            elif line.startswith("!Sample_geo_accession"):
                meta["gsm"] = [s.strip('"') for s in line.strip().split("\t")[1:]]
            elif line.startswith("!series_matrix_table_begin"):
                break  # stop reading, expression table begins

    labels = pd.DataFrame({
        "patient_sample": meta["gsm"],
        "tissue": meta["tissue"],
        "pain_intensity": meta["pain_intensity"]
    }).set_index("patient_sample")

    labels['pain_intensity'] = labels['pain_intensity'].apply(
        lambda x: "mild (≤30mm)" if "mild" in x.lower()
        else ("severe (>30mm)" if "severe" in x.lower() else "none (=0mm)")
    )
    labels["tissue"] = labels["tissue"].str.strip('"')


    return X, labels, expr_raw
