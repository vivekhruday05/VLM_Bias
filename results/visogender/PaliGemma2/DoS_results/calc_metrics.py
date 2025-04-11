import json

def calculate_ra_oo(data):
    correct_male, total_male = 0, 0
    correct_female, total_female = 0, 0

    for entry in data.values():
        occ_gender = entry["occ_gender"]
        logits = entry["logits_list_obj"]

        # Check if the highest logit matches the occupation's gender
        predicted_pronoun = max(logits, key=logits.get)
        if occ_gender == "masculine":
            total_male += 1
            if predicted_pronoun == "his":
                correct_male += 1
        elif occ_gender == "feminine":
            total_female += 1
            if predicted_pronoun == "her":
                correct_female += 1

    # Compute Resolution Accuracies
    RA_m = correct_male / total_male if total_male > 0 else 0
    RA_f = correct_female / total_female if total_female > 0 else 0
    RA_avg = (RA_m + RA_f) / 2
    GG = abs(RA_m - RA_f)

    return RA_m, RA_f, RA_avg, GG


def calculate_ra_op(data):
    correct_male, total_male = 0, 0
    correct_female, total_female = 0, 0

    for entry in data.values():
        occ_gender = entry["occ_gender"]
        logits_occ = entry["logits_list_occ_first"]
        logits_par = entry["logits_list_par_first"]

        # Check occupation pronoun prediction
        predicted_occ = max(logits_occ, key=logits_occ.get)
        if occ_gender == "masculine":
            total_male += 1
            if predicted_occ == "his":
                correct_male += 1
        elif occ_gender == "feminine":
            total_female += 1
            if predicted_occ == "her":
                correct_female += 1

    # Compute Resolution Accuracies
    RA_m = correct_male / total_male if total_male > 0 else 0
    RA_f = correct_female / total_female if total_female > 0 else 0
    RA_avg = (RA_m + RA_f) / 2
    GG = abs(RA_m - RA_f)

    return RA_m, RA_f, RA_avg, GG

for method in ["vision_frozen", "text_frozen", "none_frozen", "projections_not_frozen"]:
    # Load JSON files
    print(f"-------------- Method: {method} --------------")
    print()
    model_type = method
    with open(f"captioning_{model_type}_ContextOO.json", "r") as f:
        oo_data = json.load(f)

    with open(f"captioning_{model_type}_ContextOP.json", "r") as f:
        op_data = json.load(f)

    # Compute metrics
    ra_m_oo, ra_f_oo, ra_avg_oo, gg_oo = calculate_ra_oo(oo_data)
    ra_m_op, ra_f_op, ra_avg_op, gg_op = calculate_ra_op(op_data)

    # Print results
    print("OO Results:")
    print(f"RA_m: {ra_m_oo:.4f}, RA_f: {ra_f_oo:.4f}, RA_avg: {ra_avg_oo:.4f}, GG: {gg_oo:.4f}")

    print("\nOP Results:")
    print(f"RA_m: {ra_m_op:.4f}, RA_f: {ra_f_op:.4f}, RA_avg: {ra_avg_op:.4f}, GG: {gg_op:.4f}")
    print()
