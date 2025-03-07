import os
import re
import json
import requests
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any


def load_synthea_csvs(folder_path: str) -> Dict[str, pd.DataFrame]:
    files_needed = ["patients.csv", "conditions.csv"]
    dataframes = {}
    for fn in files_needed:
        csv_path = os.path.join(folder_path, fn)
        if not os.path.isfile(csv_path):
            print(f"Warning: '{fn}' not found at '{csv_path}'. Using empty DataFrame.")
            dataframes[fn.replace(".csv", "")] = pd.DataFrame()
        else:
            df = pd.read_csv(csv_path)
            dataframes[fn.replace(".csv", "")] = df
    return dataframes


def compute_age(birthdate_str: str) -> int:
    try:
        birthdate = datetime.strptime(birthdate_str, "%Y-%m-%d")
        return (datetime.now() - birthdate).days // 365
    except (ValueError, TypeError):
        return 0


def build_patient_data(
    df_patients: pd.DataFrame, df_conditions: pd.DataFrame
) -> pd.DataFrame:
    required_cols = ["Id", "BIRTHDATE", "FIRST", "LAST", "GENDER", "CITY", "STATE"]
    missing_cols = [c for c in required_cols if c not in df_patients.columns]
    if missing_cols:
        raise ValueError(
            f"patients.csv is missing columns: {missing_cols}\n"
            f"Found columns: {list(df_patients.columns)}"
        )

    df_patients = df_patients.copy()
    df_patients.rename(columns={"Id": "PATIENT_ID"}, inplace=True)
    df_patients["AGE"] = df_patients["BIRTHDATE"].apply(compute_age)

    if not df_conditions.empty:
        if (
            "PATIENT" not in df_conditions.columns
            or "DESCRIPTION" not in df_conditions.columns
        ):
            raise ValueError(
                "conditions.csv is missing 'PATIENT' or 'DESCRIPTION' columns.\n"
                f"Found columns: {list(df_conditions.columns)}"
            )
        df_conditions = df_conditions.copy()
        df_conditions.rename(columns={"PATIENT": "PATIENT_ID"}, inplace=True)
        cond_agg = (
            df_conditions.groupby("PATIENT_ID")["DESCRIPTION"]
            .apply(lambda x: list(x.dropna().unique()))
            .reset_index()
            .rename(columns={"DESCRIPTION": "CONDITION_LIST"})
        )
        df_patients = df_patients.merge(cond_agg, on="PATIENT_ID", how="left")
    else:
        df_patients["CONDITION_LIST"] = [[] for _ in range(len(df_patients))]

    keep_cols = [
        "PATIENT_ID",
        "FIRST",
        "LAST",
        "GENDER",
        "AGE",
        "CITY",
        "STATE",
        "CONDITION_LIST",
    ]
    final_cols = [c for c in keep_cols if c in df_patients.columns]
    return df_patients[final_cols].copy()


def parse_age_string(age_str: str) -> int:
    if not age_str or age_str.lower().startswith("n/a"):
        return 0
    parts = age_str.split()
    for p in parts:
        if p.isdigit():
            return int(p)
    return 0


def meets_structured_criteria(
    patient_age: int, patient_gender: str, eligibility: Dict[str, Any]
) -> bool:
    min_age_str = eligibility.get("minimumAge", "")
    max_age_str = eligibility.get("maximumAge", "")
    sex_str = eligibility.get("sex", "ALL")

    min_age = parse_age_string(min_age_str)
    max_age = parse_age_string(max_age_str)

    if min_age and patient_age < min_age:
        return False
    if max_age and patient_age > max_age:
        return False

    if sex_str == "FEMALE" and patient_gender.upper() not in ["FEMALE", "F"]:
        return False
    if sex_str == "MALE" and patient_gender.upper() not in ["MALE", "M"]:
        return False

    return True


def quote_for_essie(s: str) -> str:
    if re.search(r"[\s(){}\[\]]", s):
        safe_s = s.replace('"', '\\"')
        return '"' + safe_s + '"'
    return s


def format_age_essie(age: int) -> str:
    return f"{age} Years"


def fetch_ctg_studies(
    condition: str, age: int, status_list: List[str], page_size: int = 20
) -> List[Dict[str, Any]]:
    base_url = "https://clinicaltrials.gov/api/v2/studies"
    joined_status = ",".join(status_list)
    age_str = format_age_essie(age)
    safe_condition = quote_for_essie(condition)

    fields = [
        "NCTId",
        "BriefTitle",
        "protocolSection.eligibilityModule",
        "protocolSection.conditionsModule",
    ]
    params = {
        "format": "json",
        "query.cond": safe_condition,
        "filter.overallStatus": joined_status,
        "filter.advanced": f"AREA[MinimumAge]RANGE[MIN, {age_str}] AND AREA[MaximumAge]RANGE[{age_str}, MAX]",
        "fields": ",".join(fields),
        "pageSize": page_size,
        "countTotal": "true",
    }

    studies = []
    next_token = None
    while True:
        if next_token:
            params["pageToken"] = next_token
        resp = requests.get(base_url, params=params)
        resp.raise_for_status()
        data = resp.json()
        chunk = data.get("studies", [])
        studies.extend(chunk)
        next_token = data.get("nextPageToken")
        if not next_token:
            break
    return studies


def match_patient_to_trials(
    patient_row: pd.Series, statuses: List[str] = None
) -> List[Dict[str, Any]]:
    if statuses is None:
        statuses = ["RECRUITING", "ENROLLING_BY_INVITATION"]

    age = int(patient_row["AGE"])
    gender = str(patient_row["GENDER"])
    patient_conditions = patient_row.get("CONDITION_LIST", [])
    if not isinstance(patient_conditions, list):
        patient_conditions = []

    if not patient_conditions:
        return []

    primary_condition = patient_conditions[0]
    trials_data = fetch_ctg_studies(primary_condition, age, statuses, page_size=20)
    matched = []
    for st in trials_data:
        elig_mod = st.get("protocolSection", {}).get("eligibilityModule", {})
        ok_structured = meets_structured_criteria(age, gender, elig_mod)
        if not ok_structured:
            continue

        cond_mod = st.get("protocolSection", {}).get("conditionsModule", {})
        trial_conditions = cond_mod.get("conditions", []) or []
        trial_conditions_lower = {c.lower() for c in trial_conditions}
        patient_conditions_lower = {c.lower() for c in patient_conditions}
        intersecting_conds = []
        for c in trial_conditions:
            if c.lower() in patient_conditions_lower:
                intersecting_conds.append(c)

        if not intersecting_conds:
            continue

        ident_mod = st.get("protocolSection", {}).get("identificationModule", {})
        nct_id = ident_mod.get("nctId", "")
        brief_title = ident_mod.get("briefTitle", "")

        matched.append(
            {
                "trialId": nct_id,
                "trialName": brief_title,
                "eligibilityCriteriaMet": intersecting_conds,
            }
        )
    return matched


def main():
    folder_path = "csv"
    data = load_synthea_csvs(folder_path)
    df_patients = build_patient_data(data["patients"], data["conditions"])

    all_results = []
    csv_rows = []
    for _, row in df_patients.iterrows():
        patient_id = str(row["PATIENT_ID"])
        matched_trials = match_patient_to_trials(row)

        entry = {"patientId": patient_id, "eligibleTrials": matched_trials}
        all_results.append(entry)

        for mt in matched_trials:
            csv_rows.append(
                {
                    "patientId": patient_id,
                    "trialId": mt["trialId"],
                    "trialName": mt["trialName"],
                    "eligibilityCriteriaMet": "; ".join(mt["eligibilityCriteriaMet"]),
                }
            )

    pd.DataFrame(csv_rows).to_csv("patient_trial_matches.csv", index=False)

    with open("patient_trial_matches.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("Wrote patient_trial_matches.csv and patient_trial_matches.json")


if __name__ == "__main__":
    main()
