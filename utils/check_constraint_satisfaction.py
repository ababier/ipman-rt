import pandas as pd
from open_kbp import DataLoader
from oracle import Oracle

from historical_plan_bounds import HistoricalPlanBounds


def evaluate_clinical_constraints(data_name: str) -> pd.DataFrame:
    train_data_loader = DataLoader(data_name, batch_size=1)
    historical_plan_bounds = HistoricalPlanBounds.get(train_data_loader)
    oracle = Oracle(historical_plan_bounds, alternative_criteria=False)
    patient_ids = list(train_data_loader.paths_by_patient_id.keys())
    criteria = pd.DataFrame(index=patient_ids, columns=train_data_loader.full_roi_list)
    for batch in train_data_loader.get_batches():
        criteria.loc[batch.patient_list] = oracle.evaluate_criteria(batch.dose, batch)
    return criteria


if __name__ == "__main__":
    train_criteria = evaluate_clinical_constraints("train")
    validation_criteria = evaluate_clinical_constraints("validation")
    test_criteria = evaluate_clinical_constraints("test")
    all_criteria = pd.concat((train_criteria, validation_criteria, test_criteria))

    criteria_satisfaction_rates = all_criteria.where(all_criteria.isna(), all_criteria >= 0).mean()
    print(f"In this dataset, the criteria satisfaction rates were: \n{criteria_satisfaction_rates}")
