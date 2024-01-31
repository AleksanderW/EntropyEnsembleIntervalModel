import os
import os.path as op
from itertools import product

import pandas as pd

from algorithms.Constants import (
    extract_variables,
    set_params,
    empty_logs,
)
from algorithms.Evaluation import evaluate_classifiers
from algorithms.Settings import (
    datasets,
    steps,
    ALG_aggregations,
    ALG_models,
    n_splits,
    random_states,
    classifiers,
    ALG_orders,
    parallel_evaluation,
)
from algorithms.classifiers.Proposed_Model import ProposedModel
from algorithms.classifiers.Proposed_Entropy_Model import ProposedEntropyModel
from algorithms.classifiers.Proposed_Entropy_Groups_Model import ProposedEntropyGroupsModel

path = op.join("..", "results")
if not os.path.exists(path):
    os.makedirs(path)

empty_logs()

selector_file_paths = {
    "RFECV": "{}_RandomState{}_selector{}_step{}_nsplit{}_{}.xlsx",
    "RFE": "{}_RandomState{}_selector{}_step{}_nsplit{}_{}.xlsx",
    "StabilitySelection": "{}_RandomState{}_nsplit{}_{}.xlsx",
}

# for selector in feature_selectors:
#     create_logs(selector)
for step, n_split, random_state in product(steps, n_splits, random_states):
    clfs_loop = classifiers.copy()
    clfs_loop += [
        (
            ProposedModel(
                aggregation_type=agg,
                order=order,
                random_state=random_state,
                models=ALG_models,
            ),
            f"POM_ALG_{agg}_{order}",
        )
        for agg, order in product(ALG_aggregations, ALG_orders)
    ]

    clfs_loop += [
        (
            ProposedEntropyModel(
                aggregation_type=agg,
                order=order,
                random_state=random_state,
                models=ALG_models,
            ),
            f"POM_ENTROPY_{agg}_{order}",
        )
        for agg, order in product(ALG_aggregations, ALG_orders)
    ]
    clfs_loop += [
        (
            ProposedEntropyGroupsModel(
                aggregation_type=agg,
                order=order,
                random_state=random_state,
                models=ALG_models,
            ),
            f"POM_enGROUPS_{agg}_{order}",
        )
        for agg, order in product(ALG_aggregations, ALG_orders)
    ]
    clfs_loop = set_params(clfs_loop, random_state)

    for dataset in datasets:
        # file_path = op.join(
        #     path,
        #     selector_file_paths[selector].format(
        #         dataset, random_state, selector, step, n_split, selector
        #     ),
        # )
        file_path = op.join(path, f"{dataset}_{random_state}_{step}_{n_split}")
        # missing_models = check_if_models_exist(
        #     dataset, random_state, step, selector
        # )

        # if missing_models:
        #     continue
        results = []
        for clf, name in clfs_loop:
            result = evaluate_classifiers(
                clf,
                name,
                dataset,
                n_split,
                step,
                random_state,
                # selector,
                parallel=parallel_evaluation,
            )
            name, subtab, agg, ordr, split = extract_variables(name)
            results.append(
                {
                    "classifier": name,
                    "subtables": subtab,
                    "aggr": agg,
                    "order": ordr,
                    "split": split,
                    "auc": result["auc"][0],
                    "auc_std": result["auc"][1],
                    "acc": result["accuracy"][0],
                    "acc_std": result["accuracy"][1],
                    "bal_acc": result["balanced_accuracy"][0],
                    "bal_acc_std": result["balanced_accuracy"][1],
                    # 'mic_prec': result['precision_mic'][0],
                    # 'mic_prec_std': result['precision_mic'][1],
                    "mac_prec": result["precision_mac"][0],
                    "mac_prec_std": result["precision_mac"][1],
                    # 'wgt_prec': result['precision_wgt'][0],
                    # 'wgt_prec_std': result['precision_wgt'][1],
                    # 'mic_recall': result['recall_mic'][0],
                    # 'mic_recall_std': result['recall_mic'][1],
                    "mac_recall": result["recall_mac"][0],
                    "mac_recall_std": result["recall_mac"][1],
                    # 'wgt_recall': result['recall_wgt'][0],
                    # 'wgt_recall_std': result['recall_wgt'][1],
                    # 'mic_f1': result['f1_score_mic'][0],
                    # 'mic_f1_std': result['f1_score_mic'][1],
                    "mac_f1": result["f1_score_mac"][0],
                    "mac_f1_std": result["f1_score_mac"][1],
                    # 'wgt_f1': result['f1_score_wgt'][0],
                    # 'wgt_f1_std': result['f1_score_wgt'][1]
                    "log_loss": result["log_loss"]
                }
            )

        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(
            by=["auc", "acc", "bal_acc"], ascending=False
        )
        writer = pd.ExcelWriter(file_path+".xlsx", engine="xlsxwriter")
        results_df.to_excel(writer, sheet_name="Results", index=False)
        writer.close()
