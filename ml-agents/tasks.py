from crewai import Task
from agents import (data_agent, feature_agent, 
                    training_agent, eval_agent, deploy_agent)

WORKSPACE = "./output"
RAW_DATA_PATH = "./data/titanic.csv"
TARGET_COL = "Sobrevivio"

def build_tasks():
    t1 = Task(
        description=f"""Each execute_python call is a completely fresh process — always include all imports and data loading at the top of every call.
        The raw dataset is at {RAW_DATA_PATH}.
        You MUST use the execute_python tool to run all pandas code — never read the 
        raw CSV into your context directly. Use df.head(), df.dtypes, df.describe(), 
        df.isnull().sum(), and df.duplicated().sum() to inspect the data, then write 
        and execute cleaning code. When done, you MUST call execute_python to save the 
        result: df.to_csv('{WORKSPACE}/clean.csv', index=False). You MUST then call 
        write_file to save your decisions to '{WORKSPACE}/data_report.txt'.
        Do not describe what you would do — actually do it with the tools.""",
        expected_output=f"clean.csv and data_report.txt confirmed written to {WORKSPACE}.",
        agent=data_agent
    )

    t2 = Task(
        description=f"""Each execute_python call is a completely fresh process — always include all imports and data loading at the top of every call.
        Use execute_python to read {WORKSPACE}/clean.csv and apply 
        feature engineering: encode all categorical columns, scale numeric columns.
        IMPORTANT: The target column is '{TARGET_COL}'. Keep it in the final 
        dataframe — do NOT drop or encode it, leave it exactly as-is.
        Also check for class imbalance in '{TARGET_COL}' and note it in your report.
        You MUST call execute_python to save the result: 
        df.to_csv('{WORKSPACE}/features.csv', index=False). 
        You MUST call write_file to save your reasoning to '{WORKSPACE}/feature_report.txt'.
        Do not describe what you would do — actually do it with the tools.""",
        expected_output=f"features.csv and feature_report.txt confirmed written to {WORKSPACE}.",
        agent=feature_agent,
        context=[t1]
    )

    t3 = Task(
        description=f"""Each execute_python call is a completely fresh process — always include all imports and data loading at the top of every call.
        Read {WORKSPACE}/features.csv and train a classification model.
        Key facts: the target column is '{TARGET_COL}', use class_weight='balanced' 
        on all classifiers to handle class imbalance, and do an 80/20 train/test split.
        Try at least 2 algorithms, compare them, and keep the best one.
        Save the following to disk using execute_python:
        - best model → {WORKSPACE}/model.pkl  (use joblib.dump)
        - test set predictions → {WORKSPACE}/test_predictions.pkl  (use joblib.dump)
        - test set ground truth → {WORKSPACE}/y_test.pkl  (use joblib.dump — needed for evaluation)
        Then call write_file to save your methodology and results to {WORKSPACE}/training_report.txt.
        Do not describe what you would do — actually do it with the tools.""",
        expected_output=f"model.pkl, test_predictions.pkl, y_test.pkl, and training_report.txt confirmed written to {WORKSPACE}.",
        agent=training_agent,
        context=[t1, t2]
    )

    t4 = Task(
        description=f"""Each execute_python call is a completely fresh process — always include all imports and data loading at the top of every call.
        Load {WORKSPACE}/model.pkl, {WORKSPACE}/test_predictions.pkl, 
        and {WORKSPACE}/y_test.pkl using execute_python, then compute accuracy, precision, 
        recall, F1, and confusion matrix. Interpret the results — is the model biased 
        toward one class? Is it good enough for a production Titanic survival predictor?
        Call write_file to save the full analysis to {WORKSPACE}/eval_report.txt.
        The last line of the file MUST be exactly DEPLOY or DO_NOT_DEPLOY.
        Do not describe what you would do — actually do it with the tools.""",
        expected_output=f"eval_report.txt confirmed written to {WORKSPACE} with final verdict on last line.",
        agent=eval_agent,
        context=[t1, t2, t3]
    )

    t5 = Task(
        description=f"""Each execute_python call is a completely fresh process — always include all imports and data loading at the top of every call.
        Use read_file to read {WORKSPACE}/eval_report.txt and check the 
        last line for the verdict. 
        If DEPLOY: you MUST call execute_python to run: 
        import shutil, os; os.makedirs('{WORKSPACE}/prod', exist_ok=True); 
        shutil.copy('{WORKSPACE}/model.pkl', '{WORKSPACE}/prod/model.pkl'). 
        Then call write_file to write the model card to '{WORKSPACE}/prod/model_card.txt' 
        and the FastAPI serving script to '{WORKSPACE}/prod/serve.py'. 
        If DO_NOT_DEPLOY: you MUST call write_file to write a remediation plan to 
        '{WORKSPACE}/prod/remediation_plan.txt'. 
        Do not describe what you would do — actually do it with the tools.""",
        expected_output=f"Files confirmed written inside {WORKSPACE}/prod/.",
        agent=deploy_agent,
        context=[t4]
    )

    return [t1, t2, t3, t4, t5]
