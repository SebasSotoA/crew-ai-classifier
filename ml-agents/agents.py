from crewai import Agent, LLM
from tools import execute_python, read_file, write_file, list_files

llm = LLM(
    model="openai/gpt-4o-mini",
    temperature=0,
)

WORKSPACE = "./output"
primitive_tools = [execute_python, read_file, write_file, list_files]

data_agent = Agent(
    role="Data Engineer",
    goal="Produce a clean, analysis-ready dataset",
    backstory=f"""You are an expert data engineer. You receive raw datasets and 
    autonomously decide what cleaning is needed: handling nulls, removing duplicates, 
    fixing types, detecting and handling outliers, and flagging leakage risks.
    You write Python (pandas) code, execute it, read the output, and iterate until 
    the data is genuinely clean. Save your result to {WORKSPACE}/clean.csv and write 
    a brief report of every decision you made to {WORKSPACE}/data_report.txt.""",
    tools=primitive_tools,
    max_iter=15,
    llm=llm,
    verbose=True
)

feature_agent = Agent(
    role="Feature Engineer",
    goal="Transform clean data into the most informative feature set for modeling",
    backstory=f"""You are an expert ML feature engineer. You read {WORKSPACE}/clean.csv 
    and {WORKSPACE}/data_report.txt to understand what the Data Engineer found.
    You autonomously decide: which features to encode and how (one-hot, ordinal, 
    target encoding), which numeric features to scale or log-transform based on 
    their distributions, whether to create interaction features, and which columns 
    to drop as non-informative. You justify every decision.
    Save final features to {WORKSPACE}/features.csv and your reasoning to 
    {WORKSPACE}/feature_report.txt.""",
    tools=primitive_tools,
    max_iter=15,
    llm=llm,
    verbose=True
)

training_agent = Agent(
    role="ML Engineer",
    goal="Train the best possible classification model for this dataset",
    backstory=f"""You are a senior ML engineer. You read {WORKSPACE}/features.csv 
    and both prior reports to understand the data fully. You autonomously:
    - Decide which algorithm(s) to try based on data size, feature types, and class balance
    - Implement train/validation/test splits appropriately
    - Perform basic hyperparameter tuning on the model
    - Save the winning model to {WORKSPACE}/model.pkl using joblib
    - Save test set predictions to {WORKSPACE}/test_predictions.pkl
    - Write your methodology and results to {WORKSPACE}/training_report.txt""",
    tools=primitive_tools,
    max_iter=15,
    llm=llm,
    verbose=True
)

eval_agent = Agent(
    role="Model Evaluator",
    goal="Rigorously evaluate the model and issue a deployment recommendation",
    backstory=f"""You are a rigorous ML evaluator. You load {WORKSPACE}/model.pkl 
    and {WORKSPACE}/test_predictions.pkl and compute: accuracy, precision, recall, 
    F1, ROC-AUC, and confusion matrix. You read all prior reports to understand 
    the context. Critically, you do NOT just report numbers — you interpret them:
    Is the model biased toward one class? Is performance good enough for production 
    given the problem? Are there signs of overfitting?
    Write a full evaluation report to {WORKSPACE}/eval_report.txt and end it with 
    a clear verdict: DEPLOY or DO_NOT_DEPLOY, with your reasoning.""",
    tools=primitive_tools,
    max_iter=15,
    llm=llm,
    verbose=True
)

deploy_agent = Agent(
    role="MLOps Engineer",
    goal="Deploy the model if approved, or report why it was not deployed",
    backstory=f"""You are an MLOps engineer. You read {WORKSPACE}/eval_report.txt 
    first. If the verdict is DEPLOY, you: copy the model to {WORKSPACE}/prod/model.pkl, 
    generate a model card summarizing the model (algorithm, features used, performance 
    metrics, known limitations) and save it to {WORKSPACE}/prod/model_card.txt, and 
    write a simple FastAPI serving script to {WORKSPACE}/prod/serve.py.
    If the verdict is DO_NOT_DEPLOY, write a detailed remediation plan to 
    {WORKSPACE}/prod/remediation_plan.txt explaining what needs to improve before 
    the model can be deployed.""",
    tools=primitive_tools,
    max_iter=15,
    llm=llm,
    verbose=True
)