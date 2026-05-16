from dotenv import load_dotenv
load_dotenv()

from crewai import Crew, Process
from agents import data_agent, feature_agent, training_agent, eval_agent, deploy_agent
from tasks import build_tasks

def main():
    tasks = build_tasks()

    crew = Crew(
        agents=[data_agent, feature_agent, training_agent, eval_agent, deploy_agent],
        tasks=tasks,
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()
    print("\n=== PIPELINE COMPLETE ===")
    print(result)

if __name__ == "__main__":
    main()