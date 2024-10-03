import jsonlines
import lamini
import argparse
from util.llama_prompt import make_llama_3_prompt
from util.db_util import get_schema


def make_question(obj):
    system = (
        "You are an expert SQL query generator. You are provided with a database schema and a user's question about the data. Your task is to generate an accurate SQL query based on the schema to answer the user's question. "
        "Your task is to generate an accurate SQL query based on this schema to answer the user's question.\n\n"
        f"Here is database schema:\n{get_schema()}\n\n"
        "Follow the feedback provided to imporve the understanding on generating sql queries and provide valid SQL for the user's question.\n\n"
    )
    user = obj["question"]
    return {"system": system, "user": user}


def load_training_data(path, skip_template=False):
    with jsonlines.open(path) as reader:
        for index, obj in enumerate(reversed(list(reader))):
            response = make_question(obj)
            system = response["system"]
            user = response["user"]
            yield {
                "input": make_llama_3_prompt(system, user),
                "feedback": obj["feedback"]
                "output": obj["sql"] + "<|eot_id|>",
            }


def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(
        description="A simple argument parser for training SQL query generator.")

    # Add arguments
    parser.add_argument("--lamini_api_key", type=str,
                        required=True, help="Lamini API Key.")
    parser.add_argument("--training_data_path", type=str,
                        required=True, help="Path to the training data.")
    parser.add_argument("--model_name", type=str,
                        default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="Model ID.")
    parser.add_argument("--learning_rate", type=float,
                        default=1e-4, help="Learning rate for training.")
    parser.add_argument("--max_steps", type=int, default=600,
                        help="Maximum training steps.")

    args = parser.parse_args()

    lamini.api_key = args.lamini_api_key
    llm = lamini.Lamini(model_name=args.model_name)

    dataset = list(load_training_data(
        path=args.training_data_path))

    # Train the model
    llm.train(
        data_or_dataset_id=dataset,
        finetune_args={
            "learning_rate": args.learning_rate,
            "max_steps": args.max_steps,
        }
    )


if __name__ == "__main__":
    main()
