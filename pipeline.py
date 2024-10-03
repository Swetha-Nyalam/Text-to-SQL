import argparse
import os
import logging
import lamini
import shutil
import jsonlines
from tqdm import tqdm
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from tabulate import tabulate
from util.prompts import debugger_compilation_fix, debugger_df_mismatch, score_prompt
import util.prompts as prompts
from util.llama_prompt import make_llama_3_prompt
from util.db_util import get_schema, execute_sql, sql_exection_error, validate_sql_termination

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parses command-line arguments for the SQL evaluation pipeline.


def parse_arguments():
    """ Parse command-line arguments """
    parser = argparse.ArgumentParser(
        description="Run the SQL evaluation pipeline")

    # Define the expected arguments
    parser.add_argument('--db_path', type=str, required=True,
                        help='Path to the SQLite database')
    parser.add_argument('--golden_file', type=str, required=True,
                        help='Path to the golden dataset JSONL file')
    parser.add_argument('--db_type', type=str, default="SQLite",
                        help='Database type (default: SQLite)')
    parser.add_argument('--db_name', type=str, default="uber",
                        help='Database name is default uber')
    parser.add_argument("--prompt_types", required=True,
                        help="Comma-separated list of prompt types")
    return parser.parse_args()


# Query Stage


class QueryProcessor:

    def __init__(self, db_path: str, golden_file: str, db_type: str, db_name: str, prompt_type: str,  base_llm, tuned_llm):
        self.db_path = db_path
        self.golden_file = golden_file
        self.db_type = db_type
        self.base_llm = base_llm
        self.tuned_llm = tuned_llm
        self.db_name = db_name
        self.prompt_type = prompt_type
        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.output_file = f"data/results/results_{timestamp}.jsonl"

    def get_output_file(self):
        return self.output_file

    def run_debugger(self, question, db_path, error_sql):
        debugger_response = self.base_llm.generate(
            debugger_compilation_fix(error_sql, sql_exection_error(db_path, error_sql), question, self.db_type, self.db_name), output_type={"sql": "str"}, max_new_tokens=1500)
        return debugger_response["sql"]

    def process_question(self):
        if not os.path.exists(self.golden_file):
            logger.error(f"Input file {self.golden_file} does not exist.")
            return

        os.makedirs('data/results', exist_ok=True)

        # First pass to count the total number of questions
        with jsonlines.open(self.golden_file, mode='r') as reader:
            total_questions = sum(1 for _ in reader)

        # Second pass to process the questions
        with jsonlines.open(self.golden_file, mode='r') as reader, jsonlines.open(self.output_file, mode='w') as writer:
            for q in tqdm(reader, total=total_questions, desc="Processing questions"):
                if "question" in q:
                    obj = {
                        "question": "",
                        "sql": "",
                        "compile_status": True
                    }
                    obj["question"] = q["question"]
                    prompt_func = getattr(prompts, self.prompt_type, None)
                    if prompt_func is not None and callable(prompt_func):
                        prompt = prompt_func(
                            self.db_name, self.db_type, q["question"])
                    else:
                        raise ValueError(
                            f"Invalid prompt type: {self.prompt_type}"
                        )

                    response_query = self.tuned_llm.generate(
                        prompt, output_type={"sql": "str"}, max_new_tokens=1500)
                    SQLQuery = response_query["sql"]

                    SQLQuery = validate_sql_termination(SQLQuery)

                    query_result = execute_sql(args.db_path, SQLQuery)

                    if query_result is None:
                        SQLQuery = self.run_debugger(
                            q["question"], args.db_path, SQLQuery)
                        debugger_query_result = execute_sql(
                            args.db_path, SQLQuery)

                        if debugger_query_result is None:
                            obj["compile_status"] = False

                    obj["sql"] = SQLQuery
                    writer.write(obj)


# Score Stage
class ScoreGenerator:

    def __init__(self, db_type, db_name, db_path, golden_file: str, results_file: str, base_llm):
        self.golden_file = golden_file
        self.results_file = results_file
        self.exec_count = 0
        self.answer_mismatch_queries = []
        self.execution_error_queries = []
        self.base_llm = base_llm
        self.db_path = db_path
        self.db_type = db_type
        self.db_name = db_name

    def validate_dataframes(self, true_sql, model_sql):
        is_match = False
        if true_sql == model_sql:
            return True
        try:
            df_true = execute_sql(self.db_path, true_sql)
            df_model = execute_sql(self.db_path, model_sql)

            if df_model is not None and df_true.values.tolist() == df_model.values.tolist():
                is_match = True

            if not is_match:
                # logger.info(f"Answer mismatch at execution: {self.exec_count}")
                self.answer_mismatch_queries.append(self.exec_count)

        except Exception as e:
            # logger.error(f"SQL compilation or execution error at execution: {self.exec_count}")
            self.execution_error_queries.append(self.exec_count)

        return is_match

    def _read_file(self, file_path: str):
        """Read JSONL file and return list of dictionaries."""
        try:
            with jsonlines.open(file_path) as reader:
                return [obj for obj in tqdm(reader, desc=f"Reading {file_path}")]
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return []

    def _calculate_scores(self, golden_data, generated_data):
        """Calculate scores based on golden data and generated data."""

        # Convert lists to dictionaries for quick lookup
        model_sql_list = [(obj['question'], obj['sql'].strip(), obj.get(
            'compile_status', False)) for obj in generated_data]

        # Initialize counters
        total_queries = len(model_sql_list)
        compiled_queries = sum(
            1 for _, _, status in model_sql_list if status)
        correct_sql_count = 0
        correct_answer_count = 0

        for idx, (golden_entry, generated_entry) in enumerate(zip(golden_data, generated_data)):
            self.exec_count += 1
            golden_sql = golden_data[idx]['sql'].strip()
            model_sql = generated_data[idx]['sql'].strip()
            question = generated_data[idx]['question']

            if model_sql == "":
                continue

            model_sql = validate_sql_termination(model_sql)

            exec_match = self.validate_dataframes(
                golden_sql, model_sql)

            if not exec_match:
                debugger_response = self.base_llm.generate(
                    debugger_df_mismatch(model_sql, question, self.db_type, self.db_name), output_type={"sql": "str"})
                model_sql = debugger_response["sql"]
                model_sql = validate_sql_termination(model_sql)
                exec_match = self.validate_dataframes(
                    golden_sql, model_sql)
                if not exec_match:
                    print(f"golden_sql : {golden_sql}")
                    print(f"model_sql : {model_sql}")

            if exec_match:
                correct_answer_count += 1

            if golden_sql == model_sql:
                correct_sql_count += 1

        # Calculate percentages
        percent_compiled = (compiled_queries / total_queries) * \
            100 if total_queries else 0
        percent_correct_sql = (correct_sql_count /
                               total_queries) * 100 if total_queries else 0
        percent_correct_answers = (
            correct_answer_count / total_queries) * 100 if total_queries else 0

        return compiled_queries, total_queries, percent_compiled, percent_correct_sql, percent_correct_answers

    def calculate_scores(self):
        """Main method to calculate and log scores."""
        logger.info("Starting score calculation")

        # Read data
        golden_data = self._read_file(self.golden_file)
        model_data = self._read_file(self.results_file)

        # Calculate scores
        compiled_queries, total_queries, percent_compiled, percent_correct_sql, percent_correct_answers = self._calculate_scores(
            golden_data, model_data)

        # Log results
        logger.info(f"Total Queries: {total_queries}")
        logger.info(f"Compiled Queries: {compiled_queries}")
        logger.info(f"Percentage of Compiled Queries: {percent_compiled:.2f}%")
        logger.info(
            f"Percentage of Exact Match SQL Queries: {percent_correct_sql:.2f}%")
        logger.info(
            f"Percentage of Correct Answers: {percent_correct_answers:.2f}%")

        scores = {
            "Total Queries": total_queries,
            "Compiled Queries": compiled_queries,
            "Percentage of Compiled Queries": f"{percent_compiled:.2f}%",
            "Percentage of Exact Match SQL Queries": f"{percent_correct_sql:.2f}%",
            "Percentage of Correct Answers": f"{percent_correct_answers:.2f}%",
            "Answer Mismatch Indices": self.answer_mismatch_queries,
            "execution error queries": self.execution_error_queries
        }

        self._append_scores_to_results(scores)
        return scores

    def _append_scores_to_results(self, scores):
        """Append the scores to the results file."""
        try:
            with jsonlines.open(self.results_file, mode='a') as writer:
                writer.write(scores)
            logger.info(f"Scores appended to {self.results_file}")
        except Exception as e:
            logger.error(f"Error appending scores to results file: {e}")


class Pipeline:
    def __init__(self, db_path: str, golden_file: str, db_type: str, db_name: str, base_llm: str, tuned_llm: str, prompt_type: str):
        self.db_path = db_path
        self.golden_file = golden_file
        self.db_type = db_type
        self.base_llm = base_llm
        self.db_name = db_name
        self.tuned_llm = tuned_llm
        self.prompt_type = prompt_type

    def run(self):
        # Running Query Processor
        processor = QueryProcessor(
            self.db_path, self.golden_file, self.db_type, self.db_name, self.prompt_type, self.base_llm, self.tuned_llm)
        processor.process_question()
        results_file_name = processor.get_output_file()

        # Runing Score Generator
        scorer = ScoreGenerator(
            self.db_type, self.db_name, self.db_path, self.golden_file, results_file_name, self.base_llm)
        return scorer.calculate_scores()


if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    lamini.api_key = os.getenv('api_key')

    # Initialize model
    base_llm = lamini.Lamini(model_name=os.getenv('base_model'))
    tuned_llm = lamini.Lamini(model_name=os.getenv('model_name'))

    # Parse command-line arguments
    args = parse_arguments()
    prompt_types = args.prompt_types.split(",")

    results = []
    for prompt_type in prompt_types:
        print(f"\nRunning pipeline with prompt type: {prompt_type}")
        pipeline_base = Pipeline(
            args.db_path, args.golden_file, args.db_type, args.db_name, base_llm, base_llm, prompt_type)
        base_scores = pipeline_base.run()
        pipeline_tuned = Pipeline(
            args.db_path, args.golden_file, args.db_type, args.db_name, base_llm, tuned_llm, prompt_type)
        tuned_scores = pipeline_tuned.run()
        # Storing results for this prompt type
        results.append({
            "prompt_type": prompt_type,
            "base_scores": base_scores,
            "tuned_scores": tuned_scores
        })
        # Combine results and display them as a table
    for result in results:
        prompt_type = result["prompt_type"]
        base_scores = result["base_scores"]
        tuned_scores = result["tuned_scores"]
        data = [
            ["Total Queries", base_scores["Total Queries"],
                tuned_scores["Total Queries"]],
            ["Compiled Queries", base_scores["Compiled Queries"],
                tuned_scores["Compiled Queries"]],
            ["Percentage of Compiled Queries", base_scores["Percentage of Compiled Queries"],
                tuned_scores["Percentage of Compiled Queries"]],
            ["Percentage of Exact Match SQL Queries", base_scores["Percentage of Exact Match SQL Queries"],
                tuned_scores["Percentage of Exact Match SQL Queries"]],
            ["Percentage of Correct Answers", base_scores["Percentage of Correct Answers"],
                tuned_scores["Percentage of Correct Answers"]]
        ]
        headers = [
            "Metric", f"Base Model ({prompt_type})", f"Tuned Model ({prompt_type})"]
        print(f"\nEvaluation Results for {prompt_type}:")
        print(tabulate(data, headers=headers, tablefmt="simple"))
