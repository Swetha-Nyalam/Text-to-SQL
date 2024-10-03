import lamini
import jsonlines
import sqlite3
import argparse
from tqdm import tqdm
from util.db_util import get_schema
from util.llama_prompt import make_llama_3_prompt


lamini.api_key = "1ac6f38b3e7f498fa2c3ed6ea9df68b9"


class FeedbackSystem:
    """
    This class enhances the training data by generating feedback for SQL queries. It performs the following steps:

    1. Generate SQL Query:
       - For each question in the dataset, the base model generates a corresponding SQL query.

    2. Generate Feedback:
       - After generating the SQL query, the base model provides feedback by comparing it against the SQL from the training data. This feedback includes information on potential errors or areas for improvement, summarized in 2-3 lines.

    3. Append Feedback to Training Data:
       - The generated feedback is appended to the existing training data. The updated training data now includes three fields:
         - `question`: The original question.
         - `sql`: The generated SQL query.
         - `feedback`: The feedback on the SQL query.

    4. Testing with Eval Dataset:
       - The updated training data is then used to test the accuracy and effectiveness of the model using the evaluation dataset.

    5. Testing Results:(for same hyper-parameters)
       - Exact match SQL queries increased from around 15% to 75%.
       - Correct SQL answers increased from 85% to 100%.

    This class is essential for improving model performance by iteratively enhancing the training dataset with valuable feedback.
    """

    def __init__(self, model_name,  db_path, input_file, output_file):
        self.llm = lamini.Lamini(model_name=model_name)
        self.engine = sqlite3.connect(db_path)
        self.cursor = self.engine.cursor()
        self.input_file = input_file
        self.output_file = output_file

    def make_sql_generation_prompt(self):
        system = "You are an specialist in writing sql queries with over 15 years of experience working as an analyst, advising business decisions through data analysis.\n"
        system += " Given the following database table with the following schema:\n"
        system += f"{get_schema()}\n"
        user = "Understand the schema thoroughly and generate SQL queries that can be used to obtain accurate results from the Uber table.\n"
        user += "Output a valid jsonl like {'sql': 'str'}\n"
        user += "Make sure the sql generated is valid and terminates with a semicolon.\n"
        return make_llama_3_prompt(user, system)

    def compare_prompt(self, llm_sql, true_sql, data):
        system = "You are an expert SQL query generator. You are provided with a database schema and a user's question about the data. Your task is to generate an accurate SQL query based on the schema to answer the user's question.\n"
        system += "Given the following database table with the following schema:\n"
        system += f"{get_schema()}.\n"
        system += "Carefully analyze the examples on feedback.\n"
        for example in data:
            system += "true sql query : " + example["true_sql"] + "\n"
            system += "sql query to be improved : " + \
                example["error_sql"] + "\n"
            system += "feedback : " + example["feedback"] + "\n"
        user = f"You are given two queries, error sql : {llm_sql},\n"
        user += f"correct sql : {true_sql}.\n"
        user += "Write logically correct two lines of feedback in natural language with suggested improvements to correct error sql comparing it to correct sql.\n"
        user += "If two queries are equal return an empty string.Make sure to not include any kind of SQL and 'error SQL' or 'correct SQL' in feedback.\n"
        user += "Output a valid jsonl like {'feedback': 'str'} .\n"
        return make_llama_3_prompt(user, system)

    def get_sample_data(self):
        sample_feedback_data = [
            {"true_sql": "SELECT strftime('%H:%M', pickup_datetime) AS pickup_time, COUNT(*) AS pickup_count FROM uber WHERE pickup_statecode = 'NY' AND strftime('%w', pickup_datetime) BETWEEN '1' AND '5' GROUP BY pickup_time ORDER BY pickup_count DESC LIMIT 1;",
             "error_sql": "SELECT strftime('%H:%M', pickup_datetime) AS pickup_time, COUNT(*) AS pickup_count FROM uber WHERE pickup_city = 'NYC' AND strftime('%w', pickup_datetime) = '1' GROUP BY pickup_datetime ORDER BY pickup_datetime ASC LIMIT 1;",
             "feedback": "The query incorrectly filters by city ('NYC') rather than state ('NY'). It also groups by the full datetime rather than just the time of day. Additionally, it only considers pickups on Mondays instead of weekdays. The correct query should filter by the state, group by time of day, and include pickups from Monday to Friday."},
            {"true_sql": "SELECT COUNT(*) AS trip_count FROM uber WHERE substr(pickup_datetime, 1, 4) = '2010';",
             "error_sql": "SELECT COUNT(*) AS trip_count FROM uber WHERE EXTRACT(YEAR FROM pickup_datetime) = 2010;",
             "feedback": "The `EXTRACT` function is not supported by SQLite. Use `substr` to extract the year from `pickup_datetime` instead, as SQLite relies on string functions for date parts."},
            {"true_sql": "SELECT substr(pickup_datetime, 1, 10) AS trip_date, COUNT(*) AS trip_count FROM uber WHERE substr(pickup_datetime, 1, 4) = '2015' GROUP BY trip_date ORDER BY trip_count ASC LIMIT 1;",
             "error_sql": "SELECT pickup_city FROM uber WHERE passenger_count > ( SELECT AVG(passenger_count) FROM uber );",
             "feedback": "The query needs a `LIMIT` clause to return only one result. The correct query uses `LIMIT 1` to restrict results to the day with the fewest trips."},
            {"true_sql": "SELECT pickup_city FROM uber WHERE pickup_datetime NOT LIKE '%weekend%' GROUP BY pickup_city ORDER BY COUNT(*) ASC LIMIT 1;",
             "error_sql": "SELECT pickup_city FROM uber WHERE pickup_datetime NOT LIKE '%weekend%' GROUP BY pickup_city ORDER BY COUNT(*) ASC LIMIT 1;",
             "feedback": ""}
        ]
        return sample_feedback_data

    def generate_and_append_feedback(self):
        # Count the total number of lines
        with jsonlines.open(self.input_file, 'r') as reader:
            total_lines = sum(1 for _ in reader)

        sample_feedback_data = self.get_sample_data()

        # Process the data
        with jsonlines.open(self.input_file, 'r') as reader, jsonlines.open(self.output_file, 'w') as writer:
            with tqdm(total=total_lines, desc="Processing Queries") as pbar:
                for obj in reader:
                    sql_generation_prompt = make_llama_3_prompt(
                        obj["question"], self.make_sql_generation_prompt())
                    llm_sql = self.llm.generate(
                        sql_generation_prompt, output_type={"sql": "str"})
                    compare_prompt = make_llama_3_prompt(self.compare_prompt(
                        llm_sql["sql"], obj["sql"], sample_feedback_data))
                    feedback_results = self.llm.generate(compare_prompt, output_type={
                        "feedback": "str"}, max_new_tokens=100)
                    train_obj = {
                        "question": obj["question"],
                        "sql": obj["sql"],
                        "feedback": feedback_results["feedback"]
                    }
                    writer.write(train_obj)
                    pbar.update(1)


if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description="Generate SQL query feedback.")
    parser.add_argument("model_name", type=str, help="Name of the LLM model.")
    parser.add_argument("db_path", type=str,
                        help="Path to the SQLite database.")
    parser.add_argument("input_file", type=str,
                        help="Input JSONL file containing questions.")
    parser.add_argument("output_file", type=str,
                        help="Output JSONL file for feedback.")

    args = parser.parse_args()

    # Create an instance of FeedbackSystem with provided arguments
    feedback_appender = FeedbackSystem(
        args.model_name, args.db_path, args.input_file, args.output_file)

    # Generate and append feedback
    feedback_appender.generate_and_append_feedback()


class QueryLogicSystem:
    """
    This class generates SQL query generation process steps to training data with detailed steps.

    """

    def __init__(self, model_name, input_file, output_file):
        self.llm = lamini.Lamini(model_name=model_name)
        self.input_file = input_file
        self.output_file = output_file

    def get_sample_data(self):
        sample_data = [
            {
                "question": "Which city had the highest number of drop-offs in 2014 among all the cities in the Uber dataset?",
                "steps": "1. Filter records where the year in pickup_datetime is 2014 using strftime('%Y', pickup_datetime) = '2014'. 2. Group the filtered records by dropoff_city and count the number of drop-offs for each city. 3. Order the results by the count of drop-offs in descending order and select the top city with LIMIT 1."},
            {
                "question": "What is the average passenger count for Uber pickups in New York during the workweek?",
                "steps": "1. Filter the records where the pickup_city is 'New York'. 2. Filter further to include only records where the pickup_datetime falls on a weekday, excluding weekends (use strftime('%w', pickup_datetime) NOT IN ('0', '6')). 3. Calculate the average passenger count from the filtered records using the AVG function."},
            {
                "question": "Which city had the highest average fare amount during weekends in 2015?",
                "steps": "1. Filter the records where the pickup_datetime falls between '2015-01-01' and '2015-12-31'. 2. Further filter to include only records where the pickup_datetime falls on a weekend (use strftime('%w', pickup_datetime) IN ('0', '6')). 3. Group the filtered records by pickup_city and calculate the average fare amount for each city. 4. Order the cities by the average fare amount in descending order and select the city with the highest average using LIMIT 1."},
            {
                "question": "What are the top three cities with the highest number of rides during weekends in 2014?",
                "steps": "1. Filter records where the year in pickup_datetime is 2014 and the day is a weekend (Saturday or Sunday) using strftime('%Y', pickup_datetime) = '2014' and strftime('%w', pickup_datetime) IN ('0', '6'). 2. Group the filtered records by pickup_city and count the number of rides for each city. 3. Order the results by ride count in descending order and select the top three cities using LIMIT 3."}
        ]
        return sample_data

    def make_prompt(self, question):
        self.sample_data = self.get_sample_data()
        system = "You are a database specialist with over 15 years of experience working as an analyst, advising business decisions through data analysis.\n"
        system += "Understand the following schema:\n"
        system += get_schema() + "\n"
        system += "Understand the breakdown of the task of SQL generation into simpler steps for the following question:\n"
        system += f"{self.sample_data}\n"
        system += "Your job is to generate 3 steps in natural language for the query.\n"
        user = f"Query: {question}\n"
        user += "Provide the steps to derive the SQL query in JSON Lines (jsonl) format, like this:\n"
        user += "{'step1': 'str', 'step2': 'str', 'step3': 'str'}"
        return make_llama_3_prompt(user, system)

    def append_query_logic_steps(self):
        with jsonlines.open(self.output_file, 'w') as writer:
            with jsonlines.open(self.input_file, 'r') as reader:
                for obj in tqdm(reader, desc="Processing"):
                    prompt = self.make_prompt(obj["question"])
                    results = self.llm.generate(prompt, output_type={
                        "step1": "str",
                        "step2": "str",
                        "step3": "str"
                    })
                    writer.write({
                        "question": obj["question"],
                        "sql": obj["sql"],
                        "query_building_steps": f"step1: {results['step1']}\nstep2: {results['step2']}\nstep3: {results['step3']}"
                    })


# instance = QueryLogicSystem("meta-llama/Meta-Llama-3.1-8B-Instruct",
#                            "Data/train_data.jsonl", "Data/train_data_decomposition.jsonl")
# instance.append_query_logic_steps()
