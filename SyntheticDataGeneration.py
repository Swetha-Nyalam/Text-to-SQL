import lamini
import jsonlines
import pandas as pd
from tqdm import tqdm
from util.db_util import get_schema
from util.llama_prompt import make_llama_3_prompt
from util.db_util import connect
import json
import random

lamini.api_key = "1ac6f38b3e7f498fa2c3ed6ea9df68b9"


class SyntheticSQLData:
    """

    The SQLProcessor class generates new SQL queries from Text-to-SQL dataset pairs.

    It queries an LLM with the database schema, question, and original SQL.

    The LLM generates structurally different but correct SQL queries.

    A similarity score is provided by the LLM to measure closeness to the ground truth.

    Generated SQLs are validated by comparing execution results with the original SQL.

    Queries with similarity scores below 0.9 are added to the training dataset.

    """

    def __init__(self, db_type, model_name, input_file, output_file):
        self.llm = lamini.Lamini(model_name=model_name)
        self.db_type = db_type
        self.input_file = input_file
        self.output_file = output_file
        self.engine = connect(self.db_type)

    def execute_query(self, sql):
        """Execute a SQL query and return the DataFrame or None if an error occurs."""
        try:
            return pd.read_sql(sql, con=self.engine)
        except Exception as e:
            print(f"Error executing SQL query: {e}")
            return None

    def validate(self, generated_sql, true_sql):
        """Compare the DataFrame results of two SQL queries."""
        df_generated = self.execute_query(generated_sql)
        df_true = self.execute_query(true_sql)

        # Return False if either DataFrame is None
        if df_generated is None:
            return False

        # Compare the DataFrames' string representations
        return df_generated.to_string(index=False) == df_true.to_string(index=False)

    def generate_data(self):
        """Read data, generate synthetic SQL queries, and validate them."""
        with jsonlines.open(self.input_file, 'r') as reader:
            records = list(reader)
        total_records = len(records)

        with jsonlines.open(self.input_file, 'r') as reader:
            with jsonlines.open(self.output_file, 'w') as writer:
                for obj in tqdm(records, total=total_records, desc="Processing"):
                    output_type = {
                        "sql1": "str", "similarity1": "float",
                        "sql2": "str", "similarity2": "float",
                        "sql3": "str", "similarity3": "float"
                    }
                    prompt = self.make_prompt(obj["question"], obj["sql"])
                    results = self.llm.generate(
                        prompt, output_type=output_type)
                    for index in range(3):
                        sql = results["sql" + str(index + 1)]
                        similarity = results["similarity" + str(index + 1)]
                        if not sql.strip().endswith(';'):
                            sql += ';'
                        if similarity < 0.9 and self.validate(sql, obj["sql"]):
                            obj_w = {
                                "question": obj["question"],
                                "sql": sql
                            }
                            writer.write(obj_w)
        self.close_connection()

    def make_prompt(self, question, sql):
        system = "You are an expert SQL query generator. You are provided with a database schema and a user's question about the data. Your task is to generate an accurate SQL query based on the schema to answer the user's question.\n"
        system += "Understand the following schema and the database type, as the insights derived from it will aid in making business decisions.\n"
        system += get_schema() + "\n"
        system += "db type is : SQLite"
        system += "Your job is to understand the natural language queries and generate up to 3 different SQL queries using diverse commands from the original query while answering the question correctly.\n"
        system += "You need to make sure to use the same columns from the original query for the generated query. \n"
        system += "You will also generate a similarity(between 0-1) score between the original and the generated query based on how closer they are syntactically.\n"
        system += "If both the queries are same then score is 1, which means high similarity.\n"
        user = f"natural language query : {question}"
        user += f"original sql query : {sql}"
        user += "make sure the output is in the form of jsonl like {'sql1': // generated query1,'sql2': // generated query2, 'similarity1' : // similarity score (0.0-1.0), 'similarity1' : // similarity score (0.0-1.0)}"
        return make_llama_3_prompt(user, system)

    def close_connection(self):
        """Closing the database connection."""
        self.engine.close()


class llm_TrainData:

    """
    Reading randomly sampled question-SQL pairs from an input file.

    Generating new SQL queries based on schema and example questions.

    Executing the generated queries to ensure they are valid.

    Saving valid queries to an output file, thereby expanding the training data with new, diverse question-SQL pairs.

    """

    def __init__(self, num_examples, num_iterations, db_type, model_name, input_file, output_file):
        self.llm = lamini.Lamini(model_name=model_name)
        self.db_type = db_type
        self.input_file = input_file
        self.output_file = output_file
        self.num_iterations = num_iterations
        self.engine = connect(self.db_type)
        self.num_examples = num_examples
        self.data = []

    def _read_inputfile(self):
        with open(self.input_file, 'r') as file:
            for line in file:
                self.data.append(json.loads(line))

    def _generate_num_examples(self):
        # Randomly sample indices
        sampled_indices = random.sample(
            range(len(self.data)), self.num_examples)
        return [self.data[i] for i in sampled_indices]

    def make_prompt(self):
        sampled_examples = self._generate_num_examples()
        system = "You are an expert SQL query generator. You are provided with a database schema and a user's question about the data. Your task is to generate an accurate SQL query based on the schema to answer the user's question.\n"
        system += "Consider the following database table with the following schema:\n"
        system += get_schema() + "Given 3 examples : \n"
        system += f"Examples: {sampled_examples}.\n"
        user = "Generate one more question and sql pair for the given schema where the question is different to the original questions.\n"
        user += "The output has to be a valid jsonl like {'question': 'str', 'sql': 'str'}.\n"
        return make_llama_3_prompt(user, system)

    def execute_query(self, sql):
        """Execute a SQL query and return the DataFrame or None if an error occurs."""
        try:
            return pd.read_sql(sql, con=self.engine)
        except Exception as e:
            print(f"Error executing SQL query: {e}")
            return None

    def generate_data(self):
        self._read_inputfile()
        with jsonlines.open(self.output_file, 'w') as writer:
            for _ in tqdm(range(self.num_iterations), desc="Generating Data"):
                prompt = self.make_prompt()
                results = self.llm.generate(prompt, output_type={
                    "question": "str",
                    "sql": "str"
                })
                df = self.execute_query(results["sql"])
                if not (df is None or df.empty):
                    obj_w = {
                        "question": results["question"],
                        "sql": results["sql"]
                    }
                    writer.write(obj_w)


class repharse_questions:
    """

    The function takes an original natural language question as input.

    The goal is to generate multiple rephrased versions of the original question.

    The function generates synthetic data by producing alternative phrasings of the query.

    These rephrased questions help expand the dataset with diverse variations of the same query concept.

    """

    def __init__(self, model_name, input_file, output_file):
        self.llm = lamini.Lamini(model_name=model_name)
        self.input_file = input_file
        self.output_file = output_file

    def make_question_prompt(self, question, sql):
        system = "You are an expert SQL query generator. You are provided with a database schema and a user's question about the data. Your task is to generate an accurate SQL query based on the schema to answer the user's question.\n"
        system += get_schema() + "\n"
        user = "Consider the following query.\n"
        user += "Query: " + sql + "\n"
        user += f"Rephrase the following query into a different question that still requires the same SQL query: {
            sql}. The original query is: {question}.\n"
        user += "Format your response as a JSON object, i.e.\n"
        user += "{'question': 'str','explanation': 'str' }.\n"
        user += "First write an explanation in about 3-5 sentences, then write a one sentence question.\n"

        return make_llama_3_prompt(system, user)

    def generate_data(self):
        data = [
            {"question": "What is the most popular pickup time during weekdays in New York?",
                "sql": "SELECT strftime('%H:%M', pickup_datetime) AS time, COUNT(*) AS pickup_count FROM uber WHERE pickup_statecode = 'NY' AND strftime('%w', pickup_datetime) BETWEEN '1' AND '5' GROUP BY time ORDER BY pickup_count DESC LIMIT 1;"},
            {"question": "Which city in New York has the highest number of rides with exactly one passenger on weekends?",
                "sql": "SELECT pickup_city, COUNT(*) AS ride_count FROM uber WHERE pickup_statecode = 'NY' AND passenger_count = 1 AND strftime('%w', pickup_datetime) IN ('0', '6') GROUP BY pickup_city ORDER BY ride_count DESC LIMIT 1;"},
            {"question": "Which city in New York experienced the highest increase in the number of trips during the winter holidays (December) from 2011 to 2012?",
             "sql": "SELECT pickup_city, (COUNT(CASE WHEN strftime('%Y', pickup_datetime) = '2012' THEN 1 END) - COUNT(CASE WHEN strftime('%Y', pickup_datetime) = '2011' THEN 1 END)) AS increase_in_trips FROM uber WHERE pickup_statecode = 'NY' AND strftime('%m', pickup_datetime) = '12' GROUP BY pickup_city ORDER BY increase_in_trips DESC LIMIT 1;"}
        ]
        with jsonlines.open(self.output_file, 'w') as writer:
            with jsonlines.open(self.input_file, 'r') as reader:
                for obj in tqdm(list(reader), desc="Processing Entries"):
                    prompt = self.make_question_prompt(
                        obj["question"], obj["sql"])
                    response = self.llm.generate(prompt, output_type={
                        "explanation": "str",
                        "question": "str"
                    })
                    obj_w = {
                        "question": response["question"],
                        "sql": obj["sql"]
                    }
                    writer.write(obj_w)


instance = SyntheticSQLData("uber_rides.db", "meta-llama/Meta-Llama-3.1-8B-Instruct",
                            "Data/train_data.jsonl", "Data/similarity_test.jsonl")
result = instance.generate_data()
