import lamini
import jsonlines
from tqdm import tqdm
from util.db_util import get_schema
from util.llama_prompt import make_llama_3_prompt

lamini.api_key = "1ac6f38b3e7f498fa2c3ed6ea9df68b9"
llm = lamini.Lamini(
    model_name="meta-llama/Meta-Llama-3-8B-Instruct")


"""

This code processes the golden dataset of questions and SQL queries.
It generates multiple rephrased versions of each question to evaluate the
performance of a language model (LLM) on unseen and rephrased questions. The class
reads data from the GoldenDataSet.jsonl file, constructs prompts for the LLM, and
appends the generated rephrased questions back into the dataset. The purpose is
to assess how well the LLM generalizes.

"""

data = []

with jsonlines.open("Data/GoldenDataSet.jsonl", "r") as reader:
    for obj in reader:
        data.append(obj)


def make_prompt(question: str, sql: str):
    system = "You are an expert SQL query generator. You are provided with a database schema and a user's question about the data. Your task is to generate an accurate SQL query based on the schema to answer the user's question.\n"
    system += " Given the following database table with the following schema:\n"
    system += f"{get_schema()}\n"
    system += f"Understand the schema, question : {
        question} and a sql : {sql}\n"
    user = "Generate two rephrased questions to the original question for which the sql can be used to infer the results from the table.\n"
    user += "Format the queries as a JSON object, i.e.\n"
    user += '{ "query_1" : str, "query_2": str }.\n'
    return make_llama_3_prompt(user, system)


with jsonlines.open("Data/GoldenDataSet.jsonl", mode="a") as writer:
    # Wrap the data list with tqdm to show the progress bar
    for obj in tqdm(data, desc="Appending Data"):
        prompt = make_prompt(obj["question"], obj["sql"])
        results = llm.generate(prompt, output_type={
            "query_1": "str",
            "query_2": "str"
        })
        append_obj = {
            "question": results["query_1"],
            "answer": obj["answer"],
            "sql": obj["sql"]
        }
        writer.write(append_obj)
        append_obj["question"] = results["query_2"]
        writer.write(append_obj)
