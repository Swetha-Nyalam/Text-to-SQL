import os
import re
import lamini
from dotenv import load_dotenv
from util.db_util import get_schema
from util.llama_prompt import make_llama_3_prompt

load_dotenv()
lamini.api_key = os.getenv('api_key')
llm = lamini.Lamini(
    model_name=os.getenv('base_model'))


# debugger

def debugger_compilation_fix(error_sql, error_message, question, db_type, db_name):
    system = """### For the given question, use the provided schema to correct the SQLite SQL query. If the query is already correct, return it as is. Correct only if necessary.
    Ensure the SQL query ends with a semicolon; append it if missing.
    Carefully analyze the error message to identify the root cause (e.g., syntax, column names, filtering logic, or data types).
    Focus on fixing the specific issue from the error message while ensuring the query aligns with the schema and the question’s intent."""
    system += "#### Database type: \n"
    system += f"{db_type}\n"
    system += "#### Table name : \n"
    system += f"{db_name}\n"
    system += "#### Error SQL:\n"
    system += f"{error_sql}\n"
    system += "#### Error message :\n"
    system += f"{error_message}\n"
    system += "Schema Description : \n"
    system += f"{get_schema()}\n"
    system += "Ensure the SQL query is valid and ends with a semicolon.\n"
    system += "Correct the error sql for the following question : \n"
    return make_llama_3_prompt(question, system)


def debugger_df_mismatch(error_sql, question, db_type, db_name):
    system = """  # For the given question, use the provided schema to correct the SQLite SQL query. If the query is already correct, return it as is. Correct only if necessary.
    # Guidelines for Fixing the SQL Query:
    1. ** Verify Column Selection: ** Ensure that the correct columns from the schema are used in the SQL query. Select the right columns as per the schema if needed.\n
    2. **Check Filtering Logic:** Review the `IN` and `NOT IN` statements. Ensure that conditions for weekdays or weekends are applied correctly. Use `strftime('%w', pickup_datetime) IN ('0', '6')` for weekends and `strftime('%w', pickup_datetime) NOT IN ('0', '6')` for weekdays.\n
    3. **Validate Date Comparisons:** Use STRFTIME('%Y-%m-%d', column) to accurately compare dates, including year, week, or day information, only if necessary and explicitly required in the question.\n
    4. **Aggregate Functions:** For queries involving `GROUP BY`, ensure the appropriate aggregate functions (`SUM`, `COUNT`, `AVG`, etc.) are used according to the question's requirements.\n
    5. **Limit Clause:** Ensure the `LIMIT` clause returns the correct number of rows based on the question's specifications.\n
    6. **Column and Table Aliases:** Use column and table aliases where applicable, especially in subqueries or `WITH` statements, for better readability and correctness.\n
    7. **Align WHERE Conditions:** Ensure `WHERE` conditions match the original question's intent, such as filtering by specific dates, years, weekdays, or weekends.\n
    8. **Date and Aggregation Accuracy:** Verify that date filters and aggregations align with the question’s requirements (e.g., total rides, average fares).\n
    9. **Correct Filtering:** For weekend filtering, use `strftime('%w', pickup_datetime) IN ('0', '6')`. For weekdays, use `strftime('%w', pickup_datetime) NOT IN ('0', '6')`.\n
    10. **Match Query Intent:** Ensure the aggregation and filtering logic matches the question’s intent (e.g., top cities by rides, highest average fare amounts).\n"""
    system += "#### Table name : \n"
    system += f"{db_name}\n"
    system += "#### Database type: \n"
    system += f"{db_type}\n"
    system += "#### Error SQL:\n"
    system += f"{error_sql}\n"
    system += "Schema Description : \n"
    system += f"{get_schema()}\n"
    system += "Correct the error sql for the following question : \n"
    return make_llama_3_prompt(question, system)

# score_prompt


def score_prompt(reference_execution_result, model_execution_result):
    # Your evaluation model compares SQL output from the generated and reference SQL queries, using another LLM in the pipeline
    # Your evaluation model compares SQL output from the generated and reference SQL queries, using another LLM in the pipeline
    system_prompt = "Compare the following two dataframes. They are similar if they are almost identical, or if they convey the same information about the given dataset.\n"
    user_prompt = (
        f"========== Dataframe 1 =========\n"
        f"{str(model_execution_result).lower()}\n\n"
    )
    user_prompt += (
        f"========== Dataframe 2 =========\n"
        f"{str(reference_execution_result).lower()}\n\n"
    )
    user_prompt += f"Can you tell me if these dataframes are similar or convey exactly similar information?\n"
    user_prompt += f"Also write explanation in one or two lines about why you think the data frames are similar or otherwise.\n"
    user_prompt += "Respond with valid JSON {'explanation' : str, 'similar' : bool}.\n"
    return make_llama_3_prompt(user_prompt, system_prompt)


# basic_prompt


def basic_prompt(db_name, db_type, question):
    system = f"""You are an expert SQL query generator. Given the schema for the {
        db_name} table and a user’s question, generate an accurate SQL query to answer it.The database type is {db_type}."""
    system += "Understand the datatypes of the columns and try to generate a valid SQL query.\n"
    system += "Consider the following database table with the following schema and understand the column data types:\n"
    system += get_schema() + "\n"
    system += "Ensure the SQL query is valid and ends with a semicolon."
    return make_llama_3_prompt(question, system)


# chain_of_thought prompt

def chain_of_thought(db_name, db_type, question):
    system = f"""You are an expert SQL query generator. Given the schema for the {
        db_name} table and a user’s question, generate an accurate SQL query to answer it.The database type is {db_type}."""
    system += "Schema Description : \n"
    system += f"{get_schema()}\n"
    system += """
    Let's break down the question and think step-by-step to generate the SQL query. :
    1. ** Analyze the Question **: Understand the user's question, identifying the key information they are seeking.
    2. ** Map to Database Schema **: Identify which tables and columns in the schema are relevant to answering the question.
    3. ** Construct SQL Query **: Write the SQL query, ensuring it accurately reflects the user's intent, uses the correct columns, and includes necessary filters, groupings, and orderings.
    4. ** Constructed SQL Query **: Present the SQL query in a readable format.
    5. ** Validation **: Ensure that the query is correct and optimized for performance, and that it answers the user's question effectively.
    """
    system += "Answer the following question : \n"
    return make_llama_3_prompt(question, system)

# self ask prompt
# Taken from 'Measuring and Narrowing the Compositionality Gap in Language Models'(https://arxiv.org/pdf/2210.03350)
# code link : https://github.com/ofirpress/self-ask


def self_ask(db_name, db_type, question):
    system = f"""You are an expert SQL query generator. Given the schema for the {
        db_name} table and a user’s question, generate an accurate SQL query to answer it. The database type is {db_type}.
    """
    system += "Schema Description:\n"
    system += f"{get_schema()}\n"

    system += f"""Let's break down the process step-by-step by asking relevant follow-up questions to construct the SQL query.

    Question: {question}
    Are follow-up questions needed here: Yes.

    Follow-up: What are the key pieces of information the user is asking for?
    Intermediate answer: [Insert key information derived from the question].

    Follow-up: Which tables or columns in the schema are relevant for this query?
    Intermediate answer: [Insert relevant tables and columns based on the schema].

    Follow-up: What filters, groupings, or orderings are necessary for the query?
    Intermediate answer: [Insert any necessary WHERE conditions, GROUP BY, ORDER BY, etc.].

    Follow-up: How do we construct the SQL query based on the identified information?
    Intermediate answer: [Write out the SQL query, ensuring it's correctly structured].

    So the final SQL query is: [return final SQL query]."""

    system += "Ensure the SQL query is valid and ends with a semicolon.\n"

    return make_llama_3_prompt(question, system)


def chain_of_thought_2(db_name, db_type, question):
    system = f"""You are an expert SQL query generator. Given the schema for the {
        db_name} table and a user’s question, generate an accurate SQL query to answer it.The database type is {db_type}."""
    system += "Schema Description : \n"
    system += f"{get_schema()}\n"
    system += """
    Let's think step by step :
    1. ** Analyze the Question**: Understand the user's question, identifying the key information they are seeking.
        - Ask yourself: What is the main focus of the question? What data is being requested (e.g., sales, product info)? What filters (e.g., date range, product category) are needed?
    2. **Map to Database Schema**: Identify which tables and columns in the schema are relevant to answering the question.
        - Ask yourself: Which columns directly relate to the question? Are there any joins or filters required?
    3. **Construct SQL Query**: Write the SQL query, ensuring it accurately reflects the user's intent, uses the correct columns, and includes necessary filters, groupings, and orderings.
         - **SELECT**: Choose the columns to return.
         - **FROM**: Identify the table(s).
         - **JOIN**: Determine necessary table joins and conditions.
         - **WHERE**: Apply filters (e.g., date ranges, categories).
         - **GROUP BY/HAVING**: Add for aggregation or grouping.
         - **ORDER BY**: Define result ordering if needed.
        - Ask yourself: Does each part of the query answer the specific question? Are the correct columns and conditions used?
    4. **Constructed SQL Query**: Present the SQL query in a readable format.
        - Ask yourself: Is the query optimized for performance? Can any part of it be simplified?
    5. **Validation**: Ensure that the query is correct and optimized for performance, and that it answers the user's question effectively.
       - Ask yourself: Will this query return the exact data the user is asking for? Does it avoid unnecessary computations?

    """
    system += "Answer the following question : \n"
    system += "Ensure the SQL query is valid and ends with a semicolon.\n"
    return make_llama_3_prompt(question, system)

# code representaion prompt
# Taken from DIAL-SQL(https://arxiv.org/pdf/2308.15363)


def code_representation(db_name, db_type, question):
    system = "Given the following table schema:\n"
    system += f"The database type is {db_type}.\n"
    system += '''
    CREATE TABLE uber (
        Id INTEGER,
        fare_amount REAL,
        pickup_datetime DATETIME,
        passenger_count INTEGER,
        distance_miles REAL,
        pickup_statecode TEXT,
        pickup_statename TEXT,
        pickup_city TEXT,
        dropoff_statecode TEXT,
        dropoff_statename TEXT,
        dropoff_city TEXT,
        pickup_county TEXT,
        dropoff_county TEXT
    );
    '''
    user = f"/* Answer the following question: {question} */\n"
    user += "SELECT "

    return make_llama_3_prompt(user, system)


# DIN-SQL prompting
# Taken from  DIN-SQL(https://arxiv.org/pdf/2304.11015)
# code link : https://github.com/MohammadrezaPourreza/Few-shot-NL2SQL-with-prompting/blob/main/DIN-SQL.py

Intermediate_representation_examples = '''

Q : "Which city has the most number of trips during weekends overall?"
Intermediate_representation:
  - First, extract the day of the week from pickup_datetime and filter for weekends (Saturday, Sunday).
  - Group the trips by pickup_city.
  - Count the number of trips for each city (using the Id column).
  - Order the cities by the trip count in descending order.
  - Select the city with the highest count.

Q : "Which city in New York had the highest number of rides during 4 PM to 7 PM in 2015?"
Intermediate_representation:
  - First, filter the rides where the pickup_statecode is 'NY' to only consider cities in New York.
  - Extract the year from pickup_datetime and filter for rides that occurred in 2015.
  - Extract the hour from pickup_datetime and filter for rides that occurred between 16:00 and 19:00 (4 PM to 7 PM).
  - Group the rides by pickup_city.
  - Count the number of rides (using the Id column) for each city.
  - Order the cities by the ride count in descending order.
  - Select the city with the highest count.

Q : "What is the most popular time of day for pickups in New York City?"
Intermediate_representation:
  - First, filter the pickups where the pickup_statecode is 'NY' and pickup_city is 'New York City'.
  - Extract the hour from pickup_datetime to focus on the time of day.
  - Group the pickups by the extracted hour to count the number of rides for each hour.
  - Count the number of rides (using the Id column) for each hour.
  - Order the hours by the ride count in descending order.
  - Select the hour with the highest count of pickups.

Q : "Which city had the most rides in 2009?"
Intermediate_representation:
  - First, filter the rides where the pickup occurred in the year 2009 by extracting the year from pickup_datetime.
  - Group the rides by pickup_city to count the number of rides per city.
  - Count the number of rides (using the Id column) for each city.
  - Order the cities by the number of rides in descending order.
  - Select the city with the highest number of rides.

'''


def classification_prompt(question):
    system = "# For the given SQL question, classify it as EASY or MEDIUM based on the approach needed to solve it.\n"
    system += "if the question can be solved directly by aggregating columns or performing simple operations: predict 'EASY'\n"
    system += "if the question is better solved by breaking it into smaller sub-problems first before solving the main problem: predict 'MEDIUM'\n\n"
    system += "Schema Description : \n"
    system += f"{get_schema()}\n"
    user = f"Classify the input:\n{question}\n.\n"
    user += "Your output should be a valid JSON in the format: {'class': '(EASY) or (MEDIUM)'}\n"
    return make_llama_3_prompt(user, system)


def easy_prompt_maker(question, db_name, db_type):
    system = f"Use the the schema to generate the SQL queries for each of the questions.\n"
    system += f"The database type is : {db_type}.\n"
    system += f"Given the schema for the {db_name} table :\n"
    system += f"{get_schema()}\n"
    system += "The output is a valid json like : {'sql': 'str'}\n"
    return make_llama_3_prompt(question, system)


def generate_IR_prompt(question, db_name):
    system = "Given few examples of how to create an Intermediate Representation for a given question which will help in constructing the accurate SQL query based on the question.\n"
    system += f"{Intermediate_representation_examples}\n"
    system += f"Given the schema for the {db_name} table : \n"
    system += f"{get_schema()}\n"
    system += "Now, let's think step by step and create an Intermediate Representation for the given question:\n"
    return make_llama_3_prompt(question, system)


def medium_prompt_maker(question, db_name, Intermediate_representation):
    system = "# Use the the schema and Intermediate_representation to generate the SQL queries for each of the questions.\n"
    system += "Intermediate_representation is : \n"
    system += f"{Intermediate_representation}\n\n"
    system += f"Given the schema for {db_name} table : \n"
    system += f"{get_schema()}\n"
    system += "Given these pervious questions, generate sql for the following question.\n"
    system += "The output is a valid json like : {'sql': 'str'}\n"
    return make_llama_3_prompt(question, system)


def din_sql(db_name, db_type, question):
    classification = llm.generate(classification_prompt(
        question), output_type={"class": "str"})
    if classification["class"] == "EASY":
        return easy_prompt_maker(question, db_name, db_type)
    else:
        response_IR = llm.generate(generate_IR_prompt(question, db_name), output_type={
                                   "Intermediate_representation": "str"}, max_new_tokens=1500)
        return medium_prompt_maker(question, db_name, response_IR["Intermediate_representation"])


# LEAST-TO-MOST PROMPTING
# LEAST-TO-MOST PROMPTING ENABLES COMPLEX REASONING IN LARGE LANGUAGE MODELS(https: // arxiv.org/pdf/2205.10625)

decomposition_examples = '''

Eaxmple 1:
Question : 'Which city had the most rides during the summer of 2015?'
Decomposition_steps :
Step 1: Filter the rides where the pickup_datetime is between June 1, 2015, and August 31, 2015, to consider only rides during the summer months of 2015.
Step 2: Group the filtered rides by the pickup_city to organize the data by city.
Step 3: Count the rides for each city, sort the results by the number of rides in descending order, and select the city with the highest ride count.

Example 2:
Question : 'What are the top three states with the highest total revenue from trips in 2013?'
Decomposition_steps :
Step 1: Filter the trips where the pickup_datetime is between January 1, 2013, and December 31, 2013, to consider only trips that occurred in 2013.
Step 2: Group the filtered trips by the pickup_statecode (or pickup_statename) to organize the data by state.
Step 3: Sum the fare_amount for each state, sort the results by total revenue in descending order, and select the top three states with the highest total revenue.

Example 3 :
Question : 'Which city had the lowest total fare amount for all trips taken in December?'
Decomposition_steps :
Step 1: Filter the trips where the pickup_datetime is between December 1 and December 31 of the relevant year to consider only trips taken in December.
Step 2: Group the filtered trips by the pickup_city to organize the data by city.
Step 3: Sum the fare_amount for each city, sort the results by total fare amount in ascending order, and select the city with the lowest total fare amount.

'''


def decomposition_prompt(db_name, question):
    system = f"Given the schema for {db_name} table : \n"
    system += f"{get_schema()}\n"
    system += "Follow these decompostion examples on how the task could be logically broken down into three steps"
    system += "Decomposition Examples:\n"
    system += f"{decomposition_examples}"
    system += "Based on the examples provided, break down the user question into logical steps in natural language, ensuring that the outcome of each step enhances the clarity and informs the next step."
    system += "Output the three steps in valid jsonl format like : {'step1': 'str', 'step2':'str', 'step3':'str'}"
    system += "Decomposition steps : "
    return make_llama_3_prompt(question, system)

# sub problem solving stage


def generate_sql_for_decomposition_steps(db_name, db_type, question, decomposition_steps):
    accumulated_steps_and_sqls = ""
    for i, (step_key, step) in enumerate(decomposition_steps.items()):
        if i == 0:
            step_prompt = f"""Step {i + 1}: {step}
            Generate the SQL query to accomplish this step."""
        else:
            step_prompt = f"""Step {i + 1}: {step}\nUse the results from all previous steps:\n
            {accumulated_steps_and_sqls}\nGenerate the next SQL query."""

        sql_query = llm.generate(make_llama_3_prompt(step_prompt, chain_of_thought(
            db_name, db_type, question)), output_type={"sql": "str"}, max_new_tokens=1500)

        accumulated_steps_and_sqls += f"""\nStep {i + 1}: {step}
        Generated SQL: {sql_query['sql']}\n"""
    system = f"Given the schema for {db_name} table : \n"
    system += f"{get_schema()}\n"
    system += f"The database type is : {db_type}.\n"
    system += "Now that you have sequentially executed all the decomposition steps as following, let's answer the original question.\n"
    system += accumulated_steps_and_sqls
    system += "\nBased on the above steps and SQL queries, provide the final answer to the question by identifying which columns in the schema are relevant to answering the question.\n"

    return make_llama_3_prompt(question, system)


def leasttomost(db_name, db_type, question):
    decomposition_steps = llm.generate(make_llama_3_prompt(
        question, decomposition_prompt(db_name, question)), output_type={"step1": "str", "step2": "str", "step3": "str"}, max_new_tokens=3000)
    return generate_sql_for_decomposition_steps(db_name, db_type, question, decomposition_steps)
