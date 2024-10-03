import logging
import sqlite3
import pandas as pd

"""
   This function is a generic interface for running different types of data base types

"""

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseQueryExecutor:

    # Executes SQL queries on various database systems using SQLAlchemy.
    # Initializes with a database type and provides methods to run queries and handle results.

    def __init__(self, db_type, dp_path):
        self.connection_strings = {
            "SQLite": f'sqlite:///{dp_path}'
            # "PostgreSQL": 'postgresql+psycopg2://username:password@host:port/database',
            # "MySQL": 'mysql+pymysql://username:password@host:port/database',
            # "SQL Server": 'mssql+pyodbc://username:password@host:port/database?driver=ODBC+Driver+17+for+SQL+Server'
        }
        self.db_type = db_type
        self.connection_string = self.connection_strings.get(self.db_type)
        if not self.connection_string:
            logger.error(f"Unsupported database type: {self.db_type}")
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def run_query(self, query):
        try:
            engine = create_engine(self.connection_string)
            with engine.connect() as connection:
                df = pd.read_sql(query, connection)
            return df
        except Exception as e:
            logger.error(f"Failed to run SQL query: {query}. Error: {e}")
            return pd.DataFrame()

    def execute_query(self, query):
        return self.run_query(query)


"""
   This function is specific to sqlite data base 
   given the db name, the function returns the sqlite connection for query execution

"""


def connect(db_path):
    """Establish a database connection."""
    try:
        print("Database connection established.")
        return sqlite3.connect(db_path)
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None


# This function is utilized by the pipeline to execute a specific SQL query.

def execute_sql(db_path, sql):
    engine = sqlite3.connect(db_path)
    try:
        result = pd.read_sql(sql, con=engine)
        return result

    except Exception as e:
        # logger.error(f"Failed to run SQL query: {e}")
        return None


# This function is used by the debugger prompt in pipeline to retrieve the error encountered while executing the SQL query.

def sql_exection_error(db_path, sql):
    engine = sqlite3.connect(db_path)
    try:
        result = pd.read_sql(sql, con=engine)
        return result
    except Exception as e:
        return e


# defining schema of the table for colgate database

def get_schema():
    return """\
0|Id|INTEGER eg. 1
1|fare_amount|REAL eg. 5.3
2|pickup_datetime|DATETIME eg. "2009-06-26 08:22:21"
3|passenger_count|INTEGER eg. 3
4|distance_miles|REAL eg. 12.5
5|pickup_statecode|TEXT eg. "NY"
6|pickup_statename|TEXT eg. "New York"
7|pickup_city|TEXT eg. "New York"
8|dropoff_statecode|TEXT eg. "NY"
9|dropoff_statename|TEXT eg. "New York"
10|dropoff_city|TEXT eg. "New York"
11|pickup_county|TEXT eg. "Kings"
12|dropoff_county|TEXT eg. "Kings"
"""

# Appends a semicolon to the SQL query if it doesn't already end with one


def validate_sql_termination(sql):
    if sql.strip()[-1] != ";":
        sql = sql.strip() + ";"
    return sql
