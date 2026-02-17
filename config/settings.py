import os
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "Pollería Admin Dashboard"
DEBUG = os.getenv("DEBUG", "False") == "True"

SQL_SERVER = os.getenv("SQL_SERVER", "localhost")
SQL_DRIVER = os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server")

DB_AUTH_NAME = os.getenv("DB_AUTH_NAME", "polleria_admin")
DB_MART_NAME = os.getenv("DB_MART_NAME", "polleria_mart")

SQL_USER = os.getenv("SQL_USER", "")
SQL_PASSWORD = os.getenv("SQL_PASSWORD", "")

IS_AZURE = bool(SQL_USER and SQL_PASSWORD)
