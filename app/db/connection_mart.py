from sqlalchemy import create_engine
from config.settings import (
    SQL_SERVER,
    SQL_DRIVER,
    DB_MART_NAME,
    SQL_USER,
    SQL_PASSWORD,
    IS_AZURE,
)


def get_mart_engine():
    if IS_AZURE:
        connection_string = (
            f"mssql+pyodbc://{SQL_USER}:{SQL_PASSWORD}@{SQL_SERVER}/{DB_MART_NAME}"
            f"?driver={SQL_DRIVER}"
            f"&Encrypt=yes"
            f"&TrustServerCertificate=no"
        )
    else:
        connection_string = (
            f"mssql+pyodbc://@{SQL_SERVER}/{DB_MART_NAME}"
            f"?driver={SQL_DRIVER}"
            f"&trusted_connection=yes"
            f"&TrustServerCertificate=yes"
        )
    return create_engine(connection_string)
