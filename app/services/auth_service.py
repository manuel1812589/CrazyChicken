from sqlalchemy import text
from app.db.connection_auth import get_auth_engine

def authenticate_user(username: str, password: str):
    engine = get_auth_engine()

    query = text("""
        SELECT id, nombre
        FROM usuarios
        WHERE nombre = :username
          AND password = :password
    """)

    with engine.connect() as conn:
        result = conn.execute(
            query,
            {"username": username, "password": password}
        ).fetchone()

    return result
