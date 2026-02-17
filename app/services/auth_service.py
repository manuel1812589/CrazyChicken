import bcrypt
from sqlalchemy import text
from app.db.connection_auth import get_auth_engine


def authenticate_user(username: str, password: str):
    """
    Autentica un usuario verificando el hash bcrypt de la contraseña.
    Retorna el registro del usuario si es válido, None si no.
    """
    engine = get_auth_engine()

    query = text(
        """
        SELECT id, nombre, password_hash
        FROM usuarios
        WHERE nombre = :username
          AND activo = 1
    """
    )

    with engine.connect() as conn:
        result = conn.execute(query, {"username": username}).fetchone()

    if result is None:
        return None

    password_valida = bcrypt.checkpw(
        password.encode("utf-8"), result.password_hash.encode("utf-8")
    )

    if not password_valida:
        return None

    return result
