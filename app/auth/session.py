from flask_login import LoginManager, UserMixin
from sqlalchemy import text
from app.db.connection_auth import get_auth_engine

login_manager = LoginManager()
login_manager.login_view = "/login"


class User(UserMixin):
    def __init__(self, user_id, username):
        self.id = user_id
        self.username = username


@login_manager.user_loader
def load_user(user_id):
    """Carga el usuario desde la BD para mantener la sesión activa."""
    engine = get_auth_engine()
    query = text("SELECT id, nombre FROM usuarios WHERE id = :id AND activo = 1")

    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"id": int(user_id)}).fetchone()
        if result:
            return User(result.id, result.nombre)
    except Exception:
        pass

    return None
