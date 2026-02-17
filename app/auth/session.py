from flask_login import LoginManager, UserMixin

login_manager = LoginManager()
login_manager.login_view = "/login"

class User(UserMixin):
    def __init__(self, user_id, username):
        self.id = user_id
        self.username = username
