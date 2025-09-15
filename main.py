import os
from flask import Flask, render_template
from flask_login import LoginManager
from flask_restful import Api

from db import db_session
from db.models.users import User

# Инициализация программы
app = Flask(__name__)
api = Api(app)
app.config['SECRET_KEY'] = 'yandexlyceum_secret_key'
login_manager = LoginManager()
login_manager.init_app(app)


@login_manager.user_loader
def load_user(user_id):
    db_sess = db_session.create_session()
    return db_sess.query(User).get(user_id)


@app.route("/")
def index():
    return render_template("index.html", title='NeedToFly', header=True)


def main():
    db_session.global_init("db/users_saved.db")
    db_session.create_session()

    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)


if __name__ == '__main__':
    main()