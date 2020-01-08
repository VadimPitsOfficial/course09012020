from flask_wtf import FlaskForm
from wtforms import HiddenField, SubmitField
from wtforms import validators


class StoreFormEdit(FlaskForm):
    store_name = HiddenField("Name: ", [
        validators.DataRequired("Please, enter name of store."),
        validators.Length(3, 20, "Name should be from 3 to 20 symbols")
    ])

    submit = SubmitField("Save")