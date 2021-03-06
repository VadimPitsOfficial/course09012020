from flask_wtf import Form
from wtforms import StringField, SubmitField, FloatField, IntegerField
from wtforms import validators
from wtforms.validators import NumberRange, DataRequired


class CharacteristicForm(Form):
    characteristic_id = IntegerField("ID: ", validators=[NumberRange(min=1, message="Enter correct data"), DataRequired("Please enter your ID.")])

    characteristic_specification = StringField("Specification: ", [
        validators.DataRequired("Please, enter your specification."),
        validators.Length(4, 15, "Specification should be from 4 to 15 symbols")
    ])

    characteristic_price = IntegerField("Price: ", validators=[NumberRange(min=1, message="Enter correct data"), DataRequired("Please enter your price.")])
    submit = SubmitField("Save")