from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, HiddenField, IntegerField
from wtforms import validators
from wtforms.validators import NumberRange, DataRequired


class RecommendationFormEdit(FlaskForm):
    recommendation_id = HiddenField("id:")

    recommendation_name = StringField("Name: ", [
        validators.DataRequired("Please, enter name."),
        validators.Length(3, 15, "Product's name should be from 3 to 15 symbols")])

    recommendation_model = StringField("Model: ", [
        validators.DataRequired("Please, enter model."),
        validators.Length(2, 20, "Product's model should be from 2 to 20 symbols")
    ])

    recommendation_price = IntegerField("Price: ", validators=[NumberRange(min=1, message="Enter correct data"), DataRequired("Please enter your price.")])


    submit = SubmitField("Save")
