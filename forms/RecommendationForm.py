from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FloatField, IntegerField
from wtforms import validators
from wtforms.validators import NumberRange, DataRequired


class RecommendationForm(FlaskForm):
    recommendation_id = IntegerField("ID: ", validators=[NumberRange(min=0, message="Enter correct data"), DataRequired("Please enter your ID.")])

    recommendation_name = StringField("Recommendation Name: ", [
        validators.DataRequired("Please, enter name."),
        validators.Length(3, 15, "Name should be from 3 to 15 symbols")
    ])

    recommendation_model = StringField("Recommendation Model: ", [
        validators.DataRequired("Please, enter model."),
        validators.Length(2, 20, "Model should be from 2 to 20 symbols")
    ])

    recommendation_price = IntegerField("Price: ", validators=[NumberRange(min=0, message="Enter correct data"), DataRequired("Please enter your price.")])


    submit = SubmitField("Save")