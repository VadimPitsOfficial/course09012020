from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, IntegerField
from wtforms import validators


class CreateQuery(FlaskForm):

    product_name = StringField("Name: ", [validators.DataRequired("Please enter your name.")])

    product_model = StringField("Model: ", [
        validators.DataRequired("Please enter your model.")

    ])

    characteristic_price = IntegerField("Price: ", [
        validators.DataRequired("Please enter your price.")

    ])

    characteristic_specification = StringField("Specification: ", [
        validators.DataRequired("Please enter your specification.")

    ])

    submit = SubmitField("Search")

    def validate_on_submit(self):
        result = super(CreateQuery, self).validate()
        print(self.name.data.price)
        return result
