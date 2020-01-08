from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
# подключение
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:blackjack21@localhost/Vadim_Pits'
# связь
db = SQLAlchemy(app)


class Products (db.Model):
    __tablename__ = 'products'
    product_id = db.Column(db.Integer, primary_key=True)
    product_name = db.Column(db.String(20))
    product_model = db.Column(db.String(20))

    products_store_fk = db.relationship("Store", secondary="products_store")
    products_characteristic_fk = db.relationship("Characteristic", secondary="characteristic_products")
    recommendation_id = db.Column(db.Integer, db.ForeignKey('recommendation.recommendation_id'))


class Store(db.Model):
    __tablename__ = 'store'
    store_name = db.Column(db.String(20), primary_key=True)

    store_fk = db.relationship("Products", secondary="products_store")


class Products_Store(db.Model):
    __tablename__ = 'products_store'
    left_name = db.Column(db.Integer, db.ForeignKey('products.product_id'), primary_key=True)
    right_name = db.Column(db.String(20), db.ForeignKey('store.store_name'), primary_key=True)


class Recommendation(db.Model):
    __tablename__ = 'recommendation'
    recommendation_id = db.Column(db.Integer, primary_key=True)
    recommendation_price = db.Column(db.Float)
    recommendation_name = db.Column(db.String(20))
    recommendation_model = db.Column(db.String(20))

    recommendation_products = db.relationship("Products")


class Characteristic(db.Model):
    __tablename__ = 'characteristic'
    characteristic_id = db.Column(db.Integer, primary_key=True)
    characteristic_specification = db.Column(db.String(40))
    characteristic_price = db.Column(db.Float)

    characteristic_fk = db.relationship("Products", secondary="characteristic_products")


class Characteristic_Products(db.Model):
    __tablename__ = 'characteristic_products'
    left_name = db.Column(db.Integer, db.ForeignKey('characteristic.characteristic_id'), primary_key=True)
    right_name = db.Column(db.Integer, db.ForeignKey('products.product_id'), primary_key=True)