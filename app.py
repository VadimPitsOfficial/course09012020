from flask import Flask, render_template, request, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from forms.ProductsForm import ProductsForm
from forms.StoreForm import StoreForm
from forms.ProductsFormEdit import ProductsFormEdit
from forms.RecommendationForm import RecommendationForm
from forms.CharacteristicForm import CharacteristicForm
from forms.StoreFormEdit import StoreFormEdit
from forms.RecommendationFormEdit import RecommendationFormEdit
from forms.CharacteristicFormEdit import CharacteristicFormEdit
from forms.RegistrationForm import RegistrationForm
from forms.LoginForm import LoginForm
from forms.SearchForm import CreateQuery
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from sqlalchemy.sql import func
import plotly
import plotly.graph_objs as go
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from neupy import algorithms

import json


app = Flask(__name__)
app.secret_key = 'key'

ENV = 'prod'

if ENV == 'dev':
    app.debug = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:blackjack21@localhost/Vadim_Pits'
else:
    app.debug = False
    app.config['SQLALCHEMY_DATABASE_URI'] = 'postgres://ofqdxhaxghbqqj:7f571b6b061e1b78a2f0d42af68cb78740ab37209da105584ba82bfad83e50fa@ec2-174-129-33-25.compute-1.amazonaws.com:5432/dflccpekfv3rhp'

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

list_product = []


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
    recommendation_price = db.Column(db.Integer)
    recommendation_name = db.Column(db.String(20))
    recommendation_model = db.Column(db.String(20))

    recommendation_products = db.relationship("Products")


class Characteristic(db.Model):
    __tablename__ = 'characteristic'
    characteristic_id = db.Column(db.Integer, primary_key=True)
    characteristic_specification = db.Column(db.String(40))
    characteristic_price = db.Column(db.Integer)

    characteristic_fk = db.relationship("Products", secondary="characteristic_products")


class Characteristic_Products(db.Model):
    __tablename__ = 'characteristic_products'
    left_name = db.Column(db.Integer, db.ForeignKey('characteristic.characteristic_id'), primary_key=True)
    right_name = db.Column(db.Integer, db.ForeignKey('products.product_id'), primary_key=True)


class User(db.Model):
    __tablename__ = 'user'

    user_email = db.Column(db.String(20), primary_key=True)
    user_password = db.Column(db.String(30), nullable=False)


# создание всех таблиц
# db.create_all()
#
# очистрка всех таблиц
#
db.session.query(Characteristic_Products).delete()
db.session.query(Products_Store).delete()
db.session.query(Characteristic).delete()
db.session.query(Products).delete()
db.session.query(Recommendation).delete()
db.session.query(Store).delete()
db.session.query(User).delete()


# создане объектов

Lexus = Products(product_id=1,
                 product_name='Lexus',
                 product_model='LX350'
                 )

BMW = Products(product_id=2,
               product_name='BMW',
               product_model='X5'
               )

Audi = Products(product_id=3,
                product_name='Audi',
                product_model='A8'
                )

ZAZ = Products(product_id=4,
               product_name='ZAZ',
               product_model='Vida'
               )

Mazda = Products(product_id=5,
                 product_name='Mazda',
                 product_model='Model 6'
                 )

R_Lexus = Recommendation(recommendation_id=1,
                         recommendation_price=30000,
                         recommendation_name='Lexus',
                         recommendation_model='LX350')

R_BMW = Recommendation(recommendation_id=2,
                       recommendation_price=20000,
                       recommendation_name='BMW',
                       recommendation_model='X5'
                       )

R_Audi = Recommendation(recommendation_id=3,
                        recommendation_price=40000,
                        recommendation_name='Audi',
                        recommendation_model='A8'
                        )

R_ZAZ = Recommendation(recommendation_id=4,
                       recommendation_price=4000,
                       recommendation_name='ZAZ',
                       recommendation_model='Vida'
                       )

R_Mazda = Recommendation(recommendation_id=5,
                         recommendation_price=10000,
                         recommendation_name='Mazda',
                         recommendation_model='Model 6'
                         )

Char_1 = Characteristic(characteristic_id=1,
                        characteristic_specification='White',
                        characteristic_price=30000
                        )

Char_2 = Characteristic(characteristic_id=2,
                        characteristic_specification='Black',
                        characteristic_price=20000
                        )

Char_3 = Characteristic(characteristic_id=3,
                        characteristic_specification='Gold',
                        characteristic_price=40000
                        )

Char_4 = Characteristic(characteristic_id=4,
                        characteristic_specification='Pink',
                        characteristic_price=4000
                        )

Char_5 = Characteristic(characteristic_id=5,
                        characteristic_specification='Grey',
                        characteristic_price=10000
                        )

St_name_1 = Store(store_name='Ukraine')

St_name_2 = Store(store_name='USA')

St_name_3 = Store(store_name='Germany')

St_name_4 = Store(store_name='Russia')

St_name_5 = Store(store_name='Japan')


Bob = User(
    user_email='bob@gmail.com',
    user_password='777777'
)

Boba = User(
    user_email='boba@gmail.com',
    user_password='111111'
)
Boban = User(
    user_email='boban@gmail.com',
    user_password='222222'
)

R_Lexus.recommendation_products.append(Lexus)
R_BMW.recommendation_products.append(BMW)
R_Audi.recommendation_products.append(Audi)
R_ZAZ.recommendation_products.append(ZAZ)
R_Mazda.recommendation_products.append(Mazda)

Lexus.products_store_fk.append(St_name_1)
BMW.products_store_fk.append(St_name_2)
Audi.products_store_fk.append(St_name_3)
ZAZ.products_store_fk.append(St_name_4)
Mazda.products_store_fk.append(St_name_5)

Lexus.products_characteristic_fk.append(Char_1)
BMW.products_characteristic_fk.append(Char_2)
Audi.products_characteristic_fk.append(Char_3)
ZAZ.products_characteristic_fk.append(Char_4)
Mazda.products_characteristic_fk.append(Char_5)

db.session.add_all([Lexus, BMW, Audi, ZAZ, Mazda,
                    R_Lexus, R_BMW, R_Audi, R_ZAZ, R_Mazda,
                    Char_1, Char_2, Char_3, Char_4, Char_5,
                    St_name_1, St_name_2, St_name_3, St_name_4, St_name_5,
                    Bob, Boba, Boban
                    ])

db.session.commit()

def dropSession():
    session['user_email'] = ''
    session['role'] = 'unlogged'

def newSession(email, pw):
    session['user_email'] = email
    if pw == '777777':
        session['role'] = 'admin'
    else:
        session['role'] = 'user'

@app.route('/')
def root():
    if not session['user_email']:
        return redirect('/login')
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()

    if request.method == 'POST':
        if form.validate():
            try:
                res = db.session.query(User).filter(User.user_email == form.user_email.data).one()
            except:
                form.user_email.errors = ['user doesnt exist']
                return render_template('login.html', form=form)
            if res.user_password == form.user_password.data:
                newSession(res.user_email, res.user_password)
                return redirect('/')
            else:
                form.user_password.errors = ['wrong password']
                return render_template('login.html', form=form)
        else:
            return render_template('login.html', form=form)
    else:
        return render_template('login.html', form=form)


@app.route('/logout')
def logout():
    dropSession()
    return redirect('/login')


@app.route('/registration', methods=['GET', 'POST'])
def registration():
    form = RegistrationForm()
    if request.method == 'POST':
        if form.validate():
            new_user = User(
                user_email=form.user_email.data,
                user_password=form.user_confirm_password.data
            )
            db.session.add(new_user)
            db.session.commit()
            newSession(new_user.user_email, new_user.user_password)
            return render_template('index.html')
        else:
            return render_template('registration.html', form=form)

    return render_template('registration.html', form=form)


@app.route('/products', methods=['GET'])
def all_products():
    if session['role'] == 'admin':
        result = db.session.query(Products).all()
        return render_template('all_products.html', result=result)
    else:
        return redirect('/')


@app.route('/store', methods=['GET'])
def all_store():
    if session['role'] == 'admin':
        result = db.session.query(Store).all()
        return render_template('all_store.html', result=result)
    else:
        return redirect('/')


@app.route('/recommendation', methods=['GET'])
def all_recommendation():
    if session['role'] == 'admin':
        result = db.session.query(Recommendation).all()
        return render_template('all_recommendation.html', result=result)
    else:
        return redirect('/')


@app.route('/characteristic', methods=['GET'])
def all_characteristic():
    if session['role'] == 'admin':
        result = db.session.query(Characteristic).all()
        return render_template('all_characteristic.html', result=result)
    else:
        return redirect('/')


@app.route('/create_product', methods=['POST', 'GET'])
def create_product():
    form = ProductsForm()

    if request.method == 'POST':
        if form.validate():
            new_product = Products(
                product_id=form.product_id.data,
                product_name=form.product_name.data,
                product_model=form.product_model.data,
            )
            try:
                db.session.add(new_product)
                db.session.commit()
                return redirect('/products')
            except:
                form.product_id.errors = ['this ID already exists']
                return render_template('create_product.html', form=form, form_name="New product",
                                       action="new_product")
        else:
            if not form.validate():
                return render_template('create_product.html', form=form, form_name="New product",
                                       action="new_product")

    elif request.method == 'GET':
        return render_template('create_product.html', form=form)


@app.route('/delete_product/<int:id>', methods=['GET', 'POST'])
def delete_product(id):
    result = db.session.query(Products).filter(Products.product_id == id).one()

    db.session.delete(result)
    db.session.commit()

    return redirect('/products')


@app.route('/create_store', methods=['POST', 'GET'])
def create_store():
    form = StoreForm()

    if request.method == 'POST':
        if form.validate():
            new_store = Store(
                store_name=form.store_name.data,
            )
            try:
                db.session.add(new_store)
                db.session.commit()
                return redirect('/store')
            except:
                form.store_name.errors = ['this name already exists']
                return render_template('create_store.html', form=form, form_name="New store",
                                       action="new_store")
        else:
            if not form.validate():
                return render_template('create_store.html', form=form, form_name="New store",
                                       action="new_store")
    elif request.method == 'GET':
        return render_template('create_store.html', form=form)


@app.route('/delete_store/<string:name>', methods=['GET', 'POST'])
def delete_store(name):
    result = db.session.query(Store).filter(Store.store_name == name).one()

    db.session.delete(result)
    db.session.commit()

    return redirect('/store')


@app.route('/create_recommendation', methods=['POST', 'GET'])
def create_recommendation():
    form = RecommendationForm()

    if request.method == 'POST':
        if form.validate():
            new_recommendation = Recommendation(
                recommendation_id=form.recommendation_id.data,
                recommendation_name=form.recommendation_name.data,
                recommendation_model=form.recommendation_model.data,
                recommendation_price=form.recommendation_price.data,
            )
            try:
                db.session.add(new_recommendation)
                db.session.commit()
                return redirect('/recommendation')
            except:
                form.recommendation_id.errors = ['this ID already exists']
                return render_template('create_recommendation.html', form=form, form_name="New recommendation",
                                       action="new_recommendation")
        else:
            if not form.validate():
                return render_template('create_recommendation.html', form=form, form_name="New recommendation",
                                       action="new_recommendation")
    elif request.method == 'GET':
        return render_template('create_recommendation.html', form=form)


@app.route('/delete_recommendation/<string:r_id>', methods=['GET', 'POST'])
def delete_recommendation(r_id):
    result = db.session.query(Recommendation).filter(Recommendation.recommendation_id == int(r_id)).one()

    db.session.delete(result)
    db.session.commit()

    return redirect('/recommendation')


@app.route('/create_characteristic', methods=['POST', 'GET'])
def create_characteristic():
    form = CharacteristicForm()

    if request.method == 'POST':
        if form.validate():
            new_characteristic = Characteristic(
                characteristic_id=form.characteristic_id.data,
                characteristic_price=form.characteristic_price.data,
                characteristic_specification=form.characteristic_specification.data,
            )
            try:
                db.session.add(new_characteristic)
                db.session.commit()
                return redirect('/characteristic')
            except:
                form.characteristic_id.errors = ['this ID already exists']
                return render_template('create_characteristic.html', form=form, form_name="New characteristic",
                                       action="new_characteristic")
        else:
            if not form.validate():
                return render_template('create_characteristic.html', form=form, form_name="New characteristic",
                                       action="new_characteristic")
    elif request.method == 'GET':
        return render_template('create_characteristic.html', form=form)



@app.route('/delete_characteristic/<int:c_id>', methods=['GET', 'POST'])
def delete_characteristic(c_id):
    result = db.session.query(Characteristic).filter(Characteristic.characteristic_id == c_id).one()

    db.session.delete(result)
    db.session.commit()

    return redirect('/characteristic')


@app.route('/edit_product/<int:id>', methods=['GET', 'POST'])
def edit_product(id):
    form = ProductsFormEdit()
    result = db.session.query(Products).filter(Products.product_id == id).one()

    if request.method == 'GET':

        form.product_id.data = result.product_id
        form.product_name.data = result.product_name
        form.product_model.data = result.product_model


        return render_template('edit_product.html', form=form, form_name=id)
    elif request.method == 'POST':
        if form.validate():
            result.product_id = form.product_id.data
            result.product_name = form.product_name.data
            result.product_model = form.product_model.data,

            db.session.commit()
            return redirect('/products')
        else:
            return render_template('edit_product.html', form=form, form_name=id)


@app.route('/edit_store/<string:name>', methods=['GET', 'POST'])
def edit_store(name):
    form = StoreFormEdit()
    result = db.session.query(Store).filter(Store.store_name == name).one()

    if request.method == 'GET':

        form.store_name.data = result.store_name


        return render_template('edit_store.html', form=form, form_name=name)
    elif request.method == 'POST':
        if form.validate():
            result.store_name = form.store_name.data
            db.session.commit()
            return redirect('/store')
        else: return render_template('edit_store.html', form=form, form_name=name)


@app.route('/edit_recommendation/<int:r_id>', methods=['GET', 'POST'])
def edit_recommendation(r_id):
    form = RecommendationFormEdit()
    result = db.session.query(Recommendation).filter(Recommendation.recommendation_id == r_id).one()

    if request.method == 'GET':

        form.recommendation_id.data = result.recommendation_id
        form.recommendation_price.data = result.recommendation_price
        form.recommendation_name.data = result.recommendation_name
        form.recommendation_model.data = result.recommendation_model

        return render_template('edit_recommendation.html', form=form, form_name='Edit Recommendation')
    elif request.method == 'POST':
        if form.validate():

            result.recommendation_id = form.recommendation_id.data
            result.recommendation_price = form.recommendation_price.data
            result.recommendation_name = form.recommendation_name.data
            result.recommendation_model = form.recommendation_model.data
            db.session.commit()
            return redirect('/recommendation')
        else:
            return render_template('edit_recommendation.html', form=form, form_name='Edit Recommendation')


@app.route('/edit_characteristic/<int:c_id>', methods=['GET', 'POST'])
def edit_characteristic(c_id):
    form = CharacteristicFormEdit()
    result = db.session.query(Characteristic).filter(Characteristic.characteristic_id == c_id).one()

    if request.method == 'GET':

        form.characteristic_id.data = result.characteristic_id
        form.characteristic_specification.data = result.characteristic_specification
        form.characteristic_price.data = result.characteristic_price

        return render_template('edit_characteristic.html', form=form, form_name='Edit Characteristic')
    elif request.method == 'POST':

        if form.validate():
            result.characteristic_id = form.characteristic_id.data
            result.characteristic_specification = form.characteristic_specification.data
            result.characteristic_price = form.characteristic_price.data,

            db.session.commit()
            return redirect('/characteristic')
        else:
            return render_template('edit_characteristic.html', form=form, form_name='Edit Characteristic')

@app.route('/search', methods=['GET', 'POST'])
def search():
    form = CreateQuery()
    if request.method == 'POST':
        if not form.validate():
            return render_template('Search.html', form=form, form_name="Search", action="search")
        else:
            list_product.clear()
            for id, name, model in db.session.query(Products.product_id, Products.product_name, Products.product_model):
                if name == form.product_name.data and model == form.product_model.data:
                    list_product.append(id)

            return redirect(url_for('searchList'))

    return render_template('Search.html', form=form, form_name="Search", action="search")


@app.route('/search/result')
def searchList():
    res = []
    try:
        for i in list_product:
            product_name, product_model, characteristic_price, characteristic_specification = db.session \
                .query(Products.product_name, Products.product_model, Characteristic.characteristic_price, Characteristic.characteristic_specification) \
                .join(Products, Characteristic.characteristic_id == Products.product_id) \
                .filter(Products.product_id == i).one()
            res.append(
                {"name": product_name, "model": product_model, "price": characteristic_price, "specification": characteristic_specification})
    except:
        print("no data")
    print(list_product)

    return render_template('Search_list.html', name="result", results=res, action="/search/result")


@app.route('/claster', methods=['GET', 'POST'])
def claster():
    df = pd.DataFrame()

    for product_name, characteristic_price in db.session.query(Products.product_name, Characteristic.characteristic_price).join(Characteristic, Products.product_id == Characteristic.characteristic_id):
        print(product_name, characteristic_price)
        df = df.append({"product_name": product_name, "characteristic_price": characteristic_price}, ignore_index=True)

    X = pd.get_dummies(data=df)
    print(X)
    count_clasters = len(df['characteristic_price'].unique())
    # print(count_clasters)
    kmeans = KMeans(n_clusters=count_clasters, random_state=0).fit(X)
    # print(kmeans)
    test_str = [10000, 'ZAZ']
    count_columns = len(X.columns)
    test_list = [0] * count_columns
    test_list[0] = 10000
    test_list[-1] = 1
    # print(test_list)
    # print(kmeans.labels_)
    # print(kmeans.predict(np.array([test_list])))

    return render_template('claster.html', row= kmeans.predict(np.array([test_list]))[0], count_claster = count_clasters, test_str=test_str)


@app.route('/corelation', methods=['GET', 'POST'])
def correlation():
    df = pd.DataFrame()
    for name, count_rec, avg_price in db.session.query(Recommendation.recommendation_name, func.count(Recommendation.recommendation_name), func.avg(Recommendation.recommendation_price)
                                                    ).group_by(Recommendation.recommendation_name):
        print(name, count_rec, avg_price)
        df = df.append({"count_rec": float(count_rec), "avg_price": float(avg_price)}, ignore_index=True)

    scaler = StandardScaler()
    scaler.fit(df[["count_rec"]])
    train_X = scaler.transform(df[["count_rec"]])
    print(train_X,df[["avg_price"]])
    reg = LinearRegression().fit(train_X, df[["avg_price"]])

    test_array = [[3]]
    test = scaler.transform(test_array)
    result = reg.predict(test)

    query1 = db.session.query(Recommendation.recommendation_name, func.count(Recommendation.recommendation_name), func.avg(Recommendation.recommendation_price)
                                                    ).group_by(Recommendation.recommendation_name).all()
    name, count_pr, count_fl = zip(*query1)
    scatter = go.Scatter(
        x=count_pr,
        y=count_fl,
        mode='markers',
        marker_color='rgba(255, 0, 0, 100)',
        name="data"
    )
    x_line = np.linspace(0, 10)
    y_line = x_line * reg.coef_[0, 0] + reg.intercept_[0]
    line = go.Scatter(
        x=x_line,
        y=y_line,
        mode='lines',
        marker_color='rgba(0, 0, 255, 100)',
        name="regretion"
    )
    data = [scatter, line]
    graphsJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return render_template('corelation.html', row=int(round(result[0, 0])), test_data=test_array[0][0],
                           coef=reg.coef_[0],
                           coef1=reg.intercept_, graphsJSON=graphsJSON)

@app.route('/clasification', methods=['GET', 'POST'])
def clasification():
    df = pd.DataFrame()
    for name, model, price in db.session.query(Products.product_name, Products.product_model, Recommendation.recommendation_price)\
            .join(Recommendation, Recommendation.recommendation_id == Products.recommendation_id):
        print(name, model, price)
        df = df.append({"name": name, "model": model, "price": float(price)}, ignore_index=True)
    # db.session.close()

    mean_p = df['price'].mean()
    df.loc[df['price'] < mean_p, 'quality'] = 0
    df.loc[df['price'] >= mean_p, 'quality'] = 1

    X = pd.get_dummies(data=df[['name', 'model']])
    print(df)
    print(X)
    pnn = algorithms.PNN(std=10, verbose=False)
    pnn.train(X, df['quality'])
    test_str = ['BMW', 'X5']
    count_columns = len(X.columns)
    test_list = [0] * count_columns
    test_list[1] = 1
    test_list[-1] = 1
    print(test_list)
    y_predicted = pnn.predict([test_list])
    result = "Ні"
    if y_predicted - 1 < 0.0000001:
        result = "Так"

    return render_template('clasification.html', y_predicted=result, test_data=test_list, test_str = test_str)

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    query1 = (
        db.session.query(
            Products.product_name,
            func.count(Recommendation.recommendation_model).label('recommendation_model')
        ).join(Recommendation, Products.recommendation_id == Recommendation.recommendation_id).
            group_by(Products.product_name)
    ).all()

    print(query1)

    query2 = (
        db.session.query(
            Products.product_model,
            func.count(Recommendation.recommendation_price).label('recommendation_price')
        ).join(Recommendation, Products.recommendation_id == Recommendation.recommendation_id).
            group_by(Products.product_model)
    ).all()

    print(query2)

    product_name, recommendation_id = zip(*query1)
    bar = go.Bar(
        x=product_name,
        y=recommendation_id
    )

    product_model, recommendation_price = zip(*query2)
    pie = go.Pie(
        labels=product_model,
        values=recommendation_price
    )

    data = {
        "bar": [bar],
        "pie": [pie]
    }
    graphs_json = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('dashboard.html', graphsJSON=graphs_json)



if __name__ == "__main__":
    app.debug = True
    app.run()