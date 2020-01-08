from model import *

# создание всех таблиц
db.create_all()

# # очистрка всех таблиц

# db.session.query(Products).delete()
# db.session.query(Store).delete()
# db.session.query(Recommendation).delete()
# db.session.query(Characteristic).delete()
# db.session.query(characteristic_products).delete()
# db.session.query(products_store).delete()


# # # создане объектов

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
                    St_name_1, St_name_2, St_name_3, St_name_4, St_name_5
                    ])

db.session.commit()


@app.route('/')
def index():
    return "<h1 sttle='color: red'>hello flask</h1>"


if __name__ == "__main__":
    app.run()