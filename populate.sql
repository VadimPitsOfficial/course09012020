insert into Characteristic (characteristic_id, characteristic_specification) values (1, 'White');

insert into Characteristic (characteristic_id, characteristic_specification) values (3, 'Gold');

insert into Characteristic (characteristic_id, characteristic_specification) values (2, 'Black');

insert into Characteristic (characteristic_id, characteristic_specification) values (4, 'Green');

insert into Characteristic (characteristic_id, characteristic_specification) values (0, 'Pink');

insert into Store (store_name) values ('Italy');

insert into Store (store_name) values ('Germany');

insert into Store (store_name) values ('USA');

insert into Store (store_name) values ('Japan');

insert into Store (store_name) values ('Ukraine');

insert into Recommendation (recommendation_id, recommendation_name, recommendation_model) values (3, 'Lexus', 'LX350');

insert into Recommendation (recommendation_id, recommendation_name, recommendation_model) values (0, 'BMW', 'X5');

insert into Recommendation (recommendation_id, recommendation_name, recommendation_model) values (4, 'Audi', 'A8');

insert into Recommendation (recommendation_id, recommendation_name, recommendation_model) values (2, 'ZAZ', 'Vida');

insert into Recommendation (recommendation_id, recommendation_name, recommendation_model) values (1, 'Mazda', 'Model 6');

insert into Products (product_id, recommendation_id, product_name, product_model) values (2, 2, 'Mazda', 'Model 6');

insert into Products (product_id, recommendation_id, product_name, product_model) values (4, 2, 'ZAZ', 'Vida');

insert into Products (product_id, recommendation_id, product_name, product_model) values (1, 0, 'Audi', 'A8');

insert into Products (product_id, recommendation_id, product_name, product_model) values (0, 4, 'BMW', 'X5');

insert into Products (product_id, recommendation_id, product_name, product_model) values (3, 3, 'Lexus', 'LX350');

insert into Characteristic_Products (characteristic_id, product_id) values (1, 2);

insert into Characteristic_Products (characteristic_id, product_id) values (3, 4);

insert into Characteristic_Products (characteristic_id, product_id) values (2, 1);

insert into Characteristic_Products (characteristic_id, product_id) values (4, 0);

insert into Characteristic_Products (characteristic_id, product_id) values (0, 3);

insert into Products_Store (product_id, store_name) values (2, 'Italy');

insert into Products_Store (product_id, store_name) values (4, 'Germany');

insert into Products_Store (product_id, store_name) values (1, 'USA');

insert into Products_Store (product_id, store_name) values (0, 'Japan');

insert into Products_Store (product_id, store_name) values (3, 'Ukraine');