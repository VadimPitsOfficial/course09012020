/*==============================================================*/
/* DBMS name:      PostgreSQL 9.x                               */
/* Created on:     24.10.2019 21:35:51                          */
/*==============================================================*/


drop index Characteristic_PK;

drop table Characteristic;

drop index Characteristic_Products2_FK;

drop index Characteristic_Products_FK;

drop index Characteristic_Products_PK;

drop table Characteristic_Products;

drop index Recommendation_Products_FK;

drop index Products_PK;

drop table Products;

drop index Products_Store2_FK;

drop index Products_Store_FK;

drop index Products_Store_PK;

drop table Products_Store;

drop index Recommendation_PK;

drop table Recommendation;

drop index Store_PK;

drop table Store;

/*==============================================================*/
/* Table: Characteristic                                        */
/*==============================================================*/
create table Characteristic (
   characteristic_id    INT4                 not null,
   characteristic_specification VARCHAR(40)          null,
   characteristic_price FLOAT10              null,
   constraint PK_CHARACTERISTIC primary key (characteristic_id)
);

/*==============================================================*/
/* Index: Characteristic_PK                                     */
/*==============================================================*/
create unique index Characteristic_PK on Characteristic (
characteristic_id
);

/*==============================================================*/
/* Table: Characteristic_Products                               */
/*==============================================================*/
create table Characteristic_Products (
   characteristic_id    INT4                 not null,
   product_id           INT4                 not null,
   constraint PK_CHARACTERISTIC_PRODUCTS primary key (characteristic_id, product_id)
);

/*==============================================================*/
/* Index: Characteristic_Products_PK                            */
/*==============================================================*/
create unique index Characteristic_Products_PK on Characteristic_Products (
characteristic_id,
product_id
);

/*==============================================================*/
/* Index: Characteristic_Products_FK                            */
/*==============================================================*/
create  index Characteristic_Products_FK on Characteristic_Products (
characteristic_id
);

/*==============================================================*/
/* Index: Characteristic_Products2_FK                           */
/*==============================================================*/
create  index Characteristic_Products2_FK on Characteristic_Products (
product_id
);

/*==============================================================*/
/* Table: Products                                              */
/*==============================================================*/
create table Products (
   product_id           INT4                 not null,
   recommendation_id    INT4                 null,
   product_name         VARCHAR(20)          null,
   product_model        VARCHAR(20)          null,
   constraint PK_PRODUCTS primary key (product_id)
);

/*==============================================================*/
/* Index: Products_PK                                           */
/*==============================================================*/
create unique index Products_PK on Products (
product_id
);

/*==============================================================*/
/* Index: Recommendation_Products_FK                            */
/*==============================================================*/
create  index Recommendation_Products_FK on Products (
recommendation_id
);

/*==============================================================*/
/* Table: Products_Store                                        */
/*==============================================================*/
create table Products_Store (
   product_id           INT4                 not null,
   store_name           VARCHAR(20)          not null,
   constraint PK_PRODUCTS_STORE primary key (product_id, store_name)
);

/*==============================================================*/
/* Index: Products_Store_PK                                     */
/*==============================================================*/
create unique index Products_Store_PK on Products_Store (
product_id,
store_name
);

/*==============================================================*/
/* Index: Products_Store_FK                                     */
/*==============================================================*/
create  index Products_Store_FK on Products_Store (
product_id
);

/*==============================================================*/
/* Index: Products_Store2_FK                                    */
/*==============================================================*/
create  index Products_Store2_FK on Products_Store (
store_name
);

/*==============================================================*/
/* Table: Recommendation                                        */
/*==============================================================*/
create table Recommendation (
   recommendation_id    INT4                 not null,
   recommendation_price FLOAT10              null,
   recommendation_name  VARCHAR(20)          null,
   recommendation_model VARCHAR(20)          null,
   constraint PK_RECOMMENDATION primary key (recommendation_id)
);

/*==============================================================*/
/* Index: Recommendation_PK                                     */
/*==============================================================*/
create unique index Recommendation_PK on Recommendation (
recommendation_id
);

/*==============================================================*/
/* Table: Store                                                 */
/*==============================================================*/
create table Store (
   store_name           VARCHAR(20)          not null,
   constraint PK_STORE primary key (store_name)
);

/*==============================================================*/
/* Index: Store_PK                                              */
/*==============================================================*/
create unique index Store_PK on Store (
store_name
);

alter table Characteristic_Products
   add constraint FK_CHARACTE_CHARACTER_CHARACTE foreign key (characteristic_id)
      references Characteristic (characteristic_id)
      on delete restrict on update restrict;

alter table Characteristic_Products
   add constraint FK_CHARACTE_CHARACTER_PRODUCTS foreign key (product_id)
      references Products (product_id)
      on delete restrict on update restrict;

alter table Products
   add constraint FK_PRODUCTS_RECOMMEND_RECOMMEN foreign key (recommendation_id)
      references Recommendation (recommendation_id)
      on delete restrict on update restrict;

alter table Products_Store
   add constraint FK_PRODUCTS_PRODUCTS__PRODUCTS foreign key (product_id)
      references Products (product_id)
      on delete restrict on update restrict;

alter table Products_Store
   add constraint FK_PRODUCTS_PRODUCTS__STORE foreign key (store_name)
      references Store (store_name)
      on delete restrict on update restrict;

