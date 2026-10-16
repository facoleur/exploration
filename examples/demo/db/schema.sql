-- Frozen demo snapshot. Status codes are cryptic on purpose: only the glossary explains them.
CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, country TEXT, segment TEXT);
CREATE TABLE products  (product_id INTEGER PRIMARY KEY, label TEXT, category TEXT, list_price REAL);
CREATE TABLE orders    (order_id INTEGER PRIMARY KEY, customer_id INTEGER, order_date TEXT, status TEXT);
CREATE TABLE order_lines (order_id INTEGER, product_id INTEGER, qty INTEGER, unit_price REAL);
CREATE TABLE shipments (shipment_id INTEGER PRIMARY KEY, order_id INTEGER, carrier TEXT, shipped_at TEXT, delivered_at TEXT);

INSERT INTO customers VALUES
 (1,'Boulangerie Martin','FR','B2B'), (2,'Alice Durand','FR','B2C'), (3,'Hotel Lumen','BE','B2B'),
 (4,'Marco Rossi','IT','B2C'), (5,'Café des Arts','FR','B2B'), (6,'Julie Petit','FR','B2C');

INSERT INTO products VALUES
 (1,'Moulin à café','equipment',120.0), (2,'Grains arabica 1kg','coffee',24.0),
 (3,'Grains robusta 1kg','coffee',18.0), (4,'Filtres x100','supplies',6.0), (5,'Machine expresso','equipment',890.0);

INSERT INTO orders VALUES
 (101,1,'2024-11-20','DLV'), (102,2,'2025-01-05','DLV'), (103,3,'2025-02-11','DLV'), (104,1,'2025-03-02','CNL'),
 (105,4,'2025-03-15','SHP'), (106,5,'2025-04-01','DLV'), (107,2,'2025-04-20','CRT'), (108,3,'2025-05-09','DLV'),
 (109,6,'2025-05-30','DLV'), (110,5,'2025-06-12','SHP');

-- unit_price is the price actually charged (after discount), it can differ from list_price.
INSERT INTO order_lines VALUES
 (101,2,10,22.0), (101,4,5,6.0), (102,1,1,120.0), (102,2,2,24.0), (103,5,1,850.0), (103,2,20,21.0),
 (104,5,1,890.0), (105,3,3,18.0), (106,2,15,22.0), (106,4,10,5.5), (107,1,1,120.0),
 (108,2,25,21.0), (108,3,10,17.0), (109,4,2,6.0), (110,5,1,870.0), (110,2,5,22.0);

INSERT INTO shipments VALUES
 (1,101,'Colissimo','2024-11-21','2024-11-23'), (2,102,'Colissimo','2025-01-06','2025-01-08'),
 (3,103,'DHL','2025-02-12','2025-02-14'), (4,105,'UPS','2025-03-16',NULL), (5,106,'DHL','2025-04-02','2025-04-03'),
 (6,108,'DHL','2025-05-10','2025-05-13'), (7,109,'DHL','2025-05-31','2025-06-02'), (8,110,'UPS','2025-06-13',NULL);
