import mysql.connector
from mysql.connector import Error
import pandas as pd


def create_server_connection(host_name, user_name, user_password):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")

    return connection

def create_db_connection(host_name, user_name, user_password, db_name):
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("MySQL Database connection successful")
    except Error as err:
        print(f"Error: '{err}'")

    return connection


def execute_query(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")

def create_database(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Database created successfully")
    except Error as err:
        print(f"Error: '{err}'")

def read_query(connection, query):
    cursor = connection.cursor()
    result = None
    try:
        cursor.execute(query)
        result = cursor.fetchall()
        return result
    except Error as err:
        print(f"Error: '{err}'")

def execute_list_query(connection, sql, val):
    cursor = connection.cursor()
    try:
        cursor.executemany(sql, val)
        connection.commit()
        print("Query successful")
    except Error as err:
        print(f"Error: '{err}'")

create_LPRDATABASE_table = """
CREATE TABLE LPRDATABASE (
  Vehicle_No_Plate VARCHAR(30),
  Vehicle_Colour VARCHAR(30),
  first_name VARCHAR(40) NOT NULL,
  last_name VARCHAR(40) NOT NULL,
  IC_NO INT UNIQUE NOT NULL,
  phone_no VARCHAR(20)
);
"""

pop_LPRDATABASE = """
INSERT INTO LPRDATABASE VALUES
('PNX2513',  'Red', 'James', 'Tan', 12345678, 'NULL'),
('PQE4466', 'Silver',  'Stefanie',  'Lim', 44567842,  'NULL'), 
('PQX5674', 'Silver', 'Steve',  'Wang', '33456784', 'NULL'),
('PPE6548', 'Purple',  'Fredrick', 'Sim', '2214747', 'NULL'),
('PNE332', 'Black', 'Isable', 'Lau', '38291047',  NULL),
('WXY555', 'Yellow', 'Joshua', 'Lee', '34256781', 'NULL');
"""
connection = create_server_connection('sql12.freemysqlhosting.net', "sql12654624", "2GTpcCbLXC") # Connect to the Database
create_database_query = "CREATE DATABASE LPRDATABASE"
create_database(connection, create_database_query)
connection = create_db_connection("sql12.freemysqlhosting.net", "sql12654624", "2GTpcCbLXC", 'sql12654624') # Connect to the Database
execute_query(connection, create_LPRDATABASE_table) # Execute our defined query
execute_query(connection, pop_LPRDATABASE)

sql = '''
    INSERT INTO LPRDATABASE (Vehicle_No_Plate, Vehicle_Colour, first_name, last_name, IC_NO, phone_no) 
    VALUES (%s, %s, %s, %s, %s, %s)
    '''
    
val = [
    ('PNX2713', 'Red', 'Hank', 'Yeap',  13456782, 'NULL'), 
    ('PQE3590', 'Silver', 'Sue', 'Wong',  '55647892', 'NULL'),
    ('PQX2513', 'Silver', 'May', 'Khor',  '33445536', 'NULL')
]

connection = create_db_connection("sql12.freemysqlhosting.net", "sql12654624", "2GTpcCbLXC", 'sql12654624') # Connect to the Database
execute_list_query(connection, sql, val)
cursor = connection.cursor()
add_primary_key_query = "ALTER TABLE LPRDATABASE ADD COLUMN id INT AUTO_INCREMENT PRIMARY KEY"
cursor.execute(add_primary_key_query)
print("Primary key 'id' added successfully.")

update = """
UPDATE LPRDATABASE 
SET Vehicle_No_Plate = 'WXY555' 
WHERE Vehicle_No_Plate = 'WXY 555';

"""

connection = create_db_connection("sql12.freemysqlhosting.net", "sql12654624", "2GTpcCbLXC", 'sql12654624')
# execute_query(connection, update)

q1 = """
SELECT *
FROM LPRDATABASE;
"""

connection = create_db_connection("sql12.freemysqlhosting.net", "sql12654624", "2GTpcCbLXC", 'sql12654624') # Connect to the Database
results = read_query(connection, q1)



# Returns a list of lists and then creates a pandas DataFrame
from_db = []

for result in results:
  result = list(result)
  from_db.append(result)


columns = ["Vehicle_No_Plate", "Vehicle_Colour", "first_name", "last_name", "IC_NO", "phone_no", "id"]
df = pd.DataFrame(from_db, columns=columns)
print(df)
