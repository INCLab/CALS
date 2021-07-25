import pandas as pd
from sqlalchemy import Table, Column, MetaData, create_engine, insert, select
from sqlalchemy.dialects.mysql import DOUBLE, DECIMAL, INTEGER

class tracking_db:
    db_user = 'tracking'
    db_pw = '1q2w3e4r!'
    db_host = 'localhost'
    db_name = 'tracking'
    db_encoding = 'utf8'

    def __init__(self):
        # Connect to Database Server
        self.engine = create_engine("mysql+mysqldb://" + self.db_user + ":" + self.db_pw + "@" + self.db_host + "/" + self.db_name,encoding=self.db_encoding)
        self.connection = self.engine.connect()
        self.metadata = MetaData(self.engine)

        # Create CSI Table (time (unix timestamp, primary key), _0 ~ _63 (double))
        self.CSITable = Table('csi', self.metadata,
                         Column('time', DECIMAL(20, 6), primary_key=True, nullable=False))

        for i in range(0, 64):
            self.CSITable.append_column(Column('_' + str(i), DOUBLE, nullable=False))

        # Create MOT Table (time (unix timestamp), count (int))
        self.MOTTable = Table('mot', self.metadata,
                         Column('time', DECIMAL(20, 3), primary_key=True, nullable=False),
                         Column('count', INTEGER, nullable=False))

        # Run Create All Table Command
        self.metadata.create_all()

    # Insert CSI Dataframe to Database
    def insert_csi(self, csi_df):
        csi_df.to_sql(name='csi', con=self.engine, if_exists='append', index=False)

    # Get CSI Table Data
    def get_csi(self):
        result = self.connection.execute(select(self.CSITable))
        return result.fetchall()

    # Get CSI dataframe
    def get_csi_df(self):
        csi_df = pd.read_sql_table('csi', self.connection)
        return csi_df

    # Insert MOT Data List to Database
    def insert_mot(self, mot_list):
        query = insert(self.MOTTable)
        result = self.connection.execute(query, mot_list)
        result.close()

    # Get MOT Table Data
    def get_mot(self):
        result = self.connection.execute(select(self.MOTTable))
        return result.fetchall()




