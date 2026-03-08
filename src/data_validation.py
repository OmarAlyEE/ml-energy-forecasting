import pandera as pa
from pandera import Column, DataFrameSchema

schema = DataFrameSchema({
    "Voltage": Column(float, nullable=False),
    "Global_intensity": Column(float, nullable=False),
    "Global_active_power": Column(float, nullable=False),
})

def validate_data(df):
    schema.validate(df, lazy=True)
