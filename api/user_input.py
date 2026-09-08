from typing import Literal, Annotated
from pydantic import BaseModel, Field


class ShipmentInput(BaseModel):
    Warehouse_block: Annotated[Literal['A', 'B', 'C', 'D', 'F'], Field(..., description="The block where the warehouse is located")]
    Mode_of_Shipment: Annotated[Literal['Flight', 'Ship', 'Road'], Field(..., description="The mode of shipment used for delivery")]
    Customer_care_calls: Annotated[int, Field(..., ge=0, description="The number of customer care calls made by the customer")]
    Customer_rating: Annotated[int, Field(..., le=5, description="The rating given by the customer on a scale of 1 to 5")]
    Cost_of_the_Product: Annotated[float, Field(..., ge=1, description="The cost of the product in USD")]
    Prior_purchases: Annotated[int, Field(..., ge=0, description="The number of prior purchases made by the customer")]
    Product_importance: Annotated[Literal['low', 'medium', 'high'], Field(..., description="The importance of the product")]
    Discount_offered: Annotated[float, Field(..., lt=100, description="The discount offered on the product")]
    Weight_in_gms: Annotated[float, Field(..., ge=1, description="The weight of the product in grams")]
    Gender: Annotated[Literal['M', 'F'], Field(..., description="The gender of the customer")]
