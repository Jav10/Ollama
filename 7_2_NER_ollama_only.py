from ollama import chat
from pydantic import BaseModel, Field
import json

class CarInfo(BaseModel):
    brand: str = Field(description="Marca del automóvil")
    model: str = Field(description="Modelo del automóvil")
    engine: str = Field(description="Motor del automóvil")
    years: str = Field(description="Años del automóvil")

class CarList(BaseModel):
    cars: list[CarInfo] = Field(description="Lista de autos extraídos del texto")

content = """
    Tu tarea es realizar Named Entity Recognition (NER) con el siguiente objetivo:

1. Identifica únicamente entidades existentes en el texto.
2. NO inventes información. Si no está en el texto, deja el campo vacío.
3. Devuelve exclusivamente JSON válido.
4. Sigue estos pasos internamente:
   - Paso 1: Analiza el texto.
   - Paso 2: Localiza menciones de marca, modelo, motor y años.
   - Paso 3: Normaliza los valores sin modificarlos.
   - Paso 4: Construye el JSON final.

Formato esperado:
{
{
  "brand": "",
  "model": "",
  "engine": "",
  "years": ""
},
{
  "brand": "",
  "model": "",
  "engine": "",
  "years": ""
}
}
    """

messages = [
    {"role": "system", "content": content},
    {"role": "user", "content": "BALATA FRENO DE DISCO WAGNER TRASERO PARA HYUNDAI ELANTRA GT 2017-2018, SONATA 2016-2017, VELOSTER 2016-2017, KIA OPTIMA 2016-2017"}
]

response = chat(model="deepseek-r1:8b", messages=messages, think=False)

print(response)
print(response["message"]["content"])
data = json.loads("["+response["message"]["content"]+']')


car_list = CarList(cars=data)

print(car_list)