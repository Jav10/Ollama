from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
from langchain.agents import create_agent

# Named-Entity Recognition

# como ejemplo aplicar un system
# responder a Perla, es mejor usar system porque por contexto se puede perder la indicacion si lo pones en user ademas de que es mejor usarlo en system

class CarInfo(BaseModel):
    brand: str = Field(description="Marca del automóvil")
    model: str = Field(description="Modelo del automóvil")
    engine: str = Field(description="Motor del automóvil")
    years: str = Field(description="Años del automóvil")

class CarList(BaseModel):
    cars: list[CarInfo] = Field(description="Lista de autos extraídos del texto")

llm = ChatOllama(model="llama3.1:8b", temperature=0, validate_model_on_init=True)

agent = create_agent(
    model=llm,
    response_format=CarList
)


result = agent.invoke({
    "messages": [{"role": "system", "content": "Extract car information about brand, model, engine and years."},
        {"role": "user", "content": "BALATA FRENO DE DISCO WAGNER TRASERO PARA HYUNDAI ELANTRA GT 2017-2018, SONATA 2016-2017, VELOSTER 2016-2017, KIA OPTIMA 2016-2017"}]
})

print(result["structured_response"])
