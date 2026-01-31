from pydantic import BaseModel, Field
import common_utils_solar as utils

model = utils.get_solar_model(model_name="solar-pro")


class Movie(BaseModel):
    """상세한 영화정보."""

    title: str = Field(description="영화 제목")
    year: str = Field(description="개봉 연도")
    director: str = Field(description="영화 감독 이름")
    rating: float = Field(description="영화 평점(10점 만점)")


model_with_structure = model.with_structured_output(Movie)

result = model_with_structure.invoke("영화 'Incendies'에 대해서 알려줘")
print(result)

print(model.invoke("영화 'Incendies'에 대해서 알려줘"))
