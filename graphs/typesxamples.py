from typing import TypedDict;

class Movie(TypedDict):
    title: str
    year: int
    director: str
    genre: str
    rating: float

movie = Movie( #What is difference between TypedDict and dataclass?
    title="Inception",
    year=2010,
    director="Christopher Nolan",
    genre="Science Fiction",
    rating=8.8
)


from typing import Union;

def square (value: Union[int, float]) -> float: # value can be either int or float
    """Returns the square of the given value."""
    return value ** 2


from typing import Optional;

def nice_message(name: Optional[str]) -> None:
    if name is None:
        print("Hello, stranger!")
    else:
        print(f"Hello, {name}!")


from typing import Any;

def print_anything(value: Any) -> None:
    """Prints any value passed to it."""
    print(value)


square = lambda x: x ** 2  # Using a lambda function to define square
square(5)  # Example usage of the square function


nums = [1, 2, 3, 4, 5]
squares = list(map(lambda x: x * x, nums))  # Using the square function with map