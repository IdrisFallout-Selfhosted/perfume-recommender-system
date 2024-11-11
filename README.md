# Perfume Recommender System

This repository contains a Flask-based perfume recommender system. The system uses cosine similarity to generate recommendations based on a user's selected perfume.

## Features

- Recommend perfumes based on cosine similarity between scent profiles
- Filter recommendations by category and price
- Live search for perfumes

## Requirements

- Python 3.9 or later
- pip
- Docker

## Installation

1. Clone the repository:
```
git clone https://github.com/IdrisFallout-Selfhosted/perfume-recommender-system
```

2. Build the Docker image:
```
docker build -t idrisfallout/perfume_recommender_system .
```

3. Run the Docker container:
```
docker run -p 80:5000 idrisfallout/perfume_recommender_system
```

## Usage

### Home Page

The home page displays a dropdown list of all the perfumes in the dataset. Users can select a perfume from the list to get recommendations.

### Recommendation Endpoint

The `/recommend` endpoint takes a perfume name as input and returns a list of recommended perfumes.

```
POST /recommend
{
  "perfume_name": "Perfume Name"
}
```

### Search Endpoint

The `/search` endpoint takes a query as input and returns a list of perfumes that match the query.

```
GET /search
?q=Query
```

### Filter Recommendations Endpoint

The `/filter_recommendations` endpoint takes a category, minimum price, and maximum price as input and returns a list of perfumes that match the filters.

```
POST /filter_recommendations
{
  "category": "Category",
  "min_price": "Minimum Price",
  "max_price": "Maximum Price"
}
```

## Deployment

This application can be deployed using Docker Compose.

1. Create a `docker-compose.yml` file with the following content:

```
version: '3.3'

services:
  perfume_recommender_system:
    image: idrisfallout/perfume_recommender_system:1.0
    container_name: perfume_recommender_system
    environment:
      - TZ=Africa/Nairobi
    restart: unless-stopped
    ports:
      - "80:5000"
    networks:
      - my_network

networks:
  my_network:
    external: true
```

2. Run `docker-compose up -d` to deploy the application.

## License

This project is licensed under the MIT License.