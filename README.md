# Basic-Movie-Recommendation-system
This repository contains implementation of a movie recommendation system with three different machine learning models to cater to different recommendation criteria of user preferance.
A common dataset was used for this recommendation system: Movielens 100k dataset.

## Recurrent Neural Network (RNN) Model
**Description:**
Recurrent Neural Networks (RNNs) are a type of neural network designed to recognize patterns in sequences of data such as time series, speech, text, etc. RNNs maintain a memory of previous inputs through their recurrent connections, making them suitable for tasks where context and sequential order are important. This model can be used in applications like natural language processing, stock price prediction, and music generation.

**Features:**

1. Sequence Modeling: Capable of handling sequential data and capturing temporal dependencies.
2. Hidden State: Maintains a memory of previous inputs through hidden states.
3. Weight Sharing: Uses shared weights across different time steps, reducing the number of parameters.
4. Backpropagation Through Time (BPTT): Training method that unrolls the network over time and applies backpropagation.
5. Variants: Includes Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) for improved long-term dependency handling.
6. Applications: Used in tasks such as language modeling, machine translation, and time series forecasting.

## Collaborative Neural Filtering (CNF) Model
**Description:**
Collaborative Neural Filtering (CNF) combines collaborative filtering and neural network techniques to recommend items. It leverages user-item interaction data and neural networks to learn latent representations of users and items. The model aims to predict a user's preference for items based on past interactions, capturing complex patterns and relationships in the data.

**Features:**

1. Latent Factor Model: Learns low-dimensional embeddings for users and items from interaction data.
2. Neural Network Architecture: Uses layers of neural networks to capture non-linear relationships between users and items.
3. User-Item Interactions: Utilizes implicit or explicit feedback (e.g., ratings, clicks) to train the model.
4. Generalization: Ability to generalize from observed interactions to unobserved ones, providing recommendations.
5. Hybrid Approach: Combines collaborative filtering's strength in leveraging interaction data with neural networks' ability to capture complex patterns.
6. Scalability: Designed to handle large-scale datasets with numerous users and items.

## Content-Based Filtering Model
**Description:**
Content-based filtering recommends items similar to those a user has shown interest in, based on item features. This method focuses on analyzing item attributes to suggest products that share common characteristics with previously liked items. For example, in movie recommendations, if a user likes a particular genre, actor, or director, the model will recommend other movies with similar attributes.

**Features:**

1. User Profile: A profile of user preferences based on previously interacted items.
2. Item Attributes: Features of items such as genre, director, actors, plot summary, etc.
3. Similarity Metrics: Measures such as cosine similarity or Euclidean distance to compare items.
4. Feature Engineering: Extraction and selection of important features to represent items.
5. Scalability: Efficient handling of large datasets and numerous features.
6. Personalization: Tailors recommendations to individual user tastes and preferences.

### Dependencies
TensorFlow, TensorFlow Datasets, Numpy, Pandas, Keras

### Author
Jai Karaneesh Sundar
