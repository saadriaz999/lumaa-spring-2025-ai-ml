from src import utils

if __name__ == "__main__":
    user_query = input("\n\n\nWhat are you looking to watch: ")
    recommendations = utils.run_recommendation_system(user_query, "data/movies.csv")
    print("\n\nTop movie recommendations:")
    for idx, (title, score) in enumerate(recommendations, 1):
        print(f"{idx}. {title} (similarity score: {score:.2f})")